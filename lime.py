#!/usr/bin/env python3
"""
lime.py

LIME for images, grid-superpixel variant (default 16x16), with OPTIONAL
off-manifold survival filtering reusing the calibration / scoring machinery
from off_manifold_filter.py.

Core LIME (standard implementation):
  Perturbs the image by turning grid cells "on" (kept sharp) or "off" (replaced
  by a blur reference, so perturbations stay on-manifold rather than gray-
  filled), fits a weighted linear surrogate of the target probability on the
  binary on/off vectors, and paints each cell with its surrogate coefficient.
  Per-segment constants (the known LIME limitation) -> blocky map by design.
  Vectorized: cell-id gather on-device, closed-form weighted ridge.

Optional survival filtering / weighting (four regimes, selected by flags):

  (A) --filter-modes NOT set
        Plain LIME. Evaluate exactly --n-samples perturbations, solve on all.

  (B) --filter-modes set, --soft-manifold NOT set, --full NOT set
        Evaluate exactly --n-samples perturbations ONCE. Each is filled with a
        mode drawn from --filter-modes. From the single forward pass take BOTH
        the penultimate feature (-> off-manifold score) and the target prob
        (-> LIME target). Keep only ON-manifold survivors (score < threshold);
        solve on those (<= n_samples). HARD rejection.

  (C) --filter-modes set, --soft-manifold NOT set, --full set
        Same per-sample procedure, but keep batch-sampling / filtering /
        accumulating until survivors reach --n-samples (hard draw cap guards
        against infinite loops). HARD rejection.

  (D) --filter-modes set, --soft-manifold set
        SOFT survival. Evaluate exactly --n-samples perturbations ONCE, keep
        ALL of them (no rejection, no --full grind), but down-weight off-
        manifold rows in the ridge. The manifold weight multiplies the LIME
        cosine kernel weight. Two shapes:
          step: w = 1 if on-manifold (score < thr) else --off-weight (e.g 0.1)
          soft: w = sigmoid(-(score - thr)/tau)   (no threshold cliff)
        'soft' escapes the quantile / pca-dim knife-edge that determines hard
        survival counts; 'step' is the literal "1 in / 0.1 out" baseline.

"Avoid rerun": the forward pass that yields the off-manifold score (penultimate
feature) is the SAME pass that yields the LIME target (logits/probs). The model
never sees the same perturbation twice; survivors carry their target straight
into the solver.

The all-on anchor (Z=1, full sharp image) is always included in the regression;
when filtering it is retained unconditionally and when soft-weighting it gets
full weight 1.0 (it is the clean input, so on-manifold by definition).

Standalone CLI version (no .base package): ResNet-50, --input image,
--calib-glob default sample_1k/*.JPEG, --grid default 16.

Requires: torch, torchvision, numpy, pillow, opencv-python, scikit-learn.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


# =============================================================================
# Model: ResNet-50. We need BOTH the penultimate 2048-d feature (off-manifold
# score) and the class logits (LIME target) from a single forward pass.
# =============================================================================
def build_model(device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.eval().to(device)

    feature_net = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
        net.avgpool, nn.Flatten(),
    ).eval().to(device)
    fc = net.fc.eval().to(device)

    # ResNet-50 V2 preprocessing constants (applied to a [0,1] CHW tensor).
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return feature_net, fc, mean, std


def normalize(x01, mean, std):
    """x01: (B,3,H,W) in [0,1] -> ImageNet-normalized."""
    return (x01 - mean) / std


@torch.no_grad()
def forward_feats_probs(feature_net, fc, x01, mean, std):
    """One forward pass on a [0,1] batch -> (feats[N,2048], probs[N,1000])."""
    feats = feature_net(normalize(x01, mean, std))
    probs = torch.softmax(fc(feats), dim=1)
    return feats, probs


# =============================================================================
# Blur reference (on-manifold "off" fill for standard LIME).
# =============================================================================
def blur_reference(x01, sigma):
    """Gaussian-blur a [0,1] (1,3,H,W) tensor. Separable depthwise conv."""
    radius = max(1, int(round(3 * sigma)))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, device=x01.device) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x01.dtype)
    C = x01.shape[1]
    kx = g.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
    ky = g.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
    xb = F.conv2d(x01, kx, padding=(0, radius), groups=C)
    xb = F.conv2d(xb, ky, padding=(radius, 0), groups=C)
    return xb


# =============================================================================
# Calibration (mirrors off_manifold_filter.py: PCA subspace + Gaussian).
# Operates on already-normalized batches via the feature_net.
# =============================================================================
@torch.no_grad()
def fit_calibration(feature_net, mean, std, calib_glob, device, batch_size,
                    pca_dim, work_res):
    paths = sorted(glob.glob(calib_glob))
    if not paths:
        raise FileNotFoundError(f"No calibration images matched: {calib_glob}")

    feats, batch = [], []

    def load01(p):
        img = Image.open(p).convert("RGB").resize((work_res, work_res),
                                                  Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W) in [0,1]

    for p in paths:
        batch.append(load01(p))
        if len(batch) == batch_size:
            xb = torch.stack(batch).to(device)
            feats.append(feature_net(normalize(xb, mean, std)).cpu().numpy())
            batch = []
    if batch:
        xb = torch.stack(batch).to(device)
        feats.append(feature_net(normalize(xb, mean, std)).cpu().numpy())

    feats = np.concatenate(feats, axis=0).astype(np.float64)
    n = feats.shape[0]
    k = min(pca_dim, n - 1)
    if k < 1:
        raise ValueError(f"Need >=2 calibration images for PCA, got n={n}.")
    pca = PCA(n_components=k, whiten=False, svd_solver="full").fit(feats)
    z = pca.transform(feats)
    evr = float(pca.explained_variance_ratio_.sum())

    mu = z.mean(axis=0)
    lw = LedoitWolf().fit(z - mu)
    precision = lw.precision_.astype(np.float64)

    centered = z - mu
    calib_d2 = np.einsum("ni,ij,nj->n", centered, precision, centered)
    recon = pca.inverse_transform(z)
    calib_resid = np.sum((feats - recon) ** 2, axis=1)

    return {"pca": pca, "mu": mu, "precision": precision,
            "calib_d2": np.sort(calib_d2), "calib_resid": np.sort(calib_resid),
            "n_calib": n, "k": k, "evr": evr}


def features_to_d2(feats, calib):
    z = calib["pca"].transform(feats.astype(np.float64))
    centered = z - calib["mu"]
    return np.einsum("ni,ij,nj->n", centered, calib["precision"], centered)


def features_to_resid(feats, calib):
    x = feats.astype(np.float64)
    z = calib["pca"].transform(x)
    recon = calib["pca"].inverse_transform(z)
    return np.sum((x - recon) ** 2, axis=1)


def features_to_score(feats, calib, score):
    if score == "mahalanobis":
        return features_to_d2(feats, calib)
    if score == "residual":
        return features_to_resid(feats, calib)
    if score == "combo":
        d2 = features_to_d2(feats, calib)
        rs = features_to_resid(feats, calib)
        d2n = d2 / max(np.median(calib["calib_d2"]), 1e-9)
        rsn = rs / max(np.median(calib["calib_resid"]), 1e-9)
        return d2n + rsn
    raise ValueError(f"unknown score: {score}")


def calib_scores(calib, score):
    if score == "mahalanobis":
        return calib["calib_d2"]
    if score == "residual":
        return calib["calib_resid"]
    if score == "combo":
        d2n = calib["calib_d2"] / max(np.median(calib["calib_d2"]), 1e-9)
        rsn = calib["calib_resid"] / max(np.median(calib["calib_resid"]), 1e-9)
        return np.sort(d2n + rsn)
    raise ValueError(f"no calibration distribution for score: {score}")


# =============================================================================
# Soft-manifold weight: maps an off-manifold score to a ridge weight in (0,1].
#   step: w = 1 if score < thr else off_weight     (literal "1 in / 0.1 out")
#   soft: w = sigmoid(-(score - thr)/tau)           (no threshold cliff)
# =============================================================================
def manifold_weight(sc, thr, mode, off_weight, tau):
    """sc: (B,) off-manifold scores. Returns (B,) weight in (0,1]."""
    sc = np.asarray(sc, dtype=np.float64)
    if mode == "step":
        w = np.where(sc < thr, 1.0, off_weight)
    else:  # soft
        t = tau if (tau is not None and tau > 0) else max(thr * 0.25, 1e-6)
        # sigmoid(-(sc-thr)/t): 0.5 at sc==thr, ->1 well below, ->0 well above.
        w = 1.0 / (1.0 + np.exp((sc - thr) / t))
    return w.astype(np.float64)


# =============================================================================
# Fill modes (identical semantics to off_manifold_filter.py), tensor versions.
# Each builds a [0,1] (1,3,H,W) "off" reference; per-sample modes (white_noise,
# inpaint) are generated at fill time.
# =============================================================================
FILL_MODES = ["blur", "black", "white", "inpaint", "corner_mean", "white_noise"]


def make_fill_variants(x01, sigma):
    """Return dict mode -> (1,3,H,W) [0,1] reference (None for per-sample)."""
    _, C, H, W = x01.shape
    v = {}
    v["blur"] = blur_reference(x01, sigma)
    v["black"] = torch.zeros_like(x01)
    v["white"] = torch.ones_like(x01)

    ch, cw = max(1, int(0.10 * H)), max(1, int(0.10 * W))
    corners = torch.cat([
        x01[..., :ch, :cw].reshape(1, C, -1),
        x01[..., :ch, -cw:].reshape(1, C, -1),
        x01[..., -ch:, :cw].reshape(1, C, -1),
        x01[..., -ch:, -cw:].reshape(1, C, -1),
    ], dim=2)
    cmean = corners.mean(dim=2).view(1, C, 1, 1)
    v["corner_mean"] = cmean.expand(1, C, H, W).contiguous()

    v["white_noise"] = None
    v["inpaint"] = None
    return v


def build_perturbations(x01, keep_pix, mode, variants, sigma, gen):
    """keep_pix: (B,1,H,W) in {0,1} (1=keep sharp). Returns (B,3,H,W) in [0,1]."""
    if mode == "white_noise":
        ref = torch.rand(keep_pix.shape[0], *x01.shape[1:], device=x01.device,
                         generator=gen)
    elif mode == "inpaint":
        # cv2.inpaint is per-image on uint8; do it on CPU then bring back.
        B = keep_pix.shape[0]
        H, W = x01.shape[2], x01.shape[3]
        base = (x01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        outs = []
        km = keep_pix[:, 0].cpu().numpy()  # (B,H,W)
        for b in range(B):
            inpaint_mask = ((1 - km[b]) * 255).astype(np.uint8)
            fimg = cv2.inpaint(base, inpaint_mask, 3, cv2.INPAINT_TELEA)
            outs.append(torch.from_numpy(fimg.astype(np.float32) / 255.0)
                        .permute(2, 0, 1))
        ref = torch.stack(outs).to(x01.device)
    else:
        ref = variants[mode]  # (1,3,H,W), broadcasts over B
    return keep_pix * x01 + (1 - keep_pix) * ref


# =============================================================================
# LIME core: cell-id map, sampling, weighted ridge.
# =============================================================================
def cell_id_map(H, W, grid, device):
    gh, gw = grid
    ys = (torch.arange(H, device=device) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W, device=device) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)  # (H,W) in [0, gh*gw)


def sample_Z(n, n_cells, mask_prob, gen, anchor=True):
    """Binary interpretable features: 1=keep sharp, 0=off. Standard LIME uses
    keep prob 0.5 (mask_prob=0.5). Optionally force row 0 = all-on anchor."""
    Z = (torch.rand(n, n_cells, generator=gen) > mask_prob).float()
    if anchor and n > 0:
        Z[0] = 1.0
    return Z


def weighted_ridge(Z, y, w, alpha=1.0):
    """Closed-form weighted ridge with unregularized intercept. Returns per-
    feature coefficients (intercept dropped)."""
    n, d = Z.shape
    Zb = np.concatenate([Z, np.ones((n, 1))], axis=1)
    Wd = w[:, None]
    A = Zb.T @ (Wd * Zb)
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0
    A += reg
    b = Zb.T @ (w * y)
    sol = np.linalg.solve(A, b)
    return sol[:-1], float(sol[-1])


def lime_weights(Znp, kernel_width):
    """Exponential kernel over cosine distance to the all-on vector."""
    all_on = np.ones(Znp.shape[1])
    d = 1.0 - (Znp @ all_on) / (
        np.linalg.norm(Znp, axis=1) * np.linalg.norm(all_on) + 1e-12)
    return np.exp(-(d ** 2) / (kernel_width ** 2))


# =============================================================================
# Driver.
# =============================================================================
@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device = {device}")
    feature_net, fc, mean, std = build_model(device)

    filtering = args.filter_modes is not None
    soft = filtering and args.soft_manifold  # (D) soft survival regime
    grid = (args.grid, args.grid)
    n_cells = grid[0] * grid[1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    gen_gpu = torch.Generator(device=device).manual_seed(args.seed)

    # Load input as [0,1] (1,3,H,W).
    img = Image.open(args.input).convert("RGB").resize(
        (args.work_res, args.work_res), Image.BILINEAR)
    x01 = torch.from_numpy(np.asarray(img, np.float32) / 255.0)\
        .permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, H, W = x01.shape

    cells = cell_id_map(H, W, grid, device)  # (H,W)
    variants = make_fill_variants(x01, args.sigma)

    # Clean top-1 = LIME target.
    _, clean_probs = forward_feats_probs(feature_net, fc, x01, mean, std)
    clean_probs = clean_probs.cpu().numpy()
    target = int(clean_probs[0].argmax())
    print(f"[*] LIME target = top-1 class {target} "
          f"(clean prob {clean_probs[0, target]:.4f})")

    calib, thr = None, None
    if filtering:
        print(f"[*] fitting calibration from {args.calib_glob} ...")
        calib = fit_calibration(feature_net, mean, std, args.calib_glob,
                                device, args.batch_size, args.pca_dim,
                                args.work_res)
        print(f"    n_calib={calib['n_calib']}  pca_k={calib['k']}  "
              f"evr={calib['evr']:.3f}")
        cs = calib_scores(calib, args.score)
        thr = float(np.quantile(cs, args.threshold_quantile))
        print(f"    [score={args.score}] on-manifold threshold "
              f"(q={args.threshold_quantile}) = {thr:.2f}")
        active_modes = args.filter_modes
        if soft:
            tau_disp = (args.weight_tau if (args.weight_tau is not None and
                        args.weight_tau > 0) else max(thr * 0.25, 1e-6))
            print(f"[*] SOFT-manifold ON — fill modes: "
                  f"{', '.join(active_modes)} | weight={args.manifold_weight} "
                  f"(off_weight={args.off_weight}, tau={tau_disp:.2f})")
        else:
            print(f"[*] HARD filtering ON — fill modes: "
                  f"{', '.join(active_modes)}")
    else:
        active_modes = [args.default_mask]
        print(f"[*] filtering OFF — plain LIME, {args.default_mask} reference "
              f"(sigma={args.sigma})")

    keep_Z, keep_y, keep_w = [], [], []
    total_eval, total_surv = 0, 0

    def process_batch(zb_cpu, force_keep_anchor=False):
        """zb_cpu: (B,n_cells) on CPU. One forward pass; reuse for score+target.
        force_keep_anchor: row 0 is the all-on anchor; in HARD mode retained
        unconditionally, in SOFT mode given full weight 1.0 (clean input ->
        on-manifold by definition)."""
        nonlocal total_eval, total_surv
        B = zb_cpu.shape[0]
        zb = zb_cpu.to(device)
        keep_pix = zb[:, cells].unsqueeze(1)  # (B,1,H,W)

        if filtering:
            mode = active_modes[int(torch.randint(len(active_modes), (1,),
                                                  generator=gen).item())]
        else:
            mode = args.default_mask
        comp = build_perturbations(x01, keep_pix, mode, variants, args.sigma,
                                   gen_gpu)
        feats, probs = forward_feats_probs(feature_net, fc, comp, mean, std)
        tgt = probs[:, target].cpu().numpy()
        total_eval += B

        # --- (D) SOFT: keep ALL rows, attach manifold weight ---------------
        if soft:
            sc = features_to_score(feats.cpu().numpy(), calib, args.score)
            wman = manifold_weight(sc, thr, args.manifold_weight,
                                   args.off_weight, args.weight_tau)
            if force_keep_anchor:
                wman[0] = 1.0  # clean input -> full weight
            for i in range(B):
                keep_Z.append(zb_cpu[i].numpy())
                keep_y.append(float(tgt[i]))
                keep_w.append(float(wman[i]))
            total_surv += int((sc < thr).sum())  # reference count only
            return B

        # --- (A) plain LIME: keep ALL rows, unit manifold weight -----------
        if not filtering:
            for i in range(B):
                keep_Z.append(zb_cpu[i].numpy())
                keep_y.append(float(tgt[i]))
                keep_w.append(1.0)
            return B

        # --- (B)/(C) HARD filter: keep only survivors ----------------------
        sc = features_to_score(feats.cpu().numpy(), calib, args.score)
        survived = sc < thr
        if force_keep_anchor:
            survived[0] = True  # anchor always retained
        s = 0
        for i in range(B):
            if survived[i]:
                keep_Z.append(zb_cpu[i].numpy())
                keep_y.append(float(tgt[i]))
                keep_w.append(1.0)
                s += 1
        total_surv += s
        return s

    # -------------------------------------------------------------------------
    # Dispatch. SOFT and plain both evaluate exactly n_samples once (no --full,
    # no rejection). HARD uses (B) or (C).
    # -------------------------------------------------------------------------
    if soft or not filtering:
        Z = sample_Z(args.n_samples, n_cells, args.mask_prob, gen, anchor=True)
        tag = "soft-manifold" if soft else "plain"
        print(f"[*] sampling {args.n_samples} perturbations "
              f"(grid={grid[0]}x{grid[1]}, {tag}) ...")
        for s in range(0, args.n_samples, args.batch_size):
            process_batch(Z[s:s + args.batch_size], force_keep_anchor=(s == 0))
        if soft:
            print(f"[*] soft-manifold: kept all {len(keep_Z)} rows; "
                  f"{total_surv}/{args.n_samples} would survive a hard thr "
                  f"(reference only).")

    elif not args.full:
        # (B) filter the FIRST n_samples only.
        Z = sample_Z(args.n_samples, n_cells, args.mask_prob, gen, anchor=True)
        print(f"[*] sampling {args.n_samples}, keeping on-manifold survivors "
              f"(no --full) ...")
        for s in range(0, args.n_samples, args.batch_size):
            process_batch(Z[s:s + args.batch_size], force_keep_anchor=(s == 0))
        print(f"[*] survivors: {total_surv} / {args.n_samples} on-manifold")

    else:
        # (C) --full: accumulate until n_samples survivors (with draw cap).
        cap = args.max_draws if args.max_draws > 0 \
            else args.n_samples * args.full_cap_factor
        print(f"[*] --full: accumulating until {args.n_samples} survivors "
              f"(draw cap {cap}) ...")
        first = True
        while total_surv < args.n_samples and total_eval < cap:
            b = min(args.batch_size, cap - total_eval)
            zb = sample_Z(b, n_cells, args.mask_prob, gen, anchor=first)
            process_batch(zb, force_keep_anchor=first)
            first = False
        surv_rate = total_surv / max(total_eval, 1)
        print(f"[*] --full done: {total_surv} survivors / {total_eval} "
              f"evaluated (survival rate {surv_rate:.3%}).")
        if surv_rate < 0.01:
            print(f"[WARN] survival rate {surv_rate:.3%} is very low — the "
                  f"survivor set is small and biased toward light masks. "
                  f"Consider --soft-manifold, a looser --threshold-quantile, "
                  f"or lighter --mask-prob.")
        if total_surv < args.n_samples:
            print(f"[WARN] hit draw cap {cap} with {total_surv} survivors; "
                  f"solving on those.")
        if total_surv > args.n_samples:
            keep_Z[:] = keep_Z[:args.n_samples]
            keep_y[:] = keep_y[:args.n_samples]
            keep_w[:] = keep_w[:args.n_samples]

    n_solve = len(keep_Z)
    if n_solve < 2:
        raise SystemExit(f"[FATAL] only {n_solve} perturbations to solve on. "
                         f"Loosen --threshold-quantile, change --score, raise "
                         f"--n-samples / --max-draws, or use --soft-manifold.")

    Znp = np.stack(keep_Z).astype(np.float64)
    y = np.asarray(keep_y, dtype=np.float64)

    # Final ridge weight = LIME cosine kernel * manifold weight (soft only).
    weights = lime_weights(Znp, args.kernel_width)
    wman_arr = np.asarray(keep_w, dtype=np.float64)
    if soft:
        weights = weights * wman_arr

    # Effective sample size of the weighted regression.
    n_eff = float(weights.sum() ** 2 / (np.sum(weights ** 2) + 1e-12))
    print(f"[*] solving weighted ridge on {n_solve} perturbations "
          f"(evaluated {total_eval} total, n_eff={n_eff:.1f}) ...")

    coefs, intercept = weighted_ridge(Znp, y, weights, alpha=args.ridge_alpha)

    # Paint coefficients back to pixels.
    coef_t = torch.tensor(coefs, dtype=torch.float32, device=device)
    attr = coef_t[cells].cpu().numpy()  # (H,W)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "attribution.npy"), attr)
    np.save(os.path.join(args.out_dir, "coefs.npy"),
            coefs.reshape(grid[0], grid[1]))

    norm = (attr - attr.min()) / (np.ptp(attr) + 1e-12)
    heat_u8 = (255 * norm).astype(np.uint8)
    Image.fromarray(heat_u8).save(os.path.join(args.out_dir, "heatmap.png"))
    cmap = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    base_u8 = (x01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = (0.5 * base_u8 + 0.5 * cmap).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(args.out_dir, "overlay.png"))

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"input: {args.input}\ntarget_class: {target}\n")
        f.write(f"grid: {grid[0]}x{grid[1]}\nfiltering: {filtering}\n")
        f.write(f"soft_manifold: {soft}\n")
        if filtering:
            f.write(f"filter_modes: {args.filter_modes}\n")
            f.write(f"score: {args.score}  threshold_q: "
                    f"{args.threshold_quantile}  thr: {thr:.4f}\n")
            if soft:
                f.write(f"manifold_weight: {args.manifold_weight}  "
                        f"off_weight: {args.off_weight}  "
                        f"weight_tau: {args.weight_tau}\n")
            else:
                f.write(f"full: {args.full}\n")
        f.write(f"n_samples_requested: {args.n_samples}\n")
        f.write(f"total_evaluated: {total_eval}\n")
        f.write(f"perturbations_in_solve: {n_solve}\n")
        f.write(f"n_eff: {n_eff:.4f}\n")
        if filtering:
            f.write(f"would_survive_hard_thr: {total_surv}\n")
        f.write(f"ridge_intercept: {intercept:.6f}\n")
    print(f"[*] wrote attribution.npy, coefs.npy, heatmap.png, overlay.png, "
          f"summary.txt to {args.out_dir}/")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Grid-LIME (ResNet-50) with optional off-manifold "
                    "filtering / soft weighting.")
    ap.add_argument("--input", required=True, help="Input image to explain.")
    ap.add_argument("--calib-glob", default="sample_1k/*.JPEG",
                    help="Calibration images (used only with --filter-modes).")
    ap.add_argument("--grid", type=int, default=16, help="Grid size (GxG).")
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="Plain/soft/no-full: perturbations evaluated. "
                         "--full: ON-manifold survivors required.")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--mask-prob", type=float, default=0.5,
                    help="Prob. a cell is turned OFF. 0.5 = standard LIME; "
                         "0.10-0.20 if you want lighter perturbations for the "
                         "manifold filter.")
    ap.add_argument("--sigma", type=float, default=11.0,
                    help="Gaussian sigma for the blur reference.")
    ap.add_argument("--kernel-width", type=float, default=0.25,
                    help="LIME exponential-kernel width (cosine distance).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)

    # Filtering / weighting.
    ap.add_argument("--filter-modes", nargs="+", default=None,
                    choices=FILL_MODES, metavar="MODE",
                    help="Enable off-manifold survival filtering and choose the "
                         "fill mode(s). A perturbation survives if ON-manifold "
                         "(score < threshold). If omitted: plain LIME.")
    ap.add_argument("--default-mask", default="blur", choices=FILL_MODES,
                    help="Fill mode for plain LIME (no --filter-modes).")

    # Soft-manifold (D).
    ap.add_argument("--soft-manifold", action="store_true",
                    help="Soft survival: keep ALL perturbations, down-weight "
                         "off-manifold ones instead of dropping. Requires "
                         "--filter-modes for the score; ignores --full.")
    ap.add_argument("--manifold-weight", default="soft",
                    choices=["step", "soft"],
                    help="step: w=1 if on-manifold else --off-weight. "
                         "soft: w=sigmoid(-(score-thr)/tau), no threshold "
                         "cliff (default).")
    ap.add_argument("--off-weight", type=float, default=0.1,
                    help="step mode: weight for off-manifold rows.")
    ap.add_argument("--weight-tau", type=float, default=None,
                    help="soft mode: sigmoid temperature. Default = thr*0.25.")

    # Hard filtering (B/C).
    ap.add_argument("--full", action="store_true",
                    help="With --filter-modes (HARD): keep sampling until "
                         "--n-samples survivors collected (C). Else filter "
                         "first --n-samples only (B). Ignored if "
                         "--soft-manifold.")
    ap.add_argument("--max-draws", type=int, default=-1,
                    help="--full cap on total evaluated "
                         "(-1 => n_samples * --full-cap-factor).")
    ap.add_argument("--full-cap-factor", type=int, default=20)

    ap.add_argument("--score", default="residual",
                    choices=["mahalanobis", "residual", "combo"])
    ap.add_argument("--threshold-quantile", type=float, default=0.95,
                    help="Calib-score quantile; perturbations BELOW it survive "
                         "(HARD) or sit at the soft-weight midpoint (SOFT).")
    ap.add_argument("--pca-dim", type=int, default=64)

    ap.add_argument("--work-res", type=int, default=224)
    ap.add_argument("--out-dir", default="lime_out")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())