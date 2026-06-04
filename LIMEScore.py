#!/usr/bin/env python3
"""
LIMEScore.py

STANDARD LIME (grid-superpixel, ResNet-50) + a SEPARATELY REPORTED
on-manifold reliability number, "LIMEScore".

Key design decision (deliberate):
  The LIME map is PLAIN standard LIME. The on-manifold scoring does NOT touch
  the surrogate, the weights, or the attribution in any way. Every perturbation
  is kept and weighted only by the usual LIME cosine kernel. The manifold
  machinery runs in parallel on the SAME forward pass and produces a single
  scalar diagnostic that rates how on-manifold the whole perturbation set was.

Why a single number:
  Stress-test observation (e.g. --mask-prob 0.8): a fill / mask-prob setting
  whose perturbations are, on average, MORE on-manifold yields a more reliable
  LIME map. So instead of per-row filtering/weighting (which inverts badly for
  adversarial-scatter fills like white_noise — the few masks that beat the
  threshold get promoted), we just MEASURE the average on-manifold-ness of all
  masks and report it alongside the (unmodified) map. Compare LIMEScore across
  fills / mask-probs to decide which explanation to trust.

LIMEScore, reported three ways (all over ALL perturbations incl. anchor):
  * frac_on_manifold  : mean over masks of 1[score < thr]   (0..1, HIGHER better)
                        -> the "probability score" / mean prob on-manifold.
  * mean_residual     : mean PCA-reconstruction residual     (LOWER better)
  * mean_mahalanobis  : mean Mahalanobis d^2 in PCA subspace (LOWER better)
  (mean of the active --score is also printed as the headline raw value.)

"Avoid rerun": the forward pass that yields the penultimate feature (-> score)
is the SAME pass that yields the target prob (-> LIME target). One pass per
perturbation, used for both the map and the diagnostic.

Standard-LIME knobs:  --default-mask (fill), --mask-prob, --grid, --n-samples,
                      --kernel-width, --ridge-alpha, --sigma.
Scoring knobs:        --calib-glob, --pca-dim, --score, --threshold-quantile.

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
# Model: ResNet-50. BOTH penultimate 2048-d feature (manifold score) and class
# logits (LIME target) come from a single forward pass.
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
# Blur reference.
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
# Calibration: PCA subspace + Ledoit-Wolf Gaussian on penultimate features.
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
        return torch.from_numpy(arr).permute(2, 0, 1)

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
    diff2 = np.sum((feats - recon) ** 2, axis=1)
    calib_resid = diff2
    # Energy-normalized residual: fraction of feature energy OUTSIDE the
    # subspace. Scale-free -> low-norm (black/white/const) fills can't win by
    # shrinking. calib_resid_norm in [0,1).
    fnorm2 = np.sum(feats ** 2, axis=1)
    calib_resid_norm = diff2 / np.maximum(fnorm2, 1e-12)
    # Angular residual: 1 - cos(f, recon). Also scale-free.
    dot = np.sum(feats * recon, axis=1)
    rnorm = np.sqrt(np.sum(recon ** 2, axis=1))
    fnorm = np.sqrt(fnorm2)
    calib_resid_ang = 1.0 - dot / np.maximum(fnorm * rnorm, 1e-12)

    return {"pca": pca, "mu": mu, "precision": precision,
            "calib_d2": np.sort(calib_d2), "calib_resid": np.sort(calib_resid),
            "calib_resid_norm": np.sort(calib_resid_norm),
            "calib_resid_ang": np.sort(calib_resid_ang),
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


def features_to_resid_norm(feats, calib):
    """Energy-normalized residual: ||f-recon||^2 / ||f||^2. Scale-free, so a
    low-norm collapsed feature (black/white/const fill) cannot score low just
    by shrinking -- numerator and denominator shrink together."""
    x = feats.astype(np.float64)
    z = calib["pca"].transform(x)
    recon = calib["pca"].inverse_transform(z)
    diff2 = np.sum((x - recon) ** 2, axis=1)
    fnorm2 = np.sum(x ** 2, axis=1)
    return diff2 / np.maximum(fnorm2, 1e-12)


def features_to_resid_ang(feats, calib):
    """Angular residual: 1 - cos(f, recon). Scale-free; pure direction."""
    x = feats.astype(np.float64)
    z = calib["pca"].transform(x)
    recon = calib["pca"].inverse_transform(z)
    dot = np.sum(x * recon, axis=1)
    fnorm = np.sqrt(np.sum(x ** 2, axis=1))
    rnorm = np.sqrt(np.sum(recon ** 2, axis=1))
    return 1.0 - dot / np.maximum(fnorm * rnorm, 1e-12)


def features_to_score(feats, calib, score):
    if score == "mahalanobis":
        return features_to_d2(feats, calib)
    if score == "residual":
        return features_to_resid(feats, calib)
    if score == "residual_norm":
        return features_to_resid_norm(feats, calib)
    if score == "residual_ang":
        return features_to_resid_ang(feats, calib)
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
    if score == "residual_norm":
        return calib["calib_resid_norm"]
    if score == "residual_ang":
        return calib["calib_resid_ang"]
    if score == "combo":
        d2n = calib["calib_d2"] / max(np.median(calib["calib_d2"]), 1e-9)
        rsn = calib["calib_resid"] / max(np.median(calib["calib_resid"]), 1e-9)
        return np.sort(d2n + rsn)
    raise ValueError(f"no calibration distribution for score: {score}")


# =============================================================================
# Fill modes (tensor versions). Per-sample modes (white_noise, inpaint) built
# at fill time; others precomputed.
# =============================================================================
FILL_MODES = ["blur", "black", "white", "inpaint", "corner_mean", "white_noise"]


def make_fill_variants(x01, sigma):
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
        B = keep_pix.shape[0]
        base = (x01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        outs = []
        km = keep_pix[:, 0].cpu().numpy()
        for b in range(B):
            inpaint_mask = ((1 - km[b]) * 255).astype(np.uint8)
            fimg = cv2.inpaint(base, inpaint_mask, 3, cv2.INPAINT_TELEA)
            outs.append(torch.from_numpy(fimg.astype(np.float32) / 255.0)
                        .permute(2, 0, 1))
        ref = torch.stack(outs).to(x01.device)
    else:
        ref = variants[mode]
    return keep_pix * x01 + (1 - keep_pix) * ref


# =============================================================================
# LIME core.
# =============================================================================
def cell_id_map(H, W, grid, device):
    gh, gw = grid
    ys = (torch.arange(H, device=device) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W, device=device) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)


def sample_Z(n, n_cells, mask_prob, gen, anchor=True):
    """1=keep sharp, 0=off. Standard LIME keep-prob = 1-mask_prob. Row 0=anchor."""
    Z = (torch.rand(n, n_cells, generator=gen) > mask_prob).float()
    if anchor and n > 0:
        Z[0] = 1.0
    return Z


def weighted_ridge(Z, y, w, alpha=1.0):
    """Closed-form weighted ridge, unregularized intercept."""
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

    cells = cell_id_map(H, W, grid, device)
    variants = make_fill_variants(x01, args.sigma)
    fill = args.default_mask
    print(f"[*] standard LIME — fill='{fill}'  mask_prob={args.mask_prob}  "
          f"grid={grid[0]}x{grid[1]}  sigma={args.sigma}")

    # Clean top-1 = LIME target.
    _, clean_probs = forward_feats_probs(feature_net, fc, x01, mean, std)
    clean_probs = clean_probs.cpu().numpy()
    target = int(clean_probs[0].argmax())
    print(f"[*] LIME target = top-1 class {target} "
          f"(clean prob {clean_probs[0, target]:.4f})")

    # Calibration for the on-manifold DIAGNOSTIC (does not affect the map).
    print(f"[*] fitting calibration from {args.calib_glob} "
          f"(pca_dim={args.pca_dim}, score={args.score}) ...")
    calib = fit_calibration(feature_net, mean, std, args.calib_glob, device,
                            args.batch_size, args.pca_dim, args.work_res)
    print(f"    n_calib={calib['n_calib']}  pca_k={calib['k']}  "
          f"evr={calib['evr']:.3f}")
    cs = calib_scores(calib, args.score)
    thr = float(np.quantile(cs, args.threshold_quantile))
    print(f"    [score={args.score}] on-manifold threshold "
          f"(q={args.threshold_quantile}) = {thr:.4f}")

    # Accumulators for the LIME map.
    keep_Z, keep_y = [], []
    # Accumulators for the on-manifold diagnostic (ALL perturbations).
    all_resid, all_d2, all_active = [], [], []
    all_resid_norm, all_resid_ang = [], []

    def process_batch(zb_cpu, force_anchor=False):
        """One forward pass; reuse for BOTH the LIME target and the manifold
        score. The score is recorded for the diagnostic ONLY — it never enters
        keep_Z / keep_y / the ridge weights."""
        B = zb_cpu.shape[0]
        zb = zb_cpu.to(device)
        keep_pix = zb[:, cells].unsqueeze(1)
        comp = build_perturbations(x01, keep_pix, fill, variants, args.sigma,
                                   gen_gpu)
        feats, probs = forward_feats_probs(feature_net, fc, comp, mean, std)
        tgt = probs[:, target].cpu().numpy()
        feats_np = feats.cpu().numpy()

        # --- LIME map (standard, all rows kept, no manifold influence) ------
        for i in range(B):
            keep_Z.append(zb_cpu[i].numpy())
            keep_y.append(float(tgt[i]))

        # --- on-manifold diagnostic (parallel, never touches the map) -------
        all_resid.append(features_to_resid(feats_np, calib))
        all_d2.append(features_to_d2(feats_np, calib))
        all_resid_norm.append(features_to_resid_norm(feats_np, calib))
        all_resid_ang.append(features_to_resid_ang(feats_np, calib))
        all_active.append(features_to_score(feats_np, calib, args.score))

    # Standard LIME sampling: exactly n_samples, row 0 = all-on anchor.
    Z = sample_Z(args.n_samples, n_cells, args.mask_prob, gen, anchor=True)
    print(f"[*] sampling {args.n_samples} perturbations ...")
    for s in range(0, args.n_samples, args.batch_size):
        process_batch(Z[s:s + args.batch_size], force_anchor=(s == 0))

    # ----- Fit the (plain) LIME surrogate -----------------------------------
    Znp = np.stack(keep_Z).astype(np.float64)
    y = np.asarray(keep_y, dtype=np.float64)
    weights = lime_weights(Znp, args.kernel_width)
    coefs, intercept = weighted_ridge(Znp, y, weights, alpha=args.ridge_alpha)

    # ----- LIMEScore: aggregate on-manifold reliability over ALL masks ------
    resid = np.concatenate(all_resid)
    d2 = np.concatenate(all_d2)
    resid_norm = np.concatenate(all_resid_norm)
    resid_ang = np.concatenate(all_resid_ang)
    active = np.concatenate(all_active)
    n_all = active.shape[0]

    # frac_on_manifold == the old hard-filter "survival rate": fraction of
    # masks with (active score < thr). It is computed on whichever --score is
    # active, so switch --score to residual_norm / mahalanobis to get a survival
    # rate that is NOT confounded by feature energy.
    frac_on_manifold = float(np.mean(active < thr))   # == survival rate
    mean_residual = float(resid.mean())               # LOWER better (confounded)
    mean_mahalanobis = float(d2.mean())               # LOWER better
    mean_resid_norm = float(resid_norm.mean())        # LOWER better (de-confounded)
    mean_resid_ang = float(resid_ang.mean())          # LOWER better (de-confounded)
    mean_active = float(active.mean())                # headline raw, LOWER better
    median_active = float(np.median(active))

    print("")
    print("==================== LIMEScore (on-manifold reliability) ==========")
    print(f"  fill mode               : {fill}")
    print(f"  mask_prob               : {args.mask_prob}")
    print(f"  perturbations scored     : {n_all}")
    print(f"  -- probability score / SURVIVAL RATE (HIGHER = more reliable) --")
    print(f"  frac_on_manifold         : {frac_on_manifold:.4f}   "
          f"(= survival rate on '{args.score}', mean[score < thr])")
    print(f"  -- raw scores (LOWER = more reliable) --")
    print(f"  mean_residual            : {mean_residual:.4f}   "
          f"(RAW resid — energy-confounded, do NOT use cross-fill)")
    print(f"  mean_residual_norm       : {mean_resid_norm:.6f}   "
          f"(||f-recon||^2/||f||^2 — scale-free)")
    print(f"  mean_residual_ang        : {mean_resid_ang:.6f}   "
          f"(1-cos(f,recon) — scale-free)")
    print(f"  mean_mahalanobis (d^2)   : {mean_mahalanobis:.4f}")
    print(f"  mean_{args.score:<18}: {mean_active:.6f}   (headline raw)")
    print(f"  median_{args.score:<16}: {median_active:.6f}")
    print("===================================================================")
    print("")

    # ----- Paint coefficients back to pixels (standard LIME map) ------------
    coef_t = torch.tensor(coefs, dtype=torch.float32, device=device)
    attr = coef_t[cells].cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    # Tag every artifact with fill mode + mask_prob so a fill/mask-prob sweep
    # writing to one out-dir does not clobber itself. e.g. overlay__blur_mp0.80.png
    tag = f"{fill}_mp{args.mask_prob:.2f}"

    def op(name, ext):
        return os.path.join(args.out_dir, f"{name}__{tag}.{ext}")

    np.save(op("attribution", "npy"), attr)
    np.save(op("coefs", "npy"), coefs.reshape(grid[0], grid[1]))
    np.save(op("manifold_scores", "npy"), active)

    norm = (attr - attr.min()) / (np.ptp(attr) + 1e-12)
    heat_u8 = (255 * norm).astype(np.uint8)
    Image.fromarray(heat_u8).save(op("heatmap", "png"))
    cmap = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    base_u8 = (x01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = (0.5 * base_u8 + 0.5 * cmap).astype(np.uint8)
    Image.fromarray(overlay).save(op("overlay", "png"))

    # Score-distribution figure (diagnostic, not the map).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(active, bins=60, alpha=0.7, label=f"perturbations ({fill})")
        ax.hist(cs, bins=60, alpha=0.5, density=False, label="calibration",
                weights=np.full_like(cs, n_all / max(len(cs), 1)))
        ax.axvline(thr, color="k", ls="--", lw=1,
                   label=f"thr (q={args.threshold_quantile})")
        ax.set_xlabel(f"{args.score} score (lower = more on-manifold)")
        ax.set_ylabel("count")
        ax.set_title(f"LIMEScore: frac_on_manifold={frac_on_manifold:.3f}  "
                     f"mean={mean_active:.2f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(op("score_hist", "png"), dpi=120)
        plt.close(fig)
        wrote_hist = True
    except Exception as e:
        wrote_hist = False
        print(f"[warn] score histogram skipped ({e})")

    with open(op("summary", "txt"), "w") as f:
        f.write(f"input: {args.input}\ntarget_class: {target}\n")
        f.write(f"grid: {grid[0]}x{grid[1]}\nfill: {fill}\n")
        f.write(f"mask_prob: {args.mask_prob}\n")
        f.write(f"n_samples: {args.n_samples}\n")
        f.write(f"kernel_width: {args.kernel_width}  ridge_alpha: "
                f"{args.ridge_alpha}\n")
        f.write(f"ridge_intercept: {intercept:.6f}\n")
        f.write("--- on-manifold diagnostic (does NOT affect the map) ---\n")
        f.write(f"calib_glob: {args.calib_glob}\n")
        f.write(f"pca_dim: {calib['k']}  evr: {calib['evr']:.4f}\n")
        f.write(f"score: {args.score}  threshold_q: "
                f"{args.threshold_quantile}  thr: {thr:.6f}\n")
        f.write(f"LIMEScore.frac_on_manifold: {frac_on_manifold:.6f}\n")
        f.write(f"LIMEScore.mean_residual: {mean_residual:.6f}\n")
        f.write(f"LIMEScore.mean_residual_norm: {mean_resid_norm:.6f}\n")
        f.write(f"LIMEScore.mean_residual_ang: {mean_resid_ang:.6f}\n")
        f.write(f"LIMEScore.mean_mahalanobis: {mean_mahalanobis:.6f}\n")
        f.write(f"LIMEScore.mean_{args.score}: {mean_active:.6f}\n")
        f.write(f"LIMEScore.median_{args.score}: {median_active:.6f}\n")
    outs = (f"attribution__{tag}.npy, coefs__{tag}.npy, "
            f"manifold_scores__{tag}.npy, heatmap__{tag}.png, "
            f"overlay__{tag}.png, summary__{tag}.txt")
    if wrote_hist:
        outs += f", score_hist__{tag}.png"
    print(f"[*] wrote {outs} to {args.out_dir}/")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Standard grid-LIME (ResNet-50) + reported on-manifold "
                    "LIMEScore (map is plain LIME; score is diagnostic only).")
    ap.add_argument("--input", required=True, help="Input image to explain.")

    # ---- Standard LIME knobs ----
    ap.add_argument("--default-mask", default="blur", choices=FILL_MODES,
                    help="Fill / reference used for OFF cells (standard LIME).")
    ap.add_argument("--mask-prob", type=float, default=0.5,
                    help="Prob. a cell is turned OFF. 0.5 = standard LIME. "
                         "Raise (e.g. 0.8) for a heavier-perturbation stress "
                         "test; LIMEScore will report how on-manifold it stays.")
    ap.add_argument("--grid", type=int, default=16, help="Grid size (GxG).")
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="Number of perturbations evaluated (all kept).")
    ap.add_argument("--kernel-width", type=float, default=0.25,
                    help="LIME exponential-kernel width (cosine distance).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=11.0,
                    help="Gaussian sigma for the blur reference.")

    # ---- On-manifold scoring knobs (diagnostic only) ----
    ap.add_argument("--calib-glob", default="sample_1k/*.JPEG",
                    help="Calibration images for the on-manifold score.")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="PCA subspace dim for the manifold score.")
    ap.add_argument("--score", default="residual",
                    choices=["mahalanobis", "residual", "residual_norm",
                             "residual_ang", "combo"],
                    help="Headline raw score; frac_on_manifold (survival rate) "
                         "uses this too. NOTE: raw 'residual' is energy-"
                         "confounded and inverts across fills — use "
                         "'residual_norm' or 'mahalanobis' for cross-fill "
                         "comparison.")
    ap.add_argument("--threshold-quantile", type=float, default=0.95,
                    help="Calib-score quantile defining the on-manifold thr "
                         "used for frac_on_manifold.")

    # ---- Misc ----
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--work-res", type=int, default=224)
    ap.add_argument("--out-dir", default="limescore_out")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())