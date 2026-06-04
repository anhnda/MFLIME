#!/usr/bin/env python3
"""
lime.py

LIME for image classification on ResNet-50, with optional off-manifold
survival filtering reusing the calibration / scoring machinery from
off_manifold_filter.py.

Three regimes, selected by flags:

  (A) --filter-modes NOT set
        Plain LIME. Sample exactly --n-samples masks, fill masked-out cells
        (default: blur), run the model, solve the linear surrogate on all of
        them. No manifold filtering.

  (B) --filter-modes set, --full NOT set
        Sample exactly --n-samples masks ONCE. Each mask is filled with the
        fill mode it was assigned (drawn from the chosen --filter-modes). Run
        the model once per mask; from that single forward pass take BOTH the
        penultimate feature (-> off-manifold score) and the class logits
        (-> LIME target). Keep only ON-manifold (survivor) masks, i.e. those
        with score < threshold. Solve LIME on the survivors only. The total
        masks evaluated equals --n-samples (survivors <= n-samples).

  (C) --filter-modes set, --full set
        Same per-mask procedure, but keep batch-sampling / filtering /
        accumulating until the number of survivors reaches --n-samples (then
        solve on exactly --n-samples survivors). A hard cap on total draws
        prevents an infinite loop if almost nothing survives.

Key efficiency point ("avoid rerun"): the model forward pass that produces the
off-manifold score (penultimate feature) is the SAME pass that produces the
LIME target (logits). We never run the model twice on the same masked image;
survivors are handed directly to the solver with their already-computed target.

Defaults: ResNet-50 (IMAGENET1K_V2), --calib-glob sample_1k/*.JPEG, --grid 16.
LIME explains the clean image's top-1 class.

Requires: torch, torchvision, numpy, pillow, opencv-python, scikit-learn.
"""

import argparse
import glob
import os

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


# -----------------------------------------------------------------------------
# Model: ResNet-50. We need BOTH the penultimate 2048-d feature (for the
# off-manifold score) and the class logits (for the LIME target), from a single
# forward pass. So we run the truncated feature net, then apply fc ourselves.
# -----------------------------------------------------------------------------
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
    preprocess = weights.transforms()
    return feature_net, fc, preprocess


@torch.no_grad()
def forward_feats_and_probs(feature_net, fc, batch, device):
    """One forward pass -> (penultimate features [N,2048], class probs [N,1000])."""
    feats = feature_net(batch.to(device))               # (N, 2048)
    logits = fc(feats)                                  # (N, 1000)
    probs = torch.softmax(logits, dim=1)
    return feats.cpu().numpy(), probs.cpu().numpy()


# -----------------------------------------------------------------------------
# Calibration (mirrors off_manifold_filter.py: PCA subspace + Gaussian +
# residual / Mahalanobis scores).
# -----------------------------------------------------------------------------
@torch.no_grad()
def fit_calibration(feature_net, preprocess, calib_glob, device, batch_size,
                    pca_dim):
    paths = sorted(glob.glob(calib_glob))
    if not paths:
        raise FileNotFoundError(f"No calibration images matched: {calib_glob}")

    feats, batch = [], []
    for p in paths:
        batch.append(preprocess(Image.open(p).convert("RGB")))
        if len(batch) == batch_size:
            feats.append(feature_net(torch.stack(batch).to(device)).cpu().numpy())
            batch = []
    if batch:
        feats.append(feature_net(torch.stack(batch).to(device)).cpu().numpy())

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

    return {
        "pca": pca, "mu": mu, "precision": precision,
        "calib_d2": np.sort(calib_d2),
        "calib_resid": np.sort(calib_resid),
        "n_calib": n, "k": k, "evr": evr,
    }


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
    elif score == "residual":
        return features_to_resid(feats, calib)
    elif score == "combo":
        d2 = features_to_d2(feats, calib)
        rs = features_to_resid(feats, calib)
        d2n = d2 / max(np.median(calib["calib_d2"]), 1e-9)
        rsn = rs / max(np.median(calib["calib_resid"]), 1e-9)
        return d2n + rsn
    raise ValueError(f"unknown score: {score}")


def calib_scores(calib, score):
    if score == "mahalanobis":
        return calib["calib_d2"]
    elif score == "residual":
        return calib["calib_resid"]
    elif score == "combo":
        d2n = calib["calib_d2"] / max(np.median(calib["calib_d2"]), 1e-9)
        rsn = calib["calib_resid"] / max(np.median(calib["calib_resid"]), 1e-9)
        return np.sort(d2n + rsn)
    raise ValueError(f"no calibration distribution for score: {score}")


# -----------------------------------------------------------------------------
# Fill modes (identical semantics to off_manifold_filter.py).
# -----------------------------------------------------------------------------
FILL_MODES = ["blur", "black", "white", "inpaint", "corner_mean", "white_noise"]


def make_fill_variants(img_u8, blur_var=50.0):
    H, W = img_u8.shape[:2]
    variants = {}
    sigma = float(np.sqrt(blur_var))
    k = max(3, int(2 * round(3 * sigma) + 1))
    variants["blur"] = cv2.GaussianBlur(img_u8, (k, k), sigmaX=sigma, sigmaY=sigma)
    variants["black"] = np.zeros_like(img_u8)
    variants["white"] = np.full_like(img_u8, 255)

    ch, cw = max(1, int(0.10 * H)), max(1, int(0.10 * W))
    corners = np.concatenate([
        img_u8[:ch, :cw].reshape(-1, 3),
        img_u8[:ch, -cw:].reshape(-1, 3),
        img_u8[-ch:, :cw].reshape(-1, 3),
        img_u8[-ch:, -cw:].reshape(-1, 3),
    ], axis=0)
    variants["corner_mean"] = np.broadcast_to(
        corners.mean(axis=0).astype(np.uint8), img_u8.shape).copy()

    variants["white_noise"] = None  # per-sample
    variants["inpaint"] = None      # per-sample (needs the mask)
    return variants


def cell_mask_to_pixel(cell_mask, H, W, grid):
    """Upsample (grid,grid) binary cell mask to (H,W). 1=keep, 0=masked-out."""
    pix = np.kron(cell_mask, np.ones((int(np.ceil(H / grid)),
                                      int(np.ceil(W / grid))), dtype=np.uint8))
    return pix[:H, :W]


def apply_mask(img_u8, pix_mask, mode, variants):
    keep = pix_mask[..., None].astype(bool)
    if mode == "white_noise":
        fill = np.random.randint(0, 256, size=img_u8.shape, dtype=np.uint8)
    elif mode == "inpaint":
        inpaint_mask = (1 - pix_mask).astype(np.uint8) * 255
        fill = cv2.inpaint(img_u8, inpaint_mask, inpaintRadius=3,
                           flags=cv2.INPAINT_TELEA)
    else:
        fill = variants[mode]
    return np.where(keep, img_u8, fill)


# -----------------------------------------------------------------------------
# LIME core.
# -----------------------------------------------------------------------------
def lime_kernel_weights(cell_masks, mask_prob):
    """Cosine-distance LIME kernel between each perturbed mask and the all-on
    'present everything' instance. cell_masks: (N, grid, grid) in {0,1}, where
    1=keep. Returns (N,) weights in (0,1]."""
    n = cell_masks.shape[0]
    flat = cell_masks.reshape(n, -1).astype(np.float64)   # 1=present
    ref = np.ones(flat.shape[1])                          # all features present
    # cosine distance to the all-present reference
    dot = flat @ ref
    denom = np.linalg.norm(flat, axis=1) * np.linalg.norm(ref) + 1e-12
    cos = dot / denom
    cos_dist = 1.0 - cos
    # LIME exponential kernel; width scaled to typical distance.
    width = max(np.std(cos_dist), 1e-3) * 0.75
    return np.exp(-(cos_dist ** 2) / (width ** 2))


def solve_lime(cell_masks, targets, sample_weights, grid, alpha):
    """Weighted ridge of target prob on the binary (keep) mask features.
    cell_masks: (N, grid, grid), 1=keep. targets: (N,) target-class prob.
    Returns per-cell importance (grid, grid) = surrogate coefficients."""
    X = cell_masks.reshape(cell_masks.shape[0], -1).astype(np.float64)  # 1=present
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X, targets, sample_weight=sample_weights)
    return reg.coef_.reshape(grid, grid), float(reg.intercept_), reg


# -----------------------------------------------------------------------------
# Sampling helpers.
# -----------------------------------------------------------------------------
def sample_cell_masks(rng, n, grid, mask_prob):
    """Each cell kept with prob (1-mask_prob). 1=keep, 0=masked-out."""
    return (rng.random((n, grid, grid)) > mask_prob).astype(np.uint8)


def evaluate_batch(feature_net, fc, preprocess, img_u8, cell_masks, modes,
                   grid, variants, device, target_class):
    """Fill -> single forward pass -> (feats, target_prob) for a batch of masks.
    Returns (feats[N,2048], target_prob[N]). One model pass; both score and
    LIME target come from it."""
    H, W = img_u8.shape[:2]
    tensors = []
    for cmask, mode in zip(cell_masks, modes):
        pix = cell_mask_to_pixel(cmask, H, W, grid)
        filled = apply_mask(img_u8, pix, str(mode), variants)
        tensors.append(preprocess(Image.fromarray(filled)))
    batch = torch.stack(tensors)
    feats, probs = forward_feats_and_probs(feature_net, fc, batch, device)
    return feats, probs[:, target_class]


# -----------------------------------------------------------------------------
# Main driver.
# -----------------------------------------------------------------------------
@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device = {device}")

    feature_net, fc, preprocess = build_model(device)

    filtering = args.filter_modes is not None
    rng = np.random.default_rng(args.seed)
    grid = args.grid

    # Load + resize input to working resolution so masks align.
    img = Image.open(args.input).convert("RGB").resize(
        (args.work_res, args.work_res), Image.BILINEAR)
    img_u8 = np.asarray(img, dtype=np.uint8)
    H, W = img_u8.shape[:2]
    variants = make_fill_variants(img_u8, blur_var=args.blur_var)

    # Clean-input top-1 class = LIME target.
    clean_feat, clean_probs = forward_feats_and_probs(
        feature_net, fc, preprocess(Image.fromarray(img_u8)).unsqueeze(0), device)
    target_class = int(clean_probs[0].argmax())
    print(f"[*] LIME target = top-1 class {target_class} "
          f"(clean prob {clean_probs[0, target_class]:.4f})")

    # ----- Calibration + threshold (only needed when filtering) -----
    calib = None
    thr = None
    if filtering:
        print(f"[*] fitting calibration from {args.calib_glob} ...")
        calib = fit_calibration(feature_net, preprocess, args.calib_glob,
                                device, args.batch_size, args.pca_dim)
        print(f"    n_calib={calib['n_calib']}  pca_k={calib['k']}  "
              f"evr={calib['evr']:.3f}")
        cs = calib_scores(calib, args.score)
        thr = float(np.quantile(cs, args.threshold_quantile))
        print(f"    [score={args.score}] calib range=[{cs[0]:.2f}, {cs[-1]:.2f}]"
              f"  on-manifold threshold (q={args.threshold_quantile}) = {thr:.2f}")
        active_modes = args.filter_modes
        print(f"[*] filtering ON — fill modes: {', '.join(active_modes)}")
    else:
        active_modes = [args.fill_mode]
        print(f"[*] filtering OFF — plain LIME, fill mode: {args.fill_mode}")

    # Accumulators for masks that go into the solver.
    keep_masks = []     # list of (grid,grid)
    keep_targets = []   # target-class prob
    keep_modes = []
    total_evaluated = 0
    total_survivors = 0

    def process_batch(n_this):
        nonlocal total_evaluated, total_survivors
        cmasks = sample_cell_masks(rng, n_this, grid, args.mask_prob)
        modes = rng.choice(active_modes, size=n_this)
        feats, tgt = evaluate_batch(feature_net, fc, preprocess, img_u8,
                                    cmasks, modes, grid, variants, device,
                                    target_class)
        total_evaluated += n_this
        if not filtering:
            for i in range(n_this):
                keep_masks.append(cmasks[i]); keep_targets.append(float(tgt[i]))
                keep_modes.append(str(modes[i]))
            return n_this
        # Filtering: survivor = ON-manifold = score < thr. Reuse feats+tgt.
        sc = features_to_score(feats, calib, args.score)
        survived = sc < thr
        s = 0
        for i in range(n_this):
            if survived[i]:
                keep_masks.append(cmasks[i]); keep_targets.append(float(tgt[i]))
                keep_modes.append(str(modes[i]))
                s += 1
        total_survivors += s
        return s

    if not filtering:
        # (A) plain LIME on exactly n_samples masks.
        print(f"[*] sampling {args.n_samples} masks (grid={grid}x{grid}) ...")
        remaining = args.n_samples
        while remaining > 0:
            b = min(args.batch_size, remaining)
            process_batch(b)
            remaining -= b

    elif not args.full:
        # (B) filter from the FIRST n_samples only. Total evaluated == n_samples;
        # survivors (<= n_samples) go to the solver.
        print(f"[*] sampling {args.n_samples} masks, keeping on-manifold "
              f"survivors (no --full) ...")
        remaining = args.n_samples
        while remaining > 0:
            b = min(args.batch_size, remaining)
            process_batch(b)
            remaining -= b
        print(f"[*] survivors: {total_survivors} / {args.n_samples} "
              f"on-manifold")

    else:
        # (C) --full: keep batch-sampling + filtering until survivors reach
        # n_samples. Hard cap to avoid infinite loop.
        cap = args.max_draws if args.max_draws > 0 else args.n_samples * args.full_cap_factor
        print(f"[*] --full: accumulating until {args.n_samples} on-manifold "
              f"survivors (draw cap {cap}) ...")
        while total_survivors < args.n_samples and total_evaluated < cap:
            need = args.n_samples - total_survivors
            # over-draw a bit based on running survival rate to converge faster
            rate = (total_survivors / total_evaluated) if total_evaluated else 0.5
            rate = min(max(rate, 0.05), 1.0)
            draw = int(min(max(args.batch_size, need / rate), cap - total_evaluated))
            draw = max(draw, 1)
            b = min(args.batch_size, draw)
            # process in batch_size chunks up to `draw`
            done = 0
            while done < draw and total_survivors < args.n_samples \
                    and total_evaluated < cap:
                bb = min(args.batch_size, draw - done)
                process_batch(bb)
                done += bb
            print(f"    survivors {total_survivors}/{args.n_samples} "
                  f"(evaluated {total_evaluated})")
        if total_survivors < args.n_samples:
            print(f"[WARN] hit draw cap {cap} with only {total_survivors} "
                  f"survivors; solving on those.")
        # Trim to exactly n_samples survivors if we overshot.
        if total_survivors > args.n_samples:
            keep_masks[:] = keep_masks[:args.n_samples]
            keep_targets[:] = keep_targets[:args.n_samples]
            keep_modes[:] = keep_modes[:args.n_samples]

    n_solve = len(keep_masks)
    if n_solve < 2:
        raise SystemExit(f"[FATAL] only {n_solve} masks to solve on — cannot fit "
                         f"LIME. Loosen threshold (--threshold-quantile), change "
                         f"--score, or raise --n-samples / --max-draws.")

    cell_masks = np.stack(keep_masks)
    targets = np.asarray(keep_targets, dtype=np.float64)
    print(f"[*] solving LIME on {n_solve} masks "
          f"(evaluated {total_evaluated} total) ...")

    weights = lime_kernel_weights(cell_masks, args.mask_prob)
    importance, intercept, _ = solve_lime(cell_masks, targets, weights, grid,
                                          args.ridge_alpha)

    # ----- Save outputs -----
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "importance.npy"), importance)
    np.save(os.path.join(args.out_dir, "masks_used.npy"), cell_masks)
    np.save(os.path.join(args.out_dir, "targets_used.npy"), targets)

    # Heatmap upsampled to image resolution, normalized to [0,1] for viewing.
    heat = cell_mask_to_pixel(
        (importance - importance.min()) /
        (np.ptp(importance) + 1e-12), H, W, grid).astype(np.float64)
    heat_u8 = (255 * heat).astype(np.uint8)
    Image.fromarray(heat_u8).save(os.path.join(args.out_dir, "heatmap.png"))

    # Overlay heatmap on the input.
    cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_u8 + 0.5 * cmap).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(args.out_dir, "overlay.png"))

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"input: {args.input}\n")
        f.write(f"target_class: {target_class}\n")
        f.write(f"grid: {grid}x{grid}\n")
        f.write(f"filtering: {filtering}\n")
        if filtering:
            f.write(f"filter_modes: {args.filter_modes}\n")
            f.write(f"score: {args.score}  threshold_q: "
                    f"{args.threshold_quantile}  thr: {thr:.4f}\n")
            f.write(f"full: {args.full}\n")
        f.write(f"n_samples_requested: {args.n_samples}\n")
        f.write(f"total_evaluated: {total_evaluated}\n")
        f.write(f"masks_used_in_solve: {n_solve}\n")
        f.write(f"ridge_intercept: {intercept:.6f}\n")
    print(f"[*] wrote importance.npy, heatmap.png, overlay.png, summary.txt "
          f"to {args.out_dir}/")


def parse_args():
    ap = argparse.ArgumentParser(description="LIME (ResNet-50) with optional "
                                             "off-manifold survival filtering.")
    ap.add_argument("--input", required=True, help="Input image to explain.")
    ap.add_argument("--calib-glob", default="sample_1k/*.JPEG",
                    help="Glob for calibration images (used only when "
                         "--filter-modes is set).")
    ap.add_argument("--grid", type=int, default=16, help="Grid size (GxG).")
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="Plain/no-full: number of masks evaluated. "
                         "--full: number of ON-manifold survivors required.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.15,
                    help="Prob. a cell is masked-out (LIME-style).")

    # Filtering controls. If --filter-modes is omitted, plain LIME (regime A).
    ap.add_argument("--filter-modes", nargs="+", default=None,
                    choices=FILL_MODES, metavar="MODE",
                    help="Enable off-manifold survival filtering and choose the "
                         "fill mode(s) sampled per mask. A mask survives if it "
                         "is ON-manifold (score < threshold) under the mode it "
                         "was assigned. If omitted: plain LIME.")
    ap.add_argument("--full", action="store_true",
                    help="With --filter-modes: keep sampling/filtering until "
                         "--n-samples ON-manifold survivors are collected "
                         "(regime C). Without it: filter only the first "
                         "--n-samples masks (regime B).")
    ap.add_argument("--max-draws", type=int, default=-1,
                    help="--full safety cap on total masks evaluated "
                         "(-1 => n_samples * --full-cap-factor).")
    ap.add_argument("--full-cap-factor", type=int, default=20,
                    help="--full draw cap = n_samples * this (if --max-draws<0).")

    ap.add_argument("--fill-mode", default="blur", choices=FILL_MODES,
                    help="Fill mode for PLAIN LIME (when --filter-modes unset).")

    ap.add_argument("--score", default="residual",
                    choices=["mahalanobis", "residual", "combo"],
                    help="Off-manifold score for the survival test.")
    ap.add_argument("--threshold-quantile", type=float, default=0.95,
                    help="Calibration-score quantile; masks scoring BELOW it are "
                         "on-manifold survivors.")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="PCA target dim (clamped to n_calib-1).")
    ap.add_argument("--blur-var", type=float, default=50.0,
                    help="Gaussian blur variance for the blur fill mode.")

    ap.add_argument("--ridge-alpha", type=float, default=1.0,
                    help="Ridge regularization for the LIME surrogate.")
    ap.add_argument("--work-res", type=int, default=224)
    ap.add_argument("--out-dir", default="lime_out")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())