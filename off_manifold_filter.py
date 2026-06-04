#!/usr/bin/env python3
"""
off_manifold_filter.py

Pipeline:
  1. Load ResNet-50 (ImageNet weights).
  2. From a calibration image folder, extract penultimate features (2048-d,
     post-global-average-pool, pre-fc), PCA-reduce to a low-dim subspace the
     n~50 calibration points can actually populate (default min(n-1, 64)),
     then fit a Gaussian (mu, shrinkage-cov) IN THAT SUBSPACE.
  3. For a given input image, build a GRID x GRID grid, generate LIME-style
     random binary masks, fill the masked-out cells with one of several
     fill modes (blur / black / white / inpaint / corner-mean / white-noise).
  4. Batch the masked images through the model, project to the PCA subspace,
     score each by Mahalanobis distance, map to an off-manifold probability
     via a sigmoid centered on the calibration threshold (so scores near the
     boundary get intermediate p_off instead of snapping to 1.0).
  5. Output the masked samples that are flagged off-manifold (image + mask +
     p_off), saved to an output directory.

  --self-test prints leave-one-out calibration distances and the clean-input
  distance so you can see where the threshold actually sits before trusting it.

Usage:
  python off_manifold_filter.py --input path/to/image.JPEG
  python off_manifold_filter.py --input img.JPEG --calib-glob "benchmark_50/*.JPEG" \
         --grid 16 --n-samples 1000 --pca-dim 64 --mask-prob 0.15
  python off_manifold_filter.py --input img.JPEG --self-test

Requires: torch, torchvision, numpy, pillow, opencv-python (for inpaint),
          scikit-learn (PCA + Ledoit-Wolf shrinkage).
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
import torchvision.transforms as T
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


# -----------------------------------------------------------------------------
# Model: ResNet-50 truncated to the 2048-d penultimate feature.
# -----------------------------------------------------------------------------
def build_feature_model(device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.eval().to(device)

    # Feature extractor = everything up to and including avgpool, flattened.
    # The classification head is net.fc (kept separate so we can also read probs).
    feature_net = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
        net.avgpool, nn.Flatten(),
    ).eval().to(device)

    fc = net.fc.eval().to(device)
    preprocess = weights.transforms()  # resize/crop/normalize matching the weights
    return feature_net, fc, preprocess


# -----------------------------------------------------------------------------
# Calibration: fit Gaussian over penultimate features.
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_features(feature_net, batch, device):
    return feature_net(batch.to(device)).cpu().numpy()


@torch.no_grad()
def fit_calibration(feature_net, preprocess, calib_glob, device, batch_size,
                    pca_dim):
    paths = sorted(glob.glob(calib_glob))
    if not paths:
        raise FileNotFoundError(f"No calibration images matched: {calib_glob}")

    feats = []
    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) == batch_size:
            feats.append(extract_features(feature_net, torch.stack(batch), device))
            batch = []
    if batch:
        feats.append(extract_features(feature_net, torch.stack(batch), device))

    feats = np.concatenate(feats, axis=0).astype(np.float64)  # (N, 2048)
    n = feats.shape[0]

    # n (~50) << p (2048): a Gaussian in raw 2048-d space is degenerate, so
    # every perturbation lands "outside" all calibration points and p_off
    # saturates at 1.0. Fix: PCA-reduce to a subspace the N points can populate
    # (k <= N-1, since PCA on N centered points has at most N-1 nonzero comps).
    k = min(pca_dim, n - 1)
    if k < 1:
        raise ValueError(f"Need >=2 calibration images for PCA, got n={n}.")
    pca = PCA(n_components=k, whiten=False, svd_solver="full").fit(feats)
    z = pca.transform(feats)  # (N, k)
    evr = float(pca.explained_variance_ratio_.sum())

    mu = z.mean(axis=0)
    # Even at k<=N-1 the sample cov can be near-singular; shrink for stability.
    lw = LedoitWolf().fit(z - mu)
    precision = lw.precision_.astype(np.float64)  # Sigma^-1 in PCA space

    centered = z - mu
    calib_d2 = np.einsum("ni,ij,nj->n", centered, precision, centered)

    # Residual / reconstruction-error score (PCA-OOD): energy of a feature
    # vector OUTSIDE the retained subspace. Masking artifacts tend to live in
    # the discarded directions, so this is often a far stronger off-manifold
    # signal than the in-subspace Mahalanobis distance.
    #   recon = pca.inverse_transform(pca.transform(x))
    #   residual = ||x - recon||^2
    feat_mean = feats.mean(axis=0)
    recon = pca.inverse_transform(z)
    calib_resid = np.sum((feats - recon) ** 2, axis=1)

    return {
        "pca": pca,
        "mu": mu,
        "precision": precision,
        "calib_d2": np.sort(calib_d2),
        "calib_resid": np.sort(calib_resid),
        "feat_mean": feat_mean,
        "n_calib": n,
        "k": k,
        "evr": evr,
    }


def features_to_d2(feats, calib):
    """Project raw 2048-d features to PCA space, return Mahalanobis d2."""
    z = calib["pca"].transform(feats.astype(np.float64))
    centered = z - calib["mu"]
    return np.einsum("ni,ij,nj->n", centered, calib["precision"], centered)


def features_to_resid(feats, calib):
    """Reconstruction-error score: squared energy outside the PCA subspace."""
    x = feats.astype(np.float64)
    z = calib["pca"].transform(x)
    recon = calib["pca"].inverse_transform(z)
    return np.sum((x - recon) ** 2, axis=1)


def features_to_score(feats, calib, score, baseline=None):
    """Dispatch to the chosen off-manifold score.

    For *_delta scores, `baseline` is the clean-input score of the same base
    metric; the returned value is (score - baseline), i.e. how much each mask
    moves the input off-manifold relative to its own starting point. This
    cancels the input's intrinsic atypicality (where it sits in the calibration
    distribution) and measures only the effect of masking."""
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
    elif score == "residual_delta":
        if baseline is None:
            raise ValueError("residual_delta requires baseline")
        return features_to_resid(feats, calib) - baseline
    elif score == "mahalanobis_delta":
        if baseline is None:
            raise ValueError("mahalanobis_delta requires baseline")
        return features_to_d2(feats, calib) - baseline
    raise ValueError(f"unknown score: {score}")


def base_metric(score):
    """The underlying metric a (possibly delta) score is built on."""
    return {"residual_delta": "residual",
            "mahalanobis_delta": "mahalanobis"}.get(score, score)


def is_delta(score):
    return score.endswith("_delta")


def calib_scores(calib, score):
    """Sorted calibration scores for the chosen metric (for threshold/CDF).
    Delta scores have no calibration analogue (calibration images have no
    'clean baseline'), so their threshold is set separately in run()."""
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
# Fill modes. Each takes the full RGB image (uint8 HxWx3) and returns a filled
# version; the per-cell mask decides where the fill is pasted.
# -----------------------------------------------------------------------------
def make_fill_variants(img_u8, blur_var=50.0):
    H, W = img_u8.shape[:2]
    variants = {}

    # Blur: Gaussian blur with sigma s.t. variance ~= blur_var -> sigma=sqrt(var).
    sigma = float(np.sqrt(blur_var))
    k = max(3, int(2 * round(3 * sigma) + 1))  # odd kernel covering ~3 sigma
    variants["blur"] = cv2.GaussianBlur(img_u8, (k, k), sigmaX=sigma, sigmaY=sigma)

    variants["black"] = np.zeros_like(img_u8)
    variants["white"] = np.full_like(img_u8, 255)

    # Corner-mean: mean color of the four 10%x10% corner patches.
    ch, cw = max(1, int(0.10 * H)), max(1, int(0.10 * W))
    corners = np.concatenate([
        img_u8[:ch, :cw].reshape(-1, 3),
        img_u8[:ch, -cw:].reshape(-1, 3),
        img_u8[-ch:, :cw].reshape(-1, 3),
        img_u8[-ch:, -cw:].reshape(-1, 3),
    ], axis=0)
    corner_mean = corners.mean(axis=0)
    variants["corner_mean"] = np.broadcast_to(
        corner_mean.astype(np.uint8), img_u8.shape).copy()

    # White noise: uniform random RGB (regenerated per-sample at fill time
    # for "white-noise"; here we store None and handle specially).
    variants["white_noise"] = None  # generated per sample

    # Inpaint handled per-sample (needs the actual mask), store None.
    variants["inpaint"] = None
    return variants


FILL_MODES = ["blur", "black", "white", "inpaint", "corner_mean", "white_noise"]


def cell_mask_to_pixel(cell_mask, H, W, grid):
    """Upsample a (grid,grid) binary cell mask to a (H,W) pixel mask.
    Convention: 1 = keep original, 0 = masked-out (gets filled)."""
    pix = np.kron(cell_mask, np.ones((int(np.ceil(H / grid)),
                                      int(np.ceil(W / grid))), dtype=np.uint8))
    return pix[:H, :W]


def apply_mask(img_u8, pix_mask, mode, variants):
    """Return filled image where pix_mask==0 is replaced by the fill source."""
    keep = pix_mask[..., None].astype(bool)  # (H,W,1)
    if mode == "white_noise":
        fill = np.random.randint(0, 256, size=img_u8.shape, dtype=np.uint8)
    elif mode == "inpaint":
        # cv2.inpaint: mask is uint8, nonzero = region to inpaint.
        inpaint_mask = (1 - pix_mask).astype(np.uint8) * 255
        fill = cv2.inpaint(img_u8, inpaint_mask, inpaintRadius=3,
                           flags=cv2.INPAINT_TELEA)
    else:
        fill = variants[mode]
    return np.where(keep, img_u8, fill)


# -----------------------------------------------------------------------------
# Sampling + scoring.
# -----------------------------------------------------------------------------
def sigmoid_p_off(d2, thr, width):
    """Map score -> (0,1) with a logistic centered at thr. `width` controls how
    sharp the boundary is (in score units). p_off = 0.5 exactly at score == thr.

    Numerically stable: computed branchwise so np.exp never sees a large
    positive argument (which would overflow). Mathematically identical to
    1/(1+exp(-z))."""
    z = (np.asarray(d2, dtype=np.float64) - thr) / max(width, 1e-9)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))          # z>=0: exp(-z) in (0,1]
    ez = np.exp(z[~pos])                               # z<0: exp(z) in (0,1)
    out[~pos] = ez / (1.0 + ez)
    return out


@torch.no_grad()
def self_test(args, feature_net, preprocess, calib, device):
    """Calibration distances + clean-input distance, so the user can see whether
    the threshold actually separates before trusting it. Reports BOTH metrics
    (mahalanobis and residual) regardless of the active --score."""
    print("\n[self-test] in-sample calibration scores (optimistic bound)")
    paths = sorted(glob.glob(args.calib_glob))
    feats = []
    batch = []
    for p in paths:
        batch.append(preprocess(Image.open(p).convert("RGB")))
        if len(batch) == args.batch_size:
            feats.append(feature_net(torch.stack(batch).to(device)).cpu().numpy())
            batch = []
    if batch:
        feats.append(feature_net(torch.stack(batch).to(device)).cpu().numpy())
    feats = np.concatenate(feats, axis=0)

    d2 = features_to_d2(feats, calib)
    rs = features_to_resid(feats, calib)
    print(f"    calib mahalanobis: min={d2.min():.2f} med={np.median(d2):.2f} "
          f"max={d2.max():.2f}")
    print(f"    calib residual   : min={rs.min():.1f} med={np.median(rs):.1f} "
          f"max={rs.max():.1f}")

    # Clean (unperturbed) input.
    img = Image.open(args.input).convert("RGB").resize(
        (args.work_res, args.work_res), Image.BILINEAR)
    clean = feature_net(preprocess(img).unsqueeze(0).to(device)).cpu().numpy()

    if is_delta(args.score):
        # By definition the clean input's delta is 0 (it IS the baseline), so
        # the only thing to report is whether the baseline sits in-distribution
        # for the base metric, and what the delta threshold will be.
        bm = base_metric(args.score)
        b = float(features_to_score(clean, calib, bm)[0])
        base_cs = calib_scores(calib, bm)
        b_pct = 100.0 * np.searchsorted(base_cs, b) / len(base_cs)
        b_q75, b_q25 = np.percentile(base_cs, [75, 25])
        thr = args.delta_threshold * max(b_q75 - b_q25, 1e-6)
        print(f"    [score={args.score}] clean baseline ({bm}) = {b:.2f} "
              f"(calib pctile {b_pct:.0f}%)   delta threshold = {thr:.2f}")
        print(f"    NOTE delta cancels baseline atypicality: a {b_pct:.0f}%-ile "
              f"input is fine, only the mask's effect is measured.\n")
    else:
        s_clean = float(features_to_score(clean, calib, args.score)[0])
        cs = calib_scores(calib, args.score)
        thr = float(np.quantile(cs, args.threshold_quantile))
        print(f"    [score={args.score}] clean input = {s_clean:.2f}   "
              f"threshold(q={args.threshold_quantile}) = {thr:.2f}")
        if s_clean >= thr:
            print("    [WARN] clean input already exceeds threshold -> metric not "
                  "separating; lower --pca-dim, switch --score, or use a "
                  "*_delta score to cancel the input's baseline atypicality.")
        else:
            print("    [OK] clean input is inside the calibration distribution.")
    print(f"    PCA: k={calib['k']} comps, explained var ratio={calib['evr']:.3f}\n")


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device = {device}")

    feature_net, fc, preprocess = build_feature_model(device)

    print(f"[*] fitting calibration from {args.calib_glob} ...")
    calib = fit_calibration(feature_net, preprocess, args.calib_glob,
                            device, args.batch_size, args.pca_dim)
    print(f"    n_calib={calib['n_calib']}  pca_k={calib['k']}  "
          f"evr={calib['evr']:.3f}")

    # Load input image at the model's working resolution so masks align cleanly.
    img = Image.open(args.input).convert("RGB")
    work_res = args.work_res
    img = img.resize((work_res, work_res), Image.BILINEAR)
    img_u8 = np.asarray(img, dtype=np.uint8)
    H, W = img_u8.shape[:2]

    # Clean-input baseline (needed for delta scores; informative otherwise).
    clean_feat = feature_net(
        preprocess(Image.fromarray(img_u8)).unsqueeze(0).to(device)).cpu().numpy()
    base_m = base_metric(args.score)
    baseline = float(features_to_score(clean_feat, calib, base_m)[0])

    if is_delta(args.score):
        # Delta score = score(masked) - baseline. No calibration analogue, so
        # the threshold is an ABSOLUTE margin above the clean input, scaled to
        # the calibration spread of the base metric so it's comparable across
        # inputs and pca-dims.
        base_cs = calib_scores(calib, base_m)
        b_q75, b_q25 = np.percentile(base_cs, [75, 25])
        base_iqr = max(b_q75 - b_q25, 1e-6)
        thr = args.delta_threshold * base_iqr  # delta units
        width = args.sigmoid_width if args.sigmoid_width > 0 else base_iqr
        print(f"    [score={args.score}] clean baseline ({base_m}) = "
              f"{baseline:.2f}   base IQR = {base_iqr:.2f}")
        print(f"    delta threshold = {thr:.2f} "
              f"(= {args.delta_threshold} x base IQR)  sigmoid width = {width:.2f}")
    else:
        cs = calib_scores(calib, args.score)
        print(f"    [score={args.score}] calib range=[{cs[0]:.2f}, {cs[-1]:.2f}]"
              f"   clean baseline = {baseline:.2f}")
        thr = float(np.quantile(cs, args.threshold_quantile))
        if args.sigmoid_width > 0:
            width = args.sigmoid_width
        else:
            q75, q25 = np.percentile(cs, [75, 25])
            width = max(q75 - q25, 1e-6)
        print(f"    threshold (q={args.threshold_quantile}) = {thr:.2f}  "
              f"sigmoid width = {width:.2f}")

    if args.self_test:
        self_test(args, feature_net, preprocess, calib, device)

    variants = make_fill_variants(img_u8, blur_var=args.blur_var)

    rng = np.random.default_rng(args.seed)
    grid = args.grid

    os.makedirs(args.out_dir, exist_ok=True)

    # Generate samples: each = (cell_mask, fill_mode).
    n = args.n_samples
    cell_masks = (rng.random((n, grid, grid)) > args.mask_prob).astype(np.uint8)
    modes = rng.choice(FILL_MODES, size=n)

    results = []  # (idx, p_off, score, mode, filled_u8, cell_mask)
    buf_tensors, buf_meta = [], []

    def flush():
        if not buf_tensors:
            return
        batch = torch.stack(buf_tensors).to(device)
        feats = feature_net(batch).cpu().numpy()
        sc = features_to_score(feats, calib, args.score, baseline=baseline)
        p_off = sigmoid_p_off(sc, thr, width)
        for (idx, mode, filled, cmask), dd, pp in zip(buf_meta, sc, p_off):
            if dd >= thr:
                results.append((idx, float(pp), float(dd), mode, filled, cmask))
        buf_tensors.clear()
        buf_meta.clear()

    print(f"[*] sampling {n} masked variants (grid={grid}x{grid}) ...")
    for i in range(n):
        mode = str(modes[i])
        cmask = cell_masks[i]
        pix = cell_mask_to_pixel(cmask, H, W, grid)
        filled = apply_mask(img_u8, pix, mode, variants)
        buf_tensors.append(preprocess(Image.fromarray(filled)))
        buf_meta.append((i, mode, filled, cmask))
        if len(buf_tensors) == args.batch_size:
            flush()
    flush()

    # Sort off-manifold samples by probability, descending.
    results.sort(key=lambda r: r[1], reverse=True)
    print(f"[*] {len(results)} / {n} samples flagged off-manifold "
          f"(score >= {thr:.2f})")

    # Save flagged samples.
    keep_top = results if args.max_save < 0 else results[:args.max_save]
    for rank, (idx, p_off, sc, mode, filled, cmask) in enumerate(keep_top):
        base = f"offman_{rank:04d}_p{p_off:.3f}_{mode}"
        Image.fromarray(filled).save(os.path.join(args.out_dir, base + ".png"))
        np.save(os.path.join(args.out_dir, base + "_mask.npy"), cmask)
    print(f"[*] saved {len(keep_top)} flagged images + masks to {args.out_dir}/")

    # Manifest. The score column name reflects the active --score.
    with open(os.path.join(args.out_dir, "manifest.csv"), "w") as f:
        f.write(f"rank,sample_idx,p_off,{args.score}_score,fill_mode\n")
        for rank, (idx, p_off, sc, mode, _, _) in enumerate(keep_top):
            f.write(f"{rank},{idx},{p_off:.6f},{sc:.4f},{mode}\n")
    print(f"[*] manifest written: {os.path.join(args.out_dir, 'manifest.csv')}")


def parse_args():
    ap = argparse.ArgumentParser(description="Off-manifold mask filter (ResNet-50).")
    ap.add_argument("--input", required=True, help="Input image path.")
    ap.add_argument("--calib-glob", default="benchmark_50/*.JPEG",
                    help="Glob for calibration images.")
    ap.add_argument("--grid", type=int, default=16, help="Grid size (GxG).")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.15,
                    help="Prob. a cell is masked-out (LIME-style). "
                         "0.5 saturates the filter; 0.10-0.20 is discriminative.")
    ap.add_argument("--score", default="residual",
                    choices=["mahalanobis", "residual", "combo",
                             "residual_delta", "mahalanobis_delta"],
                    help="Off-manifold score. 'residual' = PCA reconstruction "
                         "error (energy outside subspace; best for masking "
                         "artifacts). 'mahalanobis' = in-subspace distance. "
                         "'combo' = normalized sum. '*_delta' = score relative "
                         "to the clean input's baseline, which cancels the "
                         "input's intrinsic atypicality and measures only the "
                         "mask's effect (recommended for edge-of-manifold "
                         "inputs / robust across pca-dim).")
    ap.add_argument("--delta-threshold", type=float, default=1.0,
                    help="For *_delta scores: threshold in units of the base "
                         "metric's calibration IQR. A mask is off-manifold if it "
                         "moves the input more than this many IQRs past clean.")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="PCA target dim (clamped to n_calib-1). Lower = stricter "
                         "manifold; raise if clean input still flags off-manifold.")
    ap.add_argument("--sigmoid-width", type=float, default=-1.0,
                    help="Logistic width in d2 units for p_off. -1 = use calib IQR.")
    ap.add_argument("--blur-var", type=float, default=50.0,
                    help="Gaussian blur variance (sigma=sqrt(var)).")
    ap.add_argument("--threshold-quantile", type=float, default=0.95,
                    help="Calibration d2 quantile used as off-manifold cutoff.")
    ap.add_argument("--work-res", type=int, default=224,
                    help="Resolution at which masking is applied.")
    ap.add_argument("--out-dir", default="off_manifold_out")
    ap.add_argument("--max-save", type=int, default=-1,
                    help="Max flagged samples to save (-1 = all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="Print calib LOO distances + clean-input distance.")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())