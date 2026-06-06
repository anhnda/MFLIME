#!/usr/bin/env python3
"""
LIMEScore.py  (mid-layer per-position residual edition)

STANDARD LIME (grid-superpixel, ResNet-50) + a SEPARATELY REPORTED
on-manifold reliability number, "LIMEScore".

WHAT CHANGED vs the penultimate-feature version
------------------------------------------------
The on-manifold residual is no longer computed on the pooled 2048-d penultimate
feature. That feature is the layer at which the backbone has ALREADY resolved
the fill into a class-plausible representation, so a scatter mask whose high
target prob comes from the *fill* (or from a few spurious cells) still lands
near the calibration manifold there -- the residual cannot see where the
on-manifold-ness came from. Upweighting by it produces scatter maps.

Instead we tap a MID layer (default `layer3`) whose activation is still a
spatial map (C, H, W). The fill artifact is still spatially legible there: an
off-manifold patch corrupts the channel statistics *at the cells covering that
patch*, before the network has the depth to repair it.

The calibration manifold is the distribution of a SINGLE natural cell's
C-dim channel vector -- i.e. the per-spatial-position channel covariance,
pooled over all positions of all calibration images. This `C x C` object is a
far lower-entropy distribution than the flattened `C*H*W` activation, so a
low-rank PCA subspace is meaningful even at a layer where the flattened global
feature is near-full-rank. (Verify the spectrum with --print-spectrum.)

Because the residual is now PER CELL, we get a residual MAP (H_act x W_act) per
perturbation, and can aggregate it two ways:

  * mean over ALL cells            -> realism of the whole perturbed image.
  * mean over KEPT cells only      -> realism of the *kept content*. This is the
                                      quantity that severs the circularity:
                                      a mask earns a low score only when the
                                      regions it claims are doing the work are
                                      themselves on-manifold, regardless of how
                                      the network later resolves the fill.

The KEPT-restricted residual is mapped from the LIME grid onto the activation
grid by spatial overlap (the activation map is coarse, e.g. 14x14 at layer3,
so each activation cell is assigned the keep-fraction of the grid cells it
covers).

Everything is still DIAGNOSTIC ONLY by default: the LIME map is plain standard
LIME, all rows kept, weighted only by the cosine kernel. Pass
--weight-by-kept-residual to additionally emit a residual-aware map (kept-
restricted soft weighting) alongside the plain one, so you can compare.

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
from sklearn.decomposition import PCA


MID_LAYERS = ["layer1", "layer2", "layer3", "layer4"]


# =============================================================================
# Model. One forward pass yields: a MID-layer activation map (C,H,W) for the
# residual, the penultimate 2048-d feature (kept for reference), and class
# probs (LIME target).
# =============================================================================
class TappedResNet(nn.Module):
    """ResNet-50 that returns (mid_activation_map, penultimate_feat, probs)
    from a single forward pass. `mid_layer` selects which residual block's
    output is used for the on-manifold residual."""

    def __init__(self, mid_layer, device):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights).eval().to(device)
        self.net = net
        self.mid_layer = mid_layer
        if mid_layer not in MID_LAYERS:
            raise ValueError(f"mid_layer must be one of {MID_LAYERS}")
        # stem
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = {
            "layer1": net.layer1, "layer2": net.layer2,
            "layer3": net.layer3, "layer4": net.layer4,
        }
        self.avgpool = net.avgpool
        self.flatten = nn.Flatten()
        self.fc = net.fc

    @torch.no_grad()
    def forward(self, x_norm):
        h = self.stem(x_norm)
        mid = None
        for name in MID_LAYERS:
            h = self.blocks[name](h)
            if name == self.mid_layer:
                mid = h  # (B, C, Ha, Wa)
            # keep going to the head regardless, so we also get probs
        feat = self.flatten(self.avgpool(h))           # (B, 2048)
        probs = torch.softmax(self.fc(feat), dim=1)     # (B, 1000)
        return mid, feat, probs


def build_model(mid_layer, device):
    model = TappedResNet(mid_layer, device).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return model, mean, std


def normalize(x01, mean, std):
    return (x01 - mean) / std


@torch.no_grad()
def forward_all(model, x01, mean, std):
    """One pass -> (mid (B,C,Ha,Wa), feat (B,2048), probs (B,1000))."""
    return model(normalize(x01, mean, std))


@torch.no_grad()
def forward_feats_probs(feature_net, fc, x01, mean, std):
    """Backward-compat shim for metrics.py (penultimate-feature signature).

    The mid-layer rewrite replaced this with forward_all, but metrics.py still
    imports and calls forward_feats_probs(feature_net, fc, x01, mean, std) ->
    (feats, probs), where feature_net is a Sequential ending in flatten and fc
    is the classifier head. run()/sweep.py build exactly that pair from the
    TappedResNet. This keeps the faithfulness-AUC path working unchanged."""
    feats = feature_net(normalize(x01, mean, std))
    probs = torch.softmax(fc(feats), dim=1)
    return feats, probs


# =============================================================================
# Blur reference (unchanged).
# =============================================================================
def blur_reference(x01, sigma):
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
# Calibration: per-position channel PCA subspace on the MID-layer activation.
#
# We pool the C-dim channel vectors over ALL spatial positions of ALL
# calibration images, then fit PCA on that pool. The residual of one cell is
# its channel vector's energy outside this subspace. Scale-free variants are
# computed per cell so that low-norm collapsed cells (black/white/const fill)
# cannot win by shrinking.
# =============================================================================
@torch.no_grad()
def fit_calibration(model, mean, std, calib_glob, device, batch_size,
                    pca_dim, work_res, max_cells_for_pca=200_000,
                    print_spectrum=False, seed=0):
    paths = sorted(glob.glob(calib_glob))
    if not paths:
        raise FileNotFoundError(f"No calibration images matched: {calib_glob}")

    rng = np.random.default_rng(seed)

    def load01(p):
        img = Image.open(p).convert("RGB").resize((work_res, work_res),
                                                  Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    cell_vecs = []      # list of (n_cells_i, C) arrays, one per image batch
    batch = []
    Ca = None
    for p in paths:
        batch.append(load01(p))
        if len(batch) == batch_size:
            xb = torch.stack(batch).to(device)
            mid, _, _ = model(normalize(xb, mean, std))   # (B,C,Ha,Wa)
            B, C, Ha, Wa = mid.shape
            Ca = C
            v = mid.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            cell_vecs.append(v.astype(np.float64))
            batch = []
    if batch:
        xb = torch.stack(batch).to(device)
        mid, _, _ = model(normalize(xb, mean, std))
        B, C, Ha, Wa = mid.shape
        Ca = C
        v = mid.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        cell_vecs.append(v.astype(np.float64))

    cells = np.concatenate(cell_vecs, axis=0)   # (N_total_cells, C)
    n_cells_total = cells.shape[0]

    # Subsample for the PCA fit if the cell pool is huge (it usually is:
    # 1000 imgs * 14*14 = ~196k rows at layer3). Subsampling the FIT is fine;
    # the subspace is stable. We still report stats over the full pool.
    if n_cells_total > max_cells_for_pca:
        idx = rng.choice(n_cells_total, size=max_cells_for_pca, replace=False)
        fit_pool = cells[idx]
    else:
        fit_pool = cells

    k = min(pca_dim, fit_pool.shape[1], fit_pool.shape[0] - 1)
    if k < 1:
        raise ValueError("Not enough calibration cells for PCA.")
    pca = PCA(n_components=k, whiten=False, svd_solver="full").fit(fit_pool)
    evr = float(pca.explained_variance_ratio_.sum())

    if print_spectrum:
        ev = pca.explained_variance_ratio_
        cum = np.cumsum(ev)
        print(f"    [spectrum @ {model.mid_layer}] C={Ca}  first 10 EVR: "
              + ", ".join(f"{e:.3f}" for e in ev[:10]))
        for tgt in (0.50, 0.80, 0.90, 0.95, 0.99):
            j = int(np.searchsorted(cum, tgt)) + 1
            print(f"      {int(tgt*100)}% energy at k={min(j, k)}")
        # Flatness heuristic: ratio of last-kept to first eigenvalue.
        print(f"      EVR[0]={ev[0]:.4f}  EVR[k-1]={ev[-1]:.6f}  "
              f"(flat spectrum -> low-rank PCA residual is weak here)")

    # Calibration residual distribution (over the FULL cell pool), all variants.
    cal = compute_cell_residuals(cells, pca)
    return {
        "pca": pca, "mid_layer": model.mid_layer, "C": Ca, "k": k, "evr": evr,
        "n_calib_imgs": len(paths), "n_cells_total": n_cells_total,
        "calib_resid": np.sort(cal["resid"]),
        "calib_resid_norm": np.sort(cal["resid_norm"]),
        "calib_resid_ang": np.sort(cal["resid_ang"]),
    }


def compute_cell_residuals(cells, pca):
    """cells: (N, C). Returns dict of per-cell residual variants (each (N,))."""
    x = cells.astype(np.float64)
    z = pca.transform(x)
    recon = pca.inverse_transform(z)
    diff2 = np.sum((x - recon) ** 2, axis=1)
    fnorm2 = np.sum(x ** 2, axis=1)
    resid = diff2
    resid_norm = diff2 / np.maximum(fnorm2, 1e-12)
    dot = np.sum(x * recon, axis=1)
    rnorm = np.sqrt(np.sum(recon ** 2, axis=1))
    fnorm = np.sqrt(fnorm2)
    resid_ang = 1.0 - dot / np.maximum(fnorm * rnorm, 1e-12)
    return {"resid": resid, "resid_norm": resid_norm, "resid_ang": resid_ang}


def mid_to_resid_maps(mid_np, pca, score):
    """mid_np: (B, C, Ha, Wa). Returns residual map (B, Ha, Wa) for `score`."""
    B, C, Ha, Wa = mid_np.shape
    cells = mid_np.transpose(0, 2, 3, 1).reshape(-1, C)
    r = compute_cell_residuals(cells, pca)
    key = {"residual": "resid", "residual_norm": "resid_norm",
           "residual_ang": "resid_ang"}[score]
    return r[key].reshape(B, Ha, Wa)


def calib_dist(calib, score):
    return {"residual": calib["calib_resid"],
            "residual_norm": calib["calib_resid_norm"],
            "residual_ang": calib["calib_resid_ang"]}[score]


# =============================================================================
# Calibration cache. Calibration depends ONLY on (mid_layer, calib_glob,
# pca_dim, work_res, seed) -- NOT on the input image or fill/alpha. So in a
# sweep we fit it ONCE and reuse. We persist the fitted PCA (components_,
# mean_, explained_variance_) plus the sorted residual distributions, and
# rehydrate a PCA object on load so the rest of the code is unchanged.
# =============================================================================
def _calib_cache_key(args):
    # Anything that changes the fitted subspace must be in the key.
    return (f"layer={args.mid_layer}|glob={args.calib_glob}|"
            f"pca={args.pca_dim}|res={args.work_res}|seed={args.seed}|"
            f"maxcells={args.max_cells_for_pca}")


def save_calibration(path, calib, key):
    pca = calib["pca"]
    np.savez_compressed(
        path,
        key=np.array(key),
        components=pca.components_.astype(np.float64),
        pca_mean=pca.mean_.astype(np.float64),
        explained_variance=pca.explained_variance_.astype(np.float64),
        mid_layer=np.array(calib["mid_layer"]),
        C=np.array(calib["C"]), k=np.array(calib["k"]),
        evr=np.array(calib["evr"]),
        n_calib_imgs=np.array(calib["n_calib_imgs"]),
        n_cells_total=np.array(calib["n_cells_total"]),
        calib_resid=calib["calib_resid"],
        calib_resid_norm=calib["calib_resid_norm"],
        calib_resid_ang=calib["calib_resid_ang"],
    )


def load_calibration(path, key):
    """Returns a calib dict identical in shape to fit_calibration's output,
    or None if the cache is missing or its key does not match."""
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    if str(d["key"]) != key:
        return None
    pca = PCA(n_components=int(d["k"]), whiten=False, svd_solver="full")
    # Rehydrate the fitted state sklearn needs for transform/inverse_transform.
    pca.components_ = d["components"]
    pca.mean_ = d["pca_mean"]
    pca.explained_variance_ = d["explained_variance"]
    pca.n_components_ = int(d["k"])
    pca.n_features_in_ = d["components"].shape[1]
    return {
        "pca": pca, "mid_layer": str(d["mid_layer"]), "C": int(d["C"]),
        "k": int(d["k"]), "evr": float(d["evr"]),
        "n_calib_imgs": int(d["n_calib_imgs"]),
        "n_cells_total": int(d["n_cells_total"]),
        "calib_resid": d["calib_resid"],
        "calib_resid_norm": d["calib_resid_norm"],
        "calib_resid_ang": d["calib_resid_ang"],
    }


# =============================================================================
# Grid <-> activation-grid overlap. The LIME grid (G x G) and the activation
# grid (Ha x Wa) differ in resolution. For the KEPT-restricted residual we need
# a per-activation-cell keep-fraction: how much of each activation cell is
# covered by KEPT grid cells. Computed by area overlap on the [0,1]^2 square.
# =============================================================================
def grid_to_act_overlap(grid, act_hw):
    """Returns W (Ha*Wa, G*G): row = activation cell, col = grid cell, entry =
    fractional area overlap (rows sum to 1). Precomputed once."""
    G = grid[0]
    Ha, Wa = act_hw
    # grid cell edges and act cell edges on [0,1]
    ge = np.linspace(0, 1, G + 1)
    aey = np.linspace(0, 1, Ha + 1)
    aex = np.linspace(0, 1, Wa + 1)

    def overlap_1d(a0, a1, edges):
        # fractional overlap of [a0,a1] with each bin in `edges`
        lo = np.maximum(a0, edges[:-1])
        hi = np.minimum(a1, edges[1:])
        ov = np.clip(hi - lo, 0, None)
        return ov  # length = len(edges)-1

    Wmat = np.zeros((Ha * Wa, G * G), dtype=np.float64)
    cell_area = (1.0 / Ha) * (1.0 / Wa)
    for ay in range(Ha):
        oy = overlap_1d(aey[ay], aey[ay + 1], ge)        # (G,) over grid rows
        for ax in range(Wa):
            ox = overlap_1d(aex[ax], aex[ax + 1], ge)    # (G,) over grid cols
            area = np.outer(oy, ox).reshape(-1)          # (G*G,)
            Wmat[ay * Wa + ax] = area / cell_area        # normalize to fractions
    return Wmat  # rows ~sum to 1


def kept_fraction_per_act_cell(Z_keep, overlap):
    """Z_keep: (B, G*G) in {0,1} (1=kept sharp). overlap: (Ha*Wa, G*G).
    Returns (B, Ha*Wa): fraction of each activation cell covered by KEPT grid
    cells."""
    return Z_keep.astype(np.float64) @ overlap.T  # (B, Ha*Wa)


# =============================================================================
# Fill modes (unchanged).
# =============================================================================
FILL_MODES = ["blur", "black", "white", "inpaint", "corner_mean",
              "white_noise", "blend"]

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


def build_perturbations(x01, keep_pix, mode, variants, sigma, gen,
                        blend_alpha=0.5):
    if mode == "white_noise":
        ref = torch.rand(keep_pix.shape[0], *x01.shape[1:], device=x01.device,
                         generator=gen)
    elif mode == "blend":
        noise = torch.rand(keep_pix.shape[0], *x01.shape[1:],
                           device=x01.device, generator=gen)
        ref = (1.0 - blend_alpha) * variants["blur"] + blend_alpha * noise
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
# LIME core (unchanged).
# =============================================================================
def cell_id_map(H, W, grid, device):
    gh, gw = grid
    ys = (torch.arange(H, device=device) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W, device=device) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)


def sample_Z(n, n_cells, mask_prob, gen, anchor=True):
    Z = (torch.rand(n, n_cells, generator=gen) > mask_prob).float()
    if anchor and n > 0:
        Z[0] = 1.0
    return Z


def weighted_ridge(Z, y, w, alpha=1.0):
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
    model, mean, std = build_model(args.mid_layer, device)

    grid = (args.grid, args.grid)
    n_cells = grid[0] * grid[1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    gen_gpu = torch.Generator(device=device).manual_seed(args.seed)

    img = Image.open(args.input).convert("RGB").resize(
        (args.work_res, args.work_res), Image.BILINEAR)
    x01 = torch.from_numpy(np.asarray(img, np.float32) / 255.0)\
        .permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, H, W = x01.shape

    cells = cell_id_map(H, W, grid, device)
    variants = make_fill_variants(x01, args.sigma)
    fill = args.default_mask
    print(f"[*] standard LIME — fill='{fill}'  mask_prob={args.mask_prob}  "
          f"grid={grid[0]}x{grid[1]}  sigma={args.sigma}  "
          f"residual_layer={args.mid_layer}")

    # Clean top-1 = LIME target. Also discover the activation map size.
    mid0, _, clean_probs = forward_all(model, x01, mean, std)
    Ha, Wa = mid0.shape[-2], mid0.shape[-1]
    clean_probs = clean_probs.cpu().numpy()
    target = int(clean_probs[0].argmax())
    print(f"[*] LIME target = top-1 class {target} "
          f"(clean prob {clean_probs[0, target]:.4f})")
    print(f"[*] {args.mid_layer} activation grid = {Ha}x{Wa}")

    # Calibration: per-position channel PCA on the mid-layer activation.
    calib = None
    cache_key = _calib_cache_key(args)
    if args.calib_cache:
        calib = load_calibration(args.calib_cache, cache_key)
        if calib is not None:
            print(f"[*] loaded calibration cache {args.calib_cache} "
                  f"(C={calib['C']}, pca_k={calib['k']}, "
                  f"evr={calib['evr']:.3f}) — skipping refit")
    if calib is None:
        print(f"[*] fitting per-position calibration from {args.calib_glob} "
              f"(layer={args.mid_layer}, pca_dim={args.pca_dim}, "
              f"score={args.score}) ...")
        calib = fit_calibration(model, mean, std, args.calib_glob, device,
                                args.batch_size, args.pca_dim, args.work_res,
                                max_cells_for_pca=args.max_cells_for_pca,
                                print_spectrum=args.print_spectrum,
                                seed=args.seed)
        if args.calib_cache:
            save_calibration(args.calib_cache, calib, cache_key)
            print(f"[*] saved calibration cache -> {args.calib_cache}")
    print(f"    n_calib_imgs={calib['n_calib_imgs']}  C={calib['C']}  "
          f"cells_pooled={calib['n_cells_total']}  pca_k={calib['k']}  "
          f"evr={calib['evr']:.3f}")
    cs = calib_dist(calib, args.score)
    thr = float(np.quantile(cs, args.threshold_quantile))
    print(f"    [score={args.score}] per-cell on-manifold threshold "
          f"(q={args.threshold_quantile}) = {thr:.6f}")

    # Grid->activation overlap (for KEPT-restricted residual).
    overlap = grid_to_act_overlap(grid, (Ha, Wa))   # (Ha*Wa, G*G)

    # Accumulators for the LIME map.
    keep_Z, keep_y = [], []
    # Accumulators for the diagnostic (per perturbation scalars).
    all_resid_all = []      # mean residual over ALL act cells
    all_resid_kept = []     # mean residual over KEPT act cells (weighted)
    all_frac_kept_onman = []  # frac of KEPT act cells below thr
    all_resid_norm_all = []
    all_resid_ang_all = []

    def process_batch(zb_cpu):
        zb = zb_cpu.to(device)
        keep_pix = zb[:, cells].unsqueeze(1)
        comp = build_perturbations(x01, keep_pix, fill, variants, args.sigma,
                                   gen_gpu, blend_alpha=args.blend_alpha)
        mid, _, probs = forward_all(model, comp, mean, std)
        tgt = probs[:, target].cpu().numpy()
        mid_np = mid.cpu().numpy()                       # (B,C,Ha,Wa)
        B = mid_np.shape[0]

        # --- LIME map (standard, all rows kept) -----------------------------
        for i in range(B):
            keep_Z.append(zb_cpu[i].numpy())
            keep_y.append(float(tgt[i]))

        # --- per-cell residual map for the active score ---------------------
        rmap = mid_to_resid_maps(mid_np, calib["pca"], args.score)  # (B,Ha,Wa)
        rmap_flat = rmap.reshape(B, Ha * Wa)

        # ALL-cell aggregates (whole perturbed image realism).
        all_resid_all.append(rmap_flat.mean(axis=1)
                             if args.score == "residual"
                             else mid_to_resid_maps(mid_np, calib["pca"],
                                                    "residual")
                                  .reshape(B, -1).mean(axis=1))
        all_resid_norm_all.append(
            mid_to_resid_maps(mid_np, calib["pca"], "residual_norm")
            .reshape(B, -1).mean(axis=1))
        all_resid_ang_all.append(
            mid_to_resid_maps(mid_np, calib["pca"], "residual_ang")
            .reshape(B, -1).mean(axis=1))

        # KEPT-restricted aggregate (realism of kept content).
        kept_frac = kept_fraction_per_act_cell(
            zb_cpu.numpy(), overlap)                      # (B, Ha*Wa)
        wsum = kept_frac.sum(axis=1)                      # (B,)
        wsum_safe = np.maximum(wsum, 1e-12)
        kept_resid = (rmap_flat * kept_frac).sum(axis=1) / wsum_safe
        all_resid_kept.append(kept_resid)

        # frac of KEPT act-cells on-manifold (weighted survival rate).
        below = (rmap_flat < thr).astype(np.float64)
        frac_kept_on = (below * kept_frac).sum(axis=1) / wsum_safe
        all_frac_kept_onman.append(frac_kept_on)

    Z = sample_Z(args.n_samples, n_cells, args.mask_prob, gen, anchor=True)
    print(f"[*] sampling {args.n_samples} perturbations ...")
    for s in range(0, args.n_samples, args.batch_size):
        process_batch(Z[s:s + args.batch_size])

    Znp = np.stack(keep_Z).astype(np.float64)
    y = np.asarray(keep_y, dtype=np.float64)
    weights = lime_weights(Znp, args.kernel_width)
    coefs, intercept = weighted_ridge(Znp, y, weights, alpha=args.ridge_alpha)

    # Optional residual-aware (kept-restricted) reweighted map for comparison.
    coefs_raw = None
    resid_kept_vec = np.concatenate(all_resid_kept)
    if args.weight_by_kept_residual:
        # Soft weight q_i = exp(-lambda * kept_residual_i). Uses KEPT residual
        # (realism of the regions the mask claims matter), NOT the penultimate
        # feature, so it does not reward on-manifold-from-fill.
        q = np.exp(-args.kept_lambda * resid_kept_vec)
        coefs_raw = coefs
        coefs, intercept = weighted_ridge(Znp, y, weights * q,
                                          alpha=args.ridge_alpha)

    # ----- LIMEScore aggregates ---------------------------------------------
    resid_all = np.concatenate(all_resid_all)
    resid_norm_all = np.concatenate(all_resid_norm_all)
    resid_ang_all = np.concatenate(all_resid_ang_all)
    frac_kept_on = np.concatenate(all_frac_kept_onman)
    n_all = resid_all.shape[0]

    mean_resid_all = float(resid_all.mean())
    mean_resid_norm_all = float(resid_norm_all.mean())
    mean_resid_ang_all = float(resid_ang_all.mean())
    mean_resid_kept = float(resid_kept_vec.mean())
    median_resid_kept = float(np.median(resid_kept_vec))
    mean_frac_kept_on = float(frac_kept_on.mean())

    print("")
    print("============== LIMEScore (mid-layer per-position residual) ========")
    print(f"  residual layer          : {args.mid_layer} ({Ha}x{Wa}, C={calib['C']})")
    print(f"  fill mode               : {fill}")
    print(f"  mask_prob               : {args.mask_prob}")
    print(f"  perturbations scored     : {n_all}")
    print(f"  active score             : {args.score}")
    print(f"  -- KEPT-restricted (realism of kept content; the de-circularized number) --")
    print(f"  mean_kept_residual       : {mean_resid_kept:.6f}   (LOWER better)")
    print(f"  median_kept_residual     : {median_resid_kept:.6f}")
    print(f"  mean_frac_kept_onmanifold: {mean_frac_kept_on:.4f}   (HIGHER better)")
    print(f"  -- ALL-cell (whole perturbed image realism) --")
    print(f"  mean_residual (raw)      : {mean_resid_all:.6f}")
    print(f"  mean_residual_norm       : {mean_resid_norm_all:.6f}")
    print(f"  mean_residual_ang        : {mean_resid_ang_all:.6f}")
    print("===================================================================")
    print("")

    coef_t = torch.tensor(coefs, dtype=torch.float32, device=device)
    attr = coef_t[cells].cpu().numpy()

    # ----- Faithfulness AUC of final map (unchanged metrics module) ---------
    avg_metrics = None
    if not args.no_metrics:
        from metrics import average_insertion_deletion, DEFAULT_FILLS
        metric_fills = args.metric_fills if args.metric_fills else DEFAULT_FILLS
        print(f"[*] evaluating insertion/deletion AUC over fills={metric_fills} "
              f"(n_steps={args.metric_steps}) ...")
        # NOTE: metrics.py expects (feature_net, fc, ...) in the old signature.
        # We provide a small adapter so the head + features come from the tap.
        feature_net = nn.Sequential(
            model.stem, model.blocks["layer1"], model.blocks["layer2"],
            model.blocks["layer3"], model.blocks["layer4"],
            model.avgpool, model.flatten).eval().to(device)
        fc = model.fc.eval().to(device)
        avg_metrics = average_insertion_deletion(
            coefs.reshape(grid[0], grid[1]), x01, grid,
            feature_net, fc, mean, std, target,
            fills=metric_fills, variants=variants, sigma=args.sigma,
            n_steps=args.metric_steps, batch_size=args.batch_size,
            seed=args.seed)
        print("")
        print("=========== Faithfulness AUC (final attribution) ==============")
        print(f"  fills averaged          : {avg_metrics['fills']}")
        print(f"  avg_insertion (HIGHER better) : "
              f"{avg_metrics['avg_insertion']:.4f}  "
              f"(+/- {avg_metrics['std_insertion']:.4f})")
        print(f"  avg_deletion  (LOWER better)  : "
              f"{avg_metrics['avg_deletion']:.4f}  "
              f"(+/- {avg_metrics['std_deletion']:.4f})")
        for f in avg_metrics["fills"]:
            pf = avg_metrics["per_fill"][f]
            print(f"    {f:<12} ins={pf['insertion_auc']:.4f}  "
                  f"del={pf['deletion_auc']:.4f}")
        print("===============================================================")
        print("")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{fill}_mp{args.mask_prob:.2f}_{args.mid_layer}"

    def op(name, ext):
        return os.path.join(args.out_dir, f"{name}__{tag}.{ext}")

    np.save(op("attribution", "npy"), attr)
    np.save(op("coefs", "npy"), coefs.reshape(grid[0], grid[1]))
    np.save(op("kept_residual_per_mask", "npy"), resid_kept_vec)
    if coefs_raw is not None:
        np.save(op("coefs_plain", "npy"), coefs_raw.reshape(grid[0], grid[1]))

    nrm = (attr - attr.min()) / (np.ptp(attr) + 1e-12)
    heat_u8 = (255 * nrm).astype(np.uint8)
    Image.fromarray(heat_u8).save(op("heatmap", "png"))
    cmap = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    base_u8 = (x01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = (0.5 * base_u8 + 0.5 * cmap).astype(np.uint8)
    Image.fromarray(overlay).save(op("overlay", "png"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(resid_kept_vec, bins=60, alpha=0.7,
                label=f"kept-residual per mask ({fill})")
        ax.hist(cs, bins=60, alpha=0.4,
                weights=np.full_like(cs, n_all / max(len(cs), 1)),
                label="calibration cells")
        ax.axvline(thr, color="k", ls="--", lw=1,
                   label=f"thr (q={args.threshold_quantile})")
        ax.set_xlabel(f"{args.score} (kept-restricted, lower=on-manifold)")
        ax.set_ylabel("count")
        ax.set_title(f"mean_kept_resid={mean_resid_kept:.3f}  "
                     f"frac_kept_on={mean_frac_kept_on:.3f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(op("kept_resid_hist", "png"), dpi=120)
        plt.close(fig)
        wrote_hist = True
    except Exception as e:
        wrote_hist = False
        print(f"[warn] histogram skipped ({e})")

    with open(op("summary", "txt"), "w") as f:
        f.write(f"input: {args.input}\ntarget_class: {target}\n")
        f.write(f"grid: {grid[0]}x{grid[1]}\nfill: {fill}\n")
        f.write(f"mask_prob: {args.mask_prob}\nn_samples: {args.n_samples}\n")
        f.write(f"kernel_width: {args.kernel_width}  ridge_alpha: "
                f"{args.ridge_alpha}\nridge_intercept: {intercept:.6f}\n")
        f.write("--- on-manifold diagnostic (mid-layer per-position) ---\n")
        f.write(f"residual_layer: {args.mid_layer}\n")
        f.write(f"act_grid: {Ha}x{Wa}  C: {calib['C']}\n")
        f.write(f"calib_glob: {args.calib_glob}\n")
        f.write(f"pca_dim: {calib['k']}  evr: {calib['evr']:.4f}  "
                f"cells_pooled: {calib['n_cells_total']}\n")
        f.write(f"score: {args.score}  threshold_q: "
                f"{args.threshold_quantile}  thr: {thr:.6f}\n")
        f.write(f"LIMEScore.mean_kept_residual: {mean_resid_kept:.6f}\n")
        f.write(f"LIMEScore.median_kept_residual: {median_resid_kept:.6f}\n")
        f.write(f"LIMEScore.mean_frac_kept_onmanifold: {mean_frac_kept_on:.6f}\n")
        f.write(f"LIMEScore.mean_residual_all: {mean_resid_all:.6f}\n")
        f.write(f"LIMEScore.mean_residual_norm_all: {mean_resid_norm_all:.6f}\n")
        f.write(f"LIMEScore.mean_residual_ang_all: {mean_resid_ang_all:.6f}\n")
        if args.weight_by_kept_residual:
            f.write(f"weight_by_kept_residual: True  "
                    f"kept_lambda: {args.kept_lambda}\n")
        if avg_metrics is not None:
            f.write("--- faithfulness AUC (map evaluation, multi-fill avg) ---\n")
            f.write(f"metric_fills: {avg_metrics['fills']}\n")
            f.write(f"avg_insertion_auc: {avg_metrics['avg_insertion']:.6f}\n")
            f.write(f"avg_deletion_auc: {avg_metrics['avg_deletion']:.6f}\n")
            for fn in avg_metrics["fills"]:
                pf = avg_metrics["per_fill"][fn]
                f.write(f"insertion_auc[{fn}]: {pf['insertion_auc']:.6f}\n")
                f.write(f"deletion_auc[{fn}]: {pf['deletion_auc']:.6f}\n")

    outs = (f"attribution__{tag}.npy, coefs__{tag}.npy, "
            f"kept_residual_per_mask__{tag}.npy, heatmap__{tag}.png, "
            f"overlay__{tag}.png, summary__{tag}.txt")
    if coefs_raw is not None:
        outs += f", coefs_plain__{tag}.npy"
    if wrote_hist:
        outs += f", kept_resid_hist__{tag}.png"
    print(f"[*] wrote {outs} to {args.out_dir}/")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Standard grid-LIME (ResNet-50) + mid-layer per-position "
                    "on-manifold residual (kept-restricted).")
    ap.add_argument("--input", required=True)

    # ---- Standard LIME knobs ----
    ap.add_argument("--default-mask", default="blur", choices=FILL_MODES)
    ap.add_argument("--mask-prob", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--kernel-width", type=float, default=0.25)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=11.0)

    # ---- Mid-layer residual knobs ----
    ap.add_argument("--mid-layer", default="layer3", choices=MID_LAYERS,
                    help="ResNet block whose activation map is used for the "
                         "per-position residual. Default layer3 (14x14).")
    ap.add_argument("--calib-glob", default="sample_1k/*.JPEG")
    ap.add_argument("--calib-cache", default=None,
                    help="Path to an .npz calibration cache. If it exists and "
                         "its key (layer/glob/pca_dim/work_res/seed/maxcells) "
                         "matches, calibration is LOADED instead of refit — "
                         "this is the big win in a sweep. If absent, fit then "
                         "save. Delete the file to force a refit.")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="PCA subspace dim over the C-dim channel vectors.")
    ap.add_argument("--score", default="residual_norm",
                    choices=["residual", "residual_norm", "residual_ang"],
                    help="Per-cell residual variant. residual_norm is scale-"
                         "free (recommended for cross-fill).")
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--max-cells-for-pca", type=int, default=200_000,
                    help="Subsample cap for the PCA FIT (stats use full pool).")
    ap.add_argument("--print-spectrum", action="store_true",
                    help="Print the per-position channel-PCA spectrum so you "
                         "can check the chosen layer supports a low-rank "
                         "residual (layer3 flattened was degenerate; the "
                         "per-position CxC object usually is not).")

    # ---- Optional residual-aware map (kept-restricted weighting) ----
    ap.add_argument("--weight-by-kept-residual", action="store_true",
                    help="Also emit a map reweighted by exp(-lambda * "
                         "kept_residual). Uses realism of KEPT content only.")
    ap.add_argument("--kept-lambda", type=float, default=2.0)

    # ---- Faithfulness-metric knobs ----
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--metric-fills", nargs="*", default=None, choices=FILL_MODES)
    ap.add_argument("--metric-steps", type=int, default=50)

    # ---- Misc ----
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--work-res", type=int, default=224)
    ap.add_argument("--out-dir", default="limescore_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blend-alpha", type=float, default=0.5)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())