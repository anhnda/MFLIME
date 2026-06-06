#!/usr/bin/env python3
"""
sweep.py — in-process blend sweep over images x alphas.

This is the fast equivalent of blend_sweep.sh. The slow part of the bash
version is NOT the per-perturbation forward passes (those are already batched
inside LIMEScore.run); it is that every one of the 50*11 = 550 invocations
rebuilds the model, re-inits a CUDA context, and REFITS calibration from
1000 images. Calibration does not depend on the input image or alpha, so it
is pure repeated work.

Here we build the model once, fit (or cache-load) calibration once, then loop.
We reuse LIMEScore's own functions so the science is identical — this file is
only a driver. Results are appended to a CSV, one row per (image, alpha).

Usage:
  python sweep.py --calib-glob sample_1k/*.JPEG \
      --img-glob 'benchmark_50/*.JPEG' \
      --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
      --n-samples 2500 --mask-prob 0.5 --mid-layer layer3 \
      --score residual_norm --out-csv blend_sweep.csv

Add --no-metrics to skip insertion/deletion (much faster; do this first to
sanity-check the residual numbers before paying for AUC).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import numpy as np
from PIL import Image
import torch

import LIMEScore as L


def _load_x01(path, work_res, device):
    img = Image.open(path).convert("RGB").resize((work_res, work_res),
                                                 Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, np.float32) / 255.0)\
        .permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def score_one(model, mean, std, calib, overlap, x01, target, grid, fill,
              alpha, mask_prob, args, device, gen_seed):
    """Run the residual diagnostic (and optionally the LIME map + AUC) for one
    (image, fill, mask_prob, alpha). Returns a flat dict of metrics. Mirrors
    LIMEScore.run but without any per-call model build or calibration fit."""
    n_cells = grid[0] * grid[1]
    H = W = args.work_res
    cells = L.cell_id_map(H, W, grid, device)
    variants = L.make_fill_variants(x01, args.sigma)
    # recover Ha,Wa from one pass
    mid0, _, _ = L.forward_all(model, x01, mean, std)
    Ha, Wa = mid0.shape[-2], mid0.shape[-1]

    cs = L.calib_dist(calib, args.score)
    thr = float(np.quantile(cs, args.threshold_quantile))

    gen = torch.Generator(device="cpu").manual_seed(gen_seed)
    gen_gpu = torch.Generator(device=device).manual_seed(gen_seed)

    keep_Z, keep_y = [], []
    resid_kept_chunks, frac_kept_chunks = [], []
    resid_all_chunks, resid_norm_chunks, resid_ang_chunks = [], [], []

    Z = L.sample_Z(args.n_samples, n_cells, mask_prob, gen, anchor=True)
    for s in range(0, args.n_samples, args.batch_size):
        zb_cpu = Z[s:s + args.batch_size]
        zb = zb_cpu.to(device)
        keep_pix = zb[:, cells].unsqueeze(1)
        comp = L.build_perturbations(x01, keep_pix, fill, variants, args.sigma,
                                     gen_gpu, blend_alpha=alpha)
        mid, _, probs = L.forward_all(model, comp, mean, std)
        tgt = probs[:, target].cpu().numpy()
        mid_np = mid.cpu().numpy()
        B = mid_np.shape[0]

        for i in range(B):
            keep_Z.append(zb_cpu[i].numpy())
            keep_y.append(float(tgt[i]))

        rmap = L.mid_to_resid_maps(mid_np, calib["pca"], args.score)
        rflat = rmap.reshape(B, Ha * Wa)
        resid_all_chunks.append(
            L.mid_to_resid_maps(mid_np, calib["pca"], "residual")
            .reshape(B, -1).mean(axis=1))
        resid_norm_chunks.append(
            L.mid_to_resid_maps(mid_np, calib["pca"], "residual_norm")
            .reshape(B, -1).mean(axis=1))
        resid_ang_chunks.append(
            L.mid_to_resid_maps(mid_np, calib["pca"], "residual_ang")
            .reshape(B, -1).mean(axis=1))

        kept_frac = L.kept_fraction_per_act_cell(zb_cpu.numpy(), overlap)
        wsum = np.maximum(kept_frac.sum(axis=1), 1e-12)
        resid_kept_chunks.append((rflat * kept_frac).sum(axis=1) / wsum)
        below = (rflat < thr).astype(np.float64)
        frac_kept_chunks.append((below * kept_frac).sum(axis=1) / wsum)

    resid_kept = np.concatenate(resid_kept_chunks)
    frac_kept = np.concatenate(frac_kept_chunks)
    resid_all = np.concatenate(resid_all_chunks)
    resid_norm = np.concatenate(resid_norm_chunks)
    resid_ang = np.concatenate(resid_ang_chunks)

    row = {
        "thr": thr,
        "mean_kept_residual": float(resid_kept.mean()),
        "median_kept_residual": float(np.median(resid_kept)),
        "mean_frac_kept_onmanifold": float(frac_kept.mean()),
        "mean_residual_all": float(resid_all.mean()),
        "mean_residual_norm_all": float(resid_norm.mean()),
        "mean_residual_ang_all": float(resid_ang.mean()),
    }

    if not args.no_metrics:
        import torch.nn as nn
        from metrics import average_insertion_deletion, DEFAULT_FILLS
        Znp = np.stack(keep_Z).astype(np.float64)
        yv = np.asarray(keep_y, dtype=np.float64)
        w = L.lime_weights(Znp, args.kernel_width)
        coefs, _ = L.weighted_ridge(Znp, yv, w, alpha=args.ridge_alpha)
        feature_net = nn.Sequential(
            model.stem, model.blocks["layer1"], model.blocks["layer2"],
            model.blocks["layer3"], model.blocks["layer4"],
            model.avgpool, model.flatten).eval().to(device)
        fc = model.fc.eval().to(device)
        m = average_insertion_deletion(
            coefs.reshape(grid[0], grid[1]), x01, grid,
            feature_net, fc, mean, std, target,
            fills=(args.metric_fills or DEFAULT_FILLS), variants=variants,
            sigma=args.sigma, n_steps=args.metric_steps,
            batch_size=args.batch_size, seed=gen_seed)
        row["insertion"] = float(m["avg_insertion"])
        row["deletion"] = float(m["avg_deletion"])

    return row


def main():
    ap = argparse.ArgumentParser(description="In-process blend sweep.")
    ap.add_argument("--img-glob", required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                             0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--out-csv", default="sweep.csv")

    # Sweep axes. The driver iterates the FULL grid fills x mask_probs, and
    # additionally x alphas for the 'blend' fill only (alpha is meaningless for
    # the other fills). This single driver subsumes blend_sweep / maskprob_sweep
    # / run_benchmark: pick the axes you want.
    ap.add_argument("--fills", nargs="+", default=["blend"],
                    choices=L.FILL_MODES,
                    help="Fills to sweep. e.g. 'blur white_noise' for the "
                         "mask-prob study, or 'blur black white corner_mean "
                         "white_noise' for the benchmark.")
    ap.add_argument("--mask-probs", nargs="+", type=float, default=[0.5],
                    help="Mask probabilities to sweep. e.g. 0.1 0.2 ... 0.9 "
                         "for the kappa(Z,W) conditioning study.")

    # mirror the LIMEScore knobs the sweep needs
    ap.add_argument("--mid-layer", default="layer3", choices=L.MID_LAYERS)
    ap.add_argument("--calib-glob", default="sample_1k/*.JPEG")
    ap.add_argument("--calib-cache", default="calib_cache.npz")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--score", default="residual_norm",
                    choices=["residual", "residual_norm", "residual_ang"])
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--max-cells-for-pca", type=int, default=200_000)
    ap.add_argument("--print-spectrum", action="store_true")

    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=2500)
    ap.add_argument("--kernel-width", type=float, default=0.25)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--work-res", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--metric-fills", nargs="*", default=None,
                    choices=L.FILL_MODES)
    ap.add_argument("--metric-steps", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device = {device}")
    model, mean, std = L.build_model(args.mid_layer, device)

    # --- calibration ONCE (cache-aware) ---
    key = L._calib_cache_key(args)
    calib = L.load_calibration(args.calib_cache, key) if args.calib_cache else None
    if calib is not None:
        print(f"[*] loaded calibration cache {args.calib_cache} "
              f"(C={calib['C']}, k={calib['k']}, evr={calib['evr']:.3f})")
    else:
        print(f"[*] fitting calibration once from {args.calib_glob} ...")
        calib = L.fit_calibration(model, mean, std, args.calib_glob, device,
                                  args.batch_size, args.pca_dim, args.work_res,
                                  max_cells_for_pca=args.max_cells_for_pca,
                                  print_spectrum=args.print_spectrum,
                                  seed=args.seed)
        if args.calib_cache:
            L.save_calibration(args.calib_cache, calib, key)
            print(f"[*] saved calibration cache -> {args.calib_cache}")
    print(f"    C={calib['C']}  k={calib['k']}  evr={calib['evr']:.3f}  "
          f"cells_pooled={calib['n_cells_total']}")

    grid = (args.grid, args.grid)
    # activation grid size from one pass on a dummy image of the first match
    images = sorted(glob.glob(args.img_glob))
    if not images:
        raise FileNotFoundError(f"No images matched: {args.img_glob}")
    x0 = _load_x01(images[0], args.work_res, device)
    mid0, _, _ = L.forward_all(model, x0, mean, std)
    Ha, Wa = mid0.shape[-2], mid0.shape[-1]
    overlap = L.grid_to_act_overlap(grid, (Ha, Wa))

    # Build the (fill, mask_prob, alpha) task list. alpha only varies for blend.
    tasks = []
    for fill in args.fills:
        for mp in args.mask_probs:
            if fill == "blend":
                for a in args.alphas:
                    tasks.append((fill, mp, a))
            else:
                tasks.append((fill, mp, 0.0))
    print(f"[*] {args.mid_layer} act grid {Ha}x{Wa}; "
          f"{len(images)} images x {len(tasks)} settings = "
          f"{len(images)*len(tasks)} runs")

    fieldnames = ["image", "fill", "mask_prob", "alpha", "target", "thr",
                  "mean_kept_residual", "median_kept_residual",
                  "mean_frac_kept_onmanifold", "mean_residual_all",
                  "mean_residual_norm_all", "mean_residual_ang_all",
                  "insertion", "deletion"]
    write_header = not os.path.exists(args.out_csv)
    f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    t_start = time.time()
    for ii, img in enumerate(images):
        x01 = _load_x01(img, args.work_res, device)
        _, _, clean_probs = L.forward_all(model, x01, mean, std)
        target = int(clean_probs[0].cpu().numpy().argmax())
        for (fill, mp, a) in tasks:
            t0 = time.time()
            # Seed keyed on (image, fill, mask_prob, alpha) so every run is
            # reproducible and independent. To instead PAIR masks across a
            # single axis (e.g. identical masks for blur vs white_noise at
            # fixed mp, so the only difference is the fill), drop the term for
            # that axis from this expression.
            gen_seed = (args.seed + 1000 * ii
                        + 100 * L.FILL_MODES.index(fill)
                        + int(round(mp * 10)) * 7
                        + int(round(a * 100)))
            row = score_one(model, mean, std, calib, overlap, x01, target,
                            grid, fill, a, mp, args, device, gen_seed)
            row["image"] = os.path.basename(img)
            row["fill"] = fill
            row["mask_prob"] = mp
            row["alpha"] = a
            row["target"] = target
            writer.writerow(row)
            f.flush()
            dt = time.time() - t0
            print(f"[{ii+1}/{len(images)}] {os.path.basename(img)} "
                  f"{fill} mp={mp:.1f} a={a:.1f}  "
                  f"kept_resid={row['mean_kept_residual']:.4f}  "
                  f"frac_on={row['mean_frac_kept_onmanifold']:.3f}"
                  + (f"  ins={row['insertion']:.3f} del={row['deletion']:.3f}"
                     if "insertion" in row else "")
                  + f"  ({dt:.1f}s)")
    f.close()
    print(f"[*] done in {time.time()-t_start:.1f}s -> {args.out_csv}")


if __name__ == "__main__":
    main()