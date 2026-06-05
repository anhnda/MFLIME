#!/usr/bin/env python3
"""
metrics.py

Vectorized, batched insertion / deletion AUC for pixel/grid attributions on a
ResNet-50 target, evaluated against MULTIPLE perturbation baselines (fills) and
averaged.

Insertion: start from the fully-perturbed (reference) image, reveal the most
           important cells first, in chunks. AUC of target-prob vs fraction
           revealed. HIGHER = better attribution.
Deletion:  start from the clean image, remove the most important cells first.
           AUC of target-prob vs fraction removed. LOWER = better attribution.

The "ordering" is induced by the attribution: cells sorted by descending
attribution value. The same ordering is scored against each fill baseline; the
per-fill AUCs are then averaged to give a fill-agnostic number (avg_insertion,
avg_deletion).

Design notes
------------
* Vectorized over the insertion/deletion *steps*: every step's composite image
  is built as one (T,3,H,W) batch and run through the model in mini-batches, so
  a full insertion curve for one fill is a handful of forward passes, not T.
* Batched over fills implicitly: each fill is one curve; loop is over fills only.
* Reuses the SAME fill machinery as LIMEScore.py (make_fill_variants /
  build_perturbations) so "blur", "white_noise", "inpaint", etc. mean exactly
  the same thing here as in the explainer.
* Pure forward passes under torch.no_grad(); attribution is treated as a fixed
  ordering and never differentiated.

Public API
----------
insertion_deletion_auc(attr_cells, ...) -> dict   (single fill)
average_insertion_deletion(attr_cells, fills, ...) -> dict  (multi-fill avg)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# Reuse the exact fill/forward machinery from the explainer so baselines match.
from LIMEScore import (
    FILL_MODES,
    build_perturbations,
    cell_id_map,
    forward_feats_probs,
    make_fill_variants,
)

# Default fills to average over (inpaint excluded by default: per-mask cv2
# inpaint is slow; include it explicitly if wanted).
DEFAULT_FILLS: List[str] = ["blur", "black", "white",
                            "corner_mean"]


def _auc_trapz(ys: np.ndarray) -> float:
    """Trapezoidal AUC of a curve sampled on a uniform x-grid in [0,1].

    ys has length T (= number of curve points incl. both endpoints). The x
    spacing is 1/(T-1), so the trapezoid integral equals
    mean of adjacent-point averages."""
    ys = np.asarray(ys, dtype=np.float64)
    if ys.shape[0] < 2:
        return float(ys.mean()) if ys.size else 0.0
    return float(np.trapz(ys, dx=1.0 / (ys.shape[0] - 1)))


def _step_order(attr_cells: np.ndarray, n_cells: int, n_steps: int) -> List[np.ndarray]:
    """Cell ranking (descending attribution) chunked into ~n_steps groups.

    Returns a list of length n_steps; entry t is the array of cell ids whose
    state flips at step t (cumulatively building the curve)."""
    order = np.argsort(-attr_cells.ravel(), kind="stable")  # most important 1st
    n_steps = max(1, min(n_steps, n_cells))
    # near-even chunk boundaries 0..n_cells
    bounds = np.linspace(0, n_cells, n_steps + 1).round().astype(int)
    bounds = np.unique(bounds)
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


@torch.no_grad()
def _score_keep_masks(
    keep_cells: torch.Tensor,        # (T, n_cells) in {0,1}, 1 = sharp/kept
    x01: torch.Tensor,               # (1,3,H,W) in [0,1]
    cells: torch.Tensor,             # (H,W) long cell-id map
    fill: str,
    variants: dict,
    sigma: float,
    feature_net, fc, mean, std,
    target: int,
    batch_size: int,
    gen_gpu: torch.Generator,
) -> np.ndarray:
    """For a stack of cell-keep masks, build composites and return target probs.

    Vectorized: expands (T,n_cells) -> (T,1,H,W) pixel masks, composites in one
    tensor, runs the model in mini-batches of batch_size."""
    device = x01.device
    T = keep_cells.shape[0]
    out = np.empty(T, dtype=np.float64)
    for s in range(0, T, batch_size):
        zb = keep_cells[s:s + batch_size].to(device)            # (b, n_cells)
        keep_pix = zb[:, cells].unsqueeze(1)                    # (b,1,H,W)
        comp = build_perturbations(x01, keep_pix, fill, variants, sigma, gen_gpu)
        _, probs = forward_feats_probs(feature_net, fc, comp, mean, std)
        out[s:s + zb.shape[0]] = probs[:, target].cpu().numpy()
    return out


@torch.no_grad()
def insertion_deletion_auc(
    attr_cells: np.ndarray,          # (gh, gw) or (n_cells,) attribution
    x01: torch.Tensor,              # (1,3,H,W) [0,1]
    grid,                           # (gh, gw)
    feature_net, fc, mean, std,
    target: int,
    fill: str = "blur",
    variants: Optional[dict] = None,
    sigma: float = 11.0,
    n_steps: int = 50,
    batch_size: int = 256,
    seed: int = 0,
) -> Dict[str, object]:
    """Insertion & deletion AUC for ONE fill baseline.

    Returns dict with insertion_auc, deletion_auc, and the raw curves."""
    device = x01.device
    gh, gw = grid
    n_cells = gh * gw
    cells = cell_id_map(x01.shape[-2], x01.shape[-1], grid, device)
    if variants is None:
        variants = make_fill_variants(x01, sigma)
    gen_gpu = torch.Generator(device=device).manual_seed(seed)

    attr = np.asarray(attr_cells, dtype=np.float64).ravel()
    if attr.shape[0] != n_cells:
        raise ValueError(f"attr has {attr.shape[0]} cells, grid implies {n_cells}")

    chunks = _step_order(attr, n_cells, n_steps)
    T = len(chunks)

    # ---- Insertion: start all-OFF (reference), cumulatively turn cells ON ---
    ins_masks = torch.zeros(T + 1, n_cells, dtype=torch.float32)
    cur = torch.zeros(n_cells, dtype=torch.float32)
    ins_masks[0] = cur.clone()                       # nothing revealed
    for t, ids in enumerate(chunks):
        cur[ids] = 1.0
        ins_masks[t + 1] = cur.clone()

    # ---- Deletion: start all-ON (clean), cumulatively turn cells OFF -------
    del_masks = torch.ones(T + 1, n_cells, dtype=torch.float32)
    cur = torch.ones(n_cells, dtype=torch.float32)
    del_masks[0] = cur.clone()                        # nothing removed
    for t, ids in enumerate(chunks):
        cur[ids] = 0.0
        del_masks[t + 1] = cur.clone()

    ins_probs = _score_keep_masks(ins_masks, x01, cells, fill, variants, sigma,
                                  feature_net, fc, mean, std, target,
                                  batch_size, gen_gpu)
    del_probs = _score_keep_masks(del_masks, x01, cells, fill, variants, sigma,
                                  feature_net, fc, mean, std, target,
                                  batch_size, gen_gpu)

    return {
        "fill": fill,
        "insertion_auc": _auc_trapz(ins_probs),
        "deletion_auc": _auc_trapz(del_probs),
        "insertion_curve": ins_probs,
        "deletion_curve": del_probs,
        "n_steps": T,
    }


@torch.no_grad()
def average_insertion_deletion(
    attr_cells: np.ndarray,
    x01: torch.Tensor,
    grid,
    feature_net, fc, mean, std,
    target: int,
    fills: Sequence[str] = DEFAULT_FILLS,
    variants: Optional[dict] = None,
    sigma: float = 11.0,
    n_steps: int = 50,
    batch_size: int = 256,
    seed: int = 0,
) -> Dict[str, object]:
    """Insertion/deletion AUC averaged over a list of fill baselines.

    The attribution ordering is fixed; only the perturbation baseline changes.
    Returns avg_insertion, avg_deletion, and the per-fill breakdown."""
    bad = [f for f in fills if f not in FILL_MODES]
    if bad:
        raise ValueError(f"unknown fill(s) {bad}; choose from {FILL_MODES}")
    if variants is None:
        variants = make_fill_variants(x01, sigma)

    per_fill: Dict[str, dict] = {}
    ins_list, del_list = [], []
    for f in fills:
        r = insertion_deletion_auc(
            attr_cells, x01, grid, feature_net, fc, mean, std, target,
            fill=f, variants=variants, sigma=sigma, n_steps=n_steps,
            batch_size=batch_size, seed=seed)
        per_fill[f] = {"insertion_auc": r["insertion_auc"],
                       "deletion_auc": r["deletion_auc"],
                       "insertion_curve": r["insertion_curve"],
                       "deletion_curve": r["deletion_curve"]}
        ins_list.append(r["insertion_auc"])
        del_list.append(r["deletion_auc"])

    return {
        "fills": list(fills),
        "avg_insertion": float(np.mean(ins_list)),
        "avg_deletion": float(np.mean(del_list)),
        "std_insertion": float(np.std(ins_list)),
        "std_deletion": float(np.std(del_list)),
        "per_fill": per_fill,
    }