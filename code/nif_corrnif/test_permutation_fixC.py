"""
Fix C (PartialConditionedNIF) causal-substitution permutation test.

STATUS (2026-07-06): this script is CODE-COMPLETE but NOT YET RUNNABLE — Fix C has not
been trained (gate APPROVE_PARTIAL_CONDITIONING_NIF_TRAINING not fired, GPU reserved for
the P1 box-convergence campaign, see AGENTS.md "P1/P2 STATUS SYNC"). Training Fix C is a
GPU run of unknown-but-likely-multi-hour duration (comparable to prior NIF retrains,
~4-6h) — schedule it as one of the overnight/unattended long runs, sequentially after (or
instead of, if only one slot is available) the P1 seed-repeat run, never concurrently
(one GPU, anti-throttling protocol). Once `pinn_training/runs/<run_name>/best.pt` exists,
point CHECKPOINT_PATH below at it and this script runs standalone (inference only, no
GPU strictly required though much faster with one).

WHAT THIS TESTS (why a same-seed-vs-cross-seed IoU/correlation was deemed insufficient
by itself, see AGENTS.md decision log 2026-07-06): does Fix C's prediction of the TARGET
region (upper z-slices) actually respond causally to a substituted OBSERVED region (lower
z-slices) from a DIFFERENT realization of the SAME alpha, or does it ignore the
conditioning and reproduce a memorized/generic structure regardless of input? Grounded in
real data: alpha=80 and alpha=85 each have 3 independent seeds (realization_id 5,6,7 and
8,9,10 respectively, see build_levelA_helical_nif_partial_conditioning_dataset.py's
source_voxels list) — enough for a real same-seed-vs-cross-seed distribution (3 same-seed
pairs, 6 cross-seed pairs per alpha) and a Mann-Whitney U test, not just a single
anecdotal pair.

Metrics computed per (query_realization, source_realization) pair, NOT just IoU (per the
2026-07-06 correction that IoU alone risks "disguised qualitative validation"):
  1. Pearson correlation of vertical occupation profiles rho_pred(z) vs rho_true(z)
     (measures fidelity of the learned vertical propagation)
  2. z-lag at max cross-correlation between rho_pred(z) and rho_true(z)
     (measures systematic vertical shift artifacts — memorization/smoothing tell)
  3. IoU (kept as a SECONDARY, not primary, metric — same as before)

Success criterion: same-realization correlation distribution significantly higher than
cross-realization distribution (Mann-Whitney U, alpha=0.05), for BOTH alpha=80 and
alpha=85 independently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.nif_partial_conditioning import PartialConditionedNIF  # noqa: E402

DATASET_DIR = Path(__file__).parent / "datasets" / "levelA_helical_nif_partial_conditioning"
# 2026-07-09: rebuilt dataset (n=8 at alpha=85, seeds 0-7) and retrained both
# variants. CHECKPOINT_PATH selects which retrained run to test; set via
# CHECKPOINT_VARIANT env var ("damped" or "unclamped", default "unclamped")
# so both can be run without editing this file between invocations.
_VARIANT_RUN_DIRS = {
    "damped": "nif_partial_conditioning_20260709_103242",
    "unclamped": "nif_partial_conditioning_unclamped_20260709_175837",
}
_variant = os.environ.get("CHECKPOINT_VARIANT", "unclamped")
CHECKPOINT_PATH = (Path(__file__).parent / "runs" /
                    _VARIANT_RUN_DIRS[_variant] / "checkpoints" / "best.pt")

# realization_id -> (alpha, seed), from build_levelA_helical_nif_partial_conditioning_dataset.py
# (rebuilt 2026-07-09: alpha=80 unchanged at n=3 [ids 5-7]; alpha=85 extended
# to n=8 [ids 8-15, seeds 0-7] -- read directly from datasets/.../metadata.json)
REALIZATION_MAP = {
    0: (60.0, 0), 1: (65.0, 0), 2: (70.0, 0), 3: (72.0, 0), 4: (75.0, 0),
    5: (80.0, 0), 6: (80.0, 1), 7: (80.0, 2),
    8: (85.0, 0), 9: (85.0, 1), 10: (85.0, 2), 11: (85.0, 3),
    12: (85.0, 4), 13: (85.0, 5), 14: (85.0, 6), 15: (85.0, 7),
    16: (87.0, 0), 17: (89.0, 0),
}
ALPHA_TO_REALIZATIONS = {80.0: [5, 6, 7], 85.0: [8, 9, 10, 11, 12, 13, 14, 15]}

N_Z_BINS = 40  # vertical profile resolution for the target region


def load_model(device):
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    # 2026-07-08: film_scale is a plain float attribute, not a persisted tensor/
    # buffer, so load_state_dict alone would silently leave it at the class
    # default (0.1) even for a checkpoint trained with a different value (e.g. the
    # unclamped film_scale=1.0 variant) -- read it back from the saved config to
    # avoid silently evaluating the wrong architecture variant.
    cfg = state.get("config", {})
    model = PartialConditionedNIF(
        hidden=int(cfg.get("hidden", 256)), n_layers=int(cfg.get("n_layers", 4)),
        n_freq=int(cfg.get("n_freq", 32)), latent_dim=int(cfg.get("pc_latent_dim", 32)),
        film_scale=float(cfg.get("film_scale", 0.1)),
    )
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device).eval()
    print(f"  loaded film_scale={model.film_scale}")
    return model


def load_split_and_maps():
    data = np.load(DATASET_DIR / "test_samples.npz")["data"]  # (N,6)
    obs_maps = np.load(DATASET_DIR / "observed_maps.npz")["observed_maps"]  # (13,50,50)
    return data, obs_maps


def vertical_profile(z_norm: np.ndarray, values: np.ndarray, n_bins=N_Z_BINS):
    """Bin (z, occupancy) into a mean profile rho(z), z in [0,1] (target-region-only range)."""
    edges = np.linspace(z_norm.min(), z_norm.max(), n_bins + 1)
    idx = np.clip(np.digitize(z_norm, edges) - 1, 0, n_bins - 1)
    profile = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            profile[b] = values[mask].mean()
    return profile


def predict_target_for_realization(model, data, query_realization_id: int,
                                    source_realization_id: int, obs_maps, device):
    """Predict occupancy for query_realization's target-region (x,y,z,alpha) points,
    but conditioned on source_realization's observed map instead of its own."""
    mask = data[:, 4].astype(int) == query_realization_id
    rows = data[mask]
    xyz_alpha = torch.tensor(rows[:, :4], dtype=torch.float32, device=device)
    obs_map = torch.tensor(obs_maps[source_realization_id], dtype=torch.float32, device=device)
    obs_map = obs_map.unsqueeze(0).unsqueeze(0).expand(len(rows), 1, *obs_map.shape)

    with torch.no_grad():
        pred = torch.sigmoid(model(xyz_alpha, obs_map)).cpu().numpy().ravel()

    z_norm = rows[:, 2]
    true_label = rows[:, 5]
    return z_norm, pred, true_label


def pearson_and_lag(z, pred_vals, true_vals):
    pred_profile = vertical_profile(z, pred_vals)
    true_profile = vertical_profile(z, true_vals)
    valid = ~(np.isnan(pred_profile) | np.isnan(true_profile))
    if valid.sum() < 5:
        return np.nan, np.nan
    p, t = pred_profile[valid], true_profile[valid]
    if p.std() == 0 or t.std() == 0:
        return 0.0, np.nan
    corr = np.corrcoef(p, t)[0, 1]

    xc = np.correlate(p - p.mean(), t - t.mean(), mode="full")
    lags = np.arange(-len(p) + 1, len(p))
    z_step = (z.max() - z.min()) / N_Z_BINS
    lag_nm = lags[np.argmax(xc)] * z_step  # in normalized-z units; convert if a physical
    # target-region thickness is known for this alpha/realization
    return corr, lag_nm


def iou(pred_vals, true_vals, thresh=0.5):
    p = pred_vals >= thresh
    t = true_vals >= thresh
    inter = (p & t).sum()
    union = (p | t).sum()
    return inter / union if union > 0 else np.nan


def run_permutation_test_for_alpha(model, data, obs_maps, alpha, device):
    realizations = ALPHA_TO_REALIZATIONS[alpha]
    same_corrs, cross_corrs = [], []
    print(f"\n=== alpha={alpha} ===")
    for query_id in realizations:
        for source_id in realizations:
            z, pred, true = predict_target_for_realization(
                model, data, query_id, source_id, obs_maps, device)
            corr, lag = pearson_and_lag(z, pred, true)
            iou_val = iou(pred, true)
            tag = "SAME" if query_id == source_id else "cross"
            print(f"  query={query_id} source={source_id} [{tag:>5}]  "
                  f"pearson={corr:.3f}  z_lag={lag:.3f}  IoU={iou_val:.3f}")
            (same_corrs if query_id == source_id else cross_corrs).append(corr)
    return np.array(same_corrs), np.array(cross_corrs)


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"NOT RUNNABLE YET: {CHECKPOINT_PATH} does not exist.")
        print("Fix C has not been trained (GPU reserved for the P1 box-convergence "
              "campaign as of 2026-07-06). Train it first via:")
        print("  conda activate gladwsl && python3 train_nif_partial_conditioning.py "
              "--config configs/nif_partial_conditioning_config.yaml --allow-train")
        print("(after issuing APPROVE_PARTIAL_CONDITIONING_NIF_TRAINING per the file's "
              "own gate check) — schedule as an overnight long run alongside/after the "
              "P1 seed-repeat, never concurrently on this single GPU.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    data, obs_maps = load_split_and_maps()

    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None

    for alpha in (80.0, 85.0):
        same, cross = run_permutation_test_for_alpha(model, data, obs_maps, alpha, device)
        print(f"  same-realization correlations:  mean={np.nanmean(same):.3f}  "
              f"(n={len(same)})")
        print(f"  cross-realization correlations: mean={np.nanmean(cross):.3f}  "
              f"(n={len(cross)})")
        if mannwhitneyu is not None:
            stat, p = mannwhitneyu(same, cross, alternative="greater")
            verdict = "PASS (causal signal)" if p < 0.05 else "FAIL (not distinguishable from cross-seed)"
            print(f"  Mann-Whitney U: p={p:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
