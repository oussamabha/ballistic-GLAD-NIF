"""
Build Level A helical NIF dataset for Tier-2 Fix C (partial-realization
conditioning). Reuses the SAME voxel .npy files as
build_levelA_helical_nif_hardlabel_dataset.py — no new simulation.

Motivation (P2 tier 2, Fix C, 2026-07-06): Fix A/B still ask the network to
predict a SPECIFIC realization's exact occupancy from (x,y,z,alpha) alone.
P1's corotating-frame oracle already showed a fixed spatial template gives
zero gain over the bulk average at every angle — i.e. column (x,y) positions
are an independent per-realization random draw, not a function of alpha.
Predicting them from macroscopic knobs alone is an ill-posed regression onto
a variable the inputs do not determine. Fix C sidesteps this by conditioning
the prediction on part of the SAME realization: split each voxel volume into
an "observed" region near the substrate (which records where nucleation
happened — the seed-specific information macroscopic (x,y,z,alpha) doesn't
carry) and a "target" region higher in the film. This is a completion task
(predict the rest of this specific structure from its own early growth), not
a "predict this seed's random draw from unrelated inputs" task — well-posed
by construction.

Split (documented explicitly per task instructions):
  - Voxel shape: (nx, ny, nz) = (50, 50, 151) for every source file.
  - Z_OBS = round(0.15 * nz) = 23 slices. Observed region = z-slices
    [0, Z_OBS) (bottom ~15%, near-substrate nucleation footprint). Target
    region = z-slices [Z_OBS, nz) (the remaining ~85% of the film height) —
    STRICTLY DISJOINT from the observed region, so this is a genuine
    completion task, not a rephrased identity function.
  - Observed-region encoding: max-pooled binary occupancy over the observed
    z-range, per (x,y) column: observed_map[i,j] = 1 if any voxel in
    arr[i,j,0:Z_OBS] is occupied (> BINARY_THRESHOLD) else 0. This is a
    (50,50) map per realization (one per (alpha,seed) source file), stored
    ONCE per realization (not duplicated per sample) in a separate
    `observed_maps.npz`. A small CNN encoder (see
    models/nif_partial_conditioning.py) turns this map into a fixed-size
    latent vector at TRAINING TIME — the raw map is what's persisted here,
    not a hand-designed pooled feature vector, so the encoder can still learn
    which spatial patterns in the nucleation footprint matter.

Sampling: same balanced 1/3 occupied + 1/3 empty + 1/3 random convention as
the hard-label builder, but restricted to the TARGET region only (z-index
Z_OBS..nz-1) — observed-region voxels are never used as prediction targets.

Output layout (`OUT_DIR`):
  - train_samples.npz / val_samples.npz / test_samples.npz:
      "data": (N, 6) float32 columns
        [x_norm, y_norm, z_norm, alpha_norm, realization_id, label]
      x_norm, y_norm: unit voxel coords in [0,1] (same convention as
        levelA_helical_nif_hardlabel).
      z_norm: (k+0.5)/nz using the FULL nz=151 (so it stays a physically
        meaningful depth coordinate, NOT renormalized to the target
        sub-range) — the network sees where in the whole film the target
        point sits, consistent with how it would need to be used if the
        observed region were held out at inference time on a real (not yet
        fully deposited) run.
      realization_id: integer (stored as float32) index into
        observed_maps.npz["observed_maps"][realization_id] — the SAME id
        space is used across train/val/test splits since observed maps are
        realization-level (not sample-level) and the split shuffles samples,
        not realizations.
      label: binary occupancy (> BINARY_THRESHOLD), same convention as the
        hard-label dataset.
  - observed_maps.npz: "observed_maps" (R, 50, 50) float32, R = number of
    source (alpha,seed) files. Shared identically across all three splits.
  - metadata.json: per-realization (alpha, seed, file, occ_frac) + the exact
    Z_OBS/threshold/sampling parameters used.
  - normalization.json: column documentation for `data`.

Usage:
  python build_levelA_helical_nif_partial_conditioning_dataset.py [--force]
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate project root")


PROJECT_ROOT  = find_project_root()
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"

VOXEL_DIR = PINN_NIF_ROOT / "pinn_training" / "datasets" / "universal_glad_pipeline" / "voxels"
OUT_DIR   = PINN_NIF_ROOT / "pinn_training" / "datasets" / "levelA_helical_nif_partial_conditioning"

BINARY_THRESHOLD = 0.5
Z_OBS_FRAC       = 0.15   # bottom 15% of z-slices = observed (near-substrate) region
SAMPLES_PER_FILE = 50_000
SEED             = 42


def alpha_seed_from_name(path: Path) -> tuple[float, int]:
    parts = path.stem.split("_")
    alpha_part = next(p for p in parts if p.startswith("alpha") and p[5:].isdigit())
    seed_part  = next(p for p in parts if p.startswith("seed")  and p[4:].isdigit())
    return float(alpha_part[5:]), int(seed_part[4:])


def make_observed_map(arr: np.ndarray, z_obs: int) -> np.ndarray:
    """arr: (nx, ny, nz). Returns (nx, ny) binary occupancy footprint over
    the observed z-range [0, z_obs)."""
    observed_slab = arr[:, :, :z_obs]
    return (observed_slab > BINARY_THRESHOLD).any(axis=2).astype(np.float32)


def sample_target_points(
    arr: np.ndarray, alpha: float, realization_id: int, z_obs: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n points from the TARGET region only (z-index >= z_obs)."""
    nx, ny, nz = arr.shape
    target = arr[:, :, z_obs:]  # (nx, ny, nz - z_obs)
    flat     = target.ravel()
    flat_bin = (flat > BINARY_THRESHOLD).astype(np.float32)

    occupied = np.flatnonzero(flat_bin > 0.5)
    empty    = np.flatnonzero(flat_bin < 0.5)

    n_occ   = min(len(occupied), n // 3)
    n_empty = min(len(empty),    n // 3)
    n_rand  = n - n_occ - n_empty

    idx_parts = []
    if n_occ:
        idx_parts.append(rng.choice(occupied, n_occ, replace=len(occupied) < n_occ))
    if n_empty:
        idx_parts.append(rng.choice(empty, n_empty, replace=len(empty) < n_empty))
    idx_parts.append(rng.integers(0, flat.size, n_rand))
    idx = np.concatenate(idx_parts)

    i, j, k_local = np.unravel_index(idx, target.shape)
    k_full = k_local + z_obs   # shift back into full-volume z-index

    x = (i + 0.5) / nx
    y = (j + 0.5) / ny
    z = (k_full + 0.5) / nz    # normalized against FULL nz, not target sub-range
    alpha_norm = (alpha - 74.5) / 14.5
    rho_bin = flat_bin[idx]

    # cols: x_norm, y_norm, z_norm, alpha_norm, realization_id, label
    return np.column_stack([
        x, y, z,
        np.full_like(x, alpha_norm),
        np.full_like(x, float(realization_id)),
        rho_bin,
    ]).astype(np.float32)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build partial-conditioning helical NIF dataset (Fix C).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT_DIR / "metadata.json").exists() and not args.force:
        print("SKIP: dataset already exists. Use --force to rebuild.")
        return 0

    rng   = np.random.default_rng(SEED)
    files = sorted(VOXEL_DIR.glob("rho_LA_helical_alpha_300nm_*.npy"))
    if not files:
        raise SystemExit(f"No voxel files found in {VOXEL_DIR}")

    samples      = []
    observed_maps = []
    meta_rows    = []

    for realization_id, path in enumerate(files):
        alpha, seed = alpha_seed_from_name(path)
        arr = np.load(path).astype(np.float32)
        nx, ny, nz = arr.shape
        z_obs = round(Z_OBS_FRAC * nz)

        obs_map = make_observed_map(arr, z_obs)
        observed_maps.append(obs_map)

        target_bin = (arr[:, :, z_obs:] > BINARY_THRESHOLD).astype(np.float32)
        occ_frac_target = float(target_bin.mean())
        occ_frac_observed = float(obs_map.mean())

        pts = sample_target_points(arr, alpha, realization_id, z_obs, SAMPLES_PER_FILE, rng)
        samples.append(pts)
        meta_rows.append({
            "realization_id": realization_id,
            "file":     str(path.relative_to(PROJECT_ROOT)),
            "alpha":    alpha,
            "seed":     seed,
            "shape":    list(arr.shape),
            "z_obs":    z_obs,
            "occ_frac_observed_region": round(occ_frac_observed, 6),
            "occ_frac_target_region":   round(occ_frac_target, 6),
        })
        print(f"  id={realization_id:>2}  alpha={alpha:.0f} seed={seed}  z_obs={z_obs}  "
              f"obs_occ={occ_frac_observed:.4f}  target_occ={occ_frac_target:.4f}  pts={len(pts)}")

    data = np.concatenate(samples, axis=0)
    rng.shuffle(data)  # shuffles SAMPLE rows only; realization_id column travels with each row,
                        # so observed_maps.npz indexing stays valid after shuffle/split.
    n = len(data)
    t1, t2 = int(0.8 * n), int(0.9 * n)

    np.savez_compressed(OUT_DIR / "train_samples.npz", data=data[:t1])
    np.savez_compressed(OUT_DIR / "val_samples.npz",   data=data[t1:t2])
    np.savez_compressed(OUT_DIR / "test_samples.npz",  data=data[t2:])
    np.savez_compressed(OUT_DIR / "observed_maps.npz",
                         observed_maps=np.stack(observed_maps, axis=0).astype(np.float32))

    (OUT_DIR / "normalization.json").write_text(json.dumps({
        "x_y_z":          "unit voxel coordinates [0,1]; z_norm uses FULL nz=151, not the target sub-range",
        "alpha_norm":     "(alpha_deg-74.5)/14.5",
        "realization_id": "integer index (stored as float32) into observed_maps.npz['observed_maps']",
        "label":          "binary {0,1} — thresholded at 0.5, TARGET region only (z-index >= z_obs)",
        "threshold":      BINARY_THRESHOLD,
        "z_obs_frac":     Z_OBS_FRAC,
        "columns":        ["x_norm", "y_norm", "z_norm", "alpha_norm", "realization_id", "label"],
    }, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "generated_at":      datetime.now().isoformat(timespec="seconds"),
        "label_scheme":      "partial_conditioning_hard_binary",
        "binary_threshold":  BINARY_THRESHOLD,
        "z_obs_frac":        Z_OBS_FRAC,
        "observed_region":   "z-slices [0, z_obs) per realization — near-substrate nucleation footprint",
        "target_region":     "z-slices [z_obs, nz) per realization — STRICTLY DISJOINT from observed region",
        "observed_map_pooling": "max over observed z-range, per (x,y) column, binarized at threshold",
        "source_voxels":     meta_rows,
        "morphology_mode":   "helical",
        "pitch_nm":          150,
        "samples_per_file":  SAMPLES_PER_FILE,
        "n_realizations":    len(files),
        "seed":              SEED,
        "warnings": [
            "Target-region binary labels thresholded at 0.5 from max-normalized bead-count histograms, "
            "same convention as levelA_helical_nif_hardlabel.",
            "observed_maps.npz is REALIZATION-level (one row per source file), not sample-level — "
            "the 'realization_id' column in each split's data array must be used to index into it; "
            "this stays valid across train/val/test because only sample ROWS are shuffled/split, "
            "never the realization_id -> observed_map mapping.",
            "Multi-seed alphas (80, 85) still contribute 3 separate realizations each (3 separate "
            "observed_map rows) — this dataset does NOT pool seeds into a single target distribution "
            "the way levelA_helical_nif_hardlabel does; each realization is conditioned on its OWN "
            "observed region, which is exactly the mechanism intended to resolve the multi-seed "
            "label-conflict issue (P2 Discussion sec:disc_multiseed) without needing seed isolation.",
        ],
    }, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "split_report.md").write_text(
        f"# Level A Partial-Conditioning Split Report\n\n"
        f"train={t1}, val={t2-t1}, test={n-t2}\n"
        f"label_scheme=partial_conditioning_hard_binary, threshold={BINARY_THRESHOLD}\n"
        f"z_obs_frac={Z_OBS_FRAC}, n_realizations={len(files)}\n",
        encoding="utf-8",
    )

    print(f"\nDataset written to: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  train={t1}  val={t2-t1}  test={n-t2}  realizations={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
