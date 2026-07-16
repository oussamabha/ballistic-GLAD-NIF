"""
Build Level A helical NIF dataset with HARD binary labels (rho thresholded at 0.5).

Motivation: the soft-label dataset (max-normalized bead-count histograms) places
90-98% of voxel labels in (0.1, 0.9), which prevents BCE from learning binary
spatial structure. This builder binarizes at 0.5 to give crisp 0/1 targets.

Sampling: balanced 1/3 occupied + 1/3 empty + 1/3 random per voxel, consistent
with the original soft-label builder except the occupied/empty split uses the
binary mask directly rather than quantile thresholds.

Usage:
  python build_levelA_helical_nif_hardlabel_dataset.py [--force]
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate project root")


PROJECT_ROOT = find_project_root()
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"

VOXEL_DIR = PINN_NIF_ROOT / "pinn_training" / "datasets" / "universal_glad_pipeline" / "voxels"
OUT_DIR    = PINN_NIF_ROOT / "pinn_training" / "datasets" / "levelA_helical_nif_hardlabel"

BINARY_THRESHOLD = 0.5
SAMPLES_PER_FILE = 50_000
SEED = 42


def alpha_seed_from_name(path: Path) -> tuple[float, int]:
    parts = path.stem.split("_")
    alpha_part = next(p for p in parts if p.startswith("alpha") and p[5:].isdigit())
    seed_part  = next(p for p in parts if p.startswith("seed")  and p[4:].isdigit())
    return float(alpha_part[5:]), int(seed_part[4:])


def sample_points(arr: np.ndarray, alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points from arr; label is binary (arr > BINARY_THRESHOLD)."""
    nx, ny, nz = arr.shape
    flat     = arr.ravel()
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

    i, j, k = np.unravel_index(idx, arr.shape)
    x = (i + 0.5) / nx
    y = (j + 0.5) / ny
    z = (k + 0.5) / nz
    alpha_norm = (alpha - 74.5) / 14.5
    rho_bin    = flat_bin[idx]

    # cols: x_norm, y_norm, z_norm, alpha_norm, rho_binary
    return np.column_stack(
        [x, y, z, np.full_like(x, alpha_norm), rho_bin]
    ).astype(np.float32)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build hard-label helical NIF dataset.")
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

    samples   = []
    meta_rows = []

    for path in files:
        alpha, seed = alpha_seed_from_name(path)
        arr = np.load(path).astype(np.float32)

        arr_bin  = (arr > BINARY_THRESHOLD).astype(np.float32)
        occ_frac = float(arr_bin.mean())

        pts = sample_points(arr, alpha, SAMPLES_PER_FILE, rng)
        samples.append(pts)
        meta_rows.append({
            "file":     str(path.relative_to(PROJECT_ROOT)),
            "alpha":    alpha,
            "seed":     seed,
            "shape":    list(arr.shape),
            "occ_frac_after_threshold": round(occ_frac, 6),
        })
        print(f"  alpha={alpha:.0f} seed={seed}  occ_frac={occ_frac:.4f}  pts={len(pts)}")

    data = np.concatenate(samples, axis=0)
    rng.shuffle(data)
    n = len(data)
    t1, t2 = int(0.8 * n), int(0.9 * n)

    np.savez_compressed(OUT_DIR / "train_samples.npz", data=data[:t1])
    np.savez_compressed(OUT_DIR / "val_samples.npz",   data=data[t1:t2])
    np.savez_compressed(OUT_DIR / "test_samples.npz",  data=data[t2:])

    (OUT_DIR / "normalization.json").write_text(json.dumps({
        "x_y_z":      "unit voxel coordinates [0,1]",
        "alpha_norm":  "(alpha_deg-74.5)/14.5",
        "rho":         "binary {0,1} — thresholded at 0.5 from max-normalized bead-count histogram",
        "threshold":   BINARY_THRESHOLD,
    }, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "label_scheme":     "hard_binary",
        "binary_threshold": BINARY_THRESHOLD,
        "source_voxels":    meta_rows,
        "morphology_mode":  "helical",
        "pitch_nm":         150,
        "samples_per_file": SAMPLES_PER_FILE,
        "seed":             SEED,
        "warnings": [
            "Binary labels thresholded at 0.5 from max-normalized bead-count histograms.",
            "Multi-seed alphas (80, 85) pool voxels from 3 seeds; same (x,y,z) coordinates "
            "may have different binary labels across seeds — averaging into soft intermediate "
            "values is avoided here but pooling still creates seed-dependent spatial patterns.",
        ],
    }, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "split_report.md").write_text(
        f"# Level A Hard-Label Split Report\n\ntrain={t1}, val={t2-t1}, test={n-t2}\n"
        f"label_scheme=hard_binary, threshold={BINARY_THRESHOLD}\n",
        encoding="utf-8",
    )

    print(f"\nDataset written to: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  train={t1}  val={t2-t1}  test={n-t2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
