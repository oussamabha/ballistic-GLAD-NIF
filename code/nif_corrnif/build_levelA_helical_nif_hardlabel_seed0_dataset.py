"""
Build Level A helical NIF dataset with HARD binary labels, seed-0 ONLY.

Motivation: the multi-seed hard-label dataset (seed0+seed1+seed2 pooled for α=80,85)
creates contradictory binary labels at the same (x,y,z) coordinates across seeds.
The same voxel position may be occupied in seed0 and empty in seed1, giving the
network two contradictory signals → converges to per-alpha class mean, not spatial boundaries.

This builder uses ONLY seed 0 for all alpha angles, eliminating label conflicts.
Confirmed representable by single-film overfit test 2026-06-15:
  α=65 seed0: IoU 0.737 vs trivial 0.625 (Δ=+0.112) at 200 epochs.

Usage:
  python build_levelA_helical_nif_hardlabel_seed0_dataset.py [--force]
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
OUT_DIR    = PINN_NIF_ROOT / "pinn_training" / "datasets" / "levelA_helical_nif_hardlabel_seed0"

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

    return np.column_stack(
        [x, y, z, np.full_like(x, alpha_norm), rho_bin]
    ).astype(np.float32)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build seed-0-only hard-label helical NIF dataset.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT_DIR / "metadata.json").exists() and not args.force:
        print("SKIP: dataset already exists. Use --force to rebuild.")
        return 0

    rng = np.random.default_rng(SEED)

    # Only seed 0 — filters out seed001, seed002 pooling for α=80,85
    files = sorted(f for f in VOXEL_DIR.glob("rho_LA_helical_alpha_300nm_*.npy")
                   if "seed000" in f.stem)
    if not files:
        raise SystemExit(f"No seed000 voxel files found in {VOXEL_DIR}")

    print(f"Found {len(files)} seed-0 voxel files:")
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

    global_occ = float(data[:t1, 4].mean())

    (OUT_DIR / "normalization.json").write_text(json.dumps({
        "x_y_z":      "unit voxel coordinates [0,1]",
        "alpha_norm":  "(alpha_deg-74.5)/14.5",
        "rho":         "binary {0,1} — thresholded at 0.5 from max-normalized bead-count histogram",
        "threshold":   BINARY_THRESHOLD,
        "global_occ_frac_train": round(global_occ, 6),
        "recommended_focal_alpha": round(1 - global_occ, 4),
    }, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "label_scheme":     "hard_binary_seed0_only",
        "binary_threshold": BINARY_THRESHOLD,
        "seed_filter":      "seed000 only — eliminates multi-seed label conflicts at α=80,85",
        "source_voxels":    meta_rows,
        "morphology_mode":  "helical",
        "pitch_nm":         150,
        "samples_per_file": SAMPLES_PER_FILE,
        "seed":             SEED,
        "n_files":          len(files),
        "overfit_test_result": {
            "alpha": 65, "seed": 0,
            "model_iou": 0.7372, "trivial_iou": 0.6253, "delta": 0.1119,
            "verdict": "MEMORIZABLE — structure representable by FiLMConditionedNIF",
            "date": "2026-06-15",
        },
    }, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "split_report.md").write_text(
        f"# Level A Hard-Label Seed-0 Split Report\n\n"
        f"train={t1}, val={t2-t1}, test={n-t2}\n"
        f"label_scheme=hard_binary_seed0_only, threshold={BINARY_THRESHOLD}\n"
        f"global_occ_frac_train={global_occ:.4f}\n"
        f"n_files={len(files)}\n",
        encoding="utf-8",
    )

    print(f"\nDataset written to: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  train={t1}  val={t2-t1}  test={n-t2}")
    print(f"  global occ_frac (train): {global_occ:.4f}")
    print(f"  recommended focal_alpha: {1-global_occ:.4f} (positive class weight)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
