"""
Build per-alpha NIF datasets — one network per angle, no shared FiLM competition.

Strategy:
  - train/val: 50/50 balanced (40k occ + 40k empty for train, 5k+5k for val)
    → standard BCE works without pos_weight; Adam normalization not distorted
  - test: FULL VOXEL GRID (all 377,500 voxels, true film distribution)
    → honest precision/recall; not inflated by balanced sampling

Why separate networks?
  The FiLM multi-film experiment showed: FiLM conditioning collapses to
  per-alpha class mean in <10 epochs, killing spatial learning signal.
  Per-alpha networks have no such competition: each network learns one film.

Gate: APPROVE_PER_ALPHA_SEPARATE_TRAINING
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
VOXEL_DIR     = PINN_NIF_ROOT / "pinn_training/datasets/universal_glad_pipeline/voxels"
OUT_BASE      = PINN_NIF_ROOT / "pinn_training/datasets/nif_per_alpha"

BINARY_THR    = 0.5
N_TRAIN_EACH  = 40_000   # occupied + empty → 80k train total
N_VAL_EACH    = 5_000    # occupied + empty → 10k val total
SEED          = 42
ALPHAS        = [65, 70, 75, 80, 85, 87, 89]   # skip 60 (occ=0.18%, degenerate)


def build_one(alpha_deg: int, rng: np.random.Generator, force: bool = False) -> dict:
    vf = next(VOXEL_DIR.glob(f"rho_LA_helical_alpha_300nm_alpha{alpha_deg:03d}_seed000*.npy"))
    out_dir = OUT_BASE / f"alpha_{alpha_deg:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metadata.json").exists() and not force:
        print(f"  α={alpha_deg:3d}: SKIP (already exists)")
        return json.loads((out_dir / "metadata.json").read_text())

    arr = np.load(vf).astype(np.float32)
    nx, ny, nz = arr.shape
    flat_bin = (arr.ravel() > BINARY_THR).astype(np.float32)
    n_total  = flat_bin.size
    occ_frac = float(flat_bin.mean())

    occupied = np.flatnonzero(flat_bin > 0.5)
    empty    = np.flatnonzero(flat_bin < 0.5)
    n_occ    = len(occupied)
    n_empty  = len(empty)

    def make_balanced(n_each: int) -> np.ndarray:
        idx_occ   = rng.choice(occupied, n_each, replace=(n_occ < n_each))
        idx_empty = rng.choice(empty,    n_each, replace=(n_empty < n_each))
        idx = np.concatenate([idx_occ, idx_empty])
        rng.shuffle(idx)
        i, j, k = np.unravel_index(idx, arr.shape)
        x = (i + 0.5) / nx
        y = (j + 0.5) / ny
        z = (k + 0.5) / nz
        alpha_norm = (alpha_deg - 74.5) / 14.5
        label = flat_bin[idx]
        return np.column_stack([x, y, z, np.full(len(idx), alpha_norm), label]).astype(np.float32)

    train_data = make_balanced(N_TRAIN_EACH)
    val_data   = make_balanced(N_VAL_EACH)

    # Full voxel grid — for honest test evaluation on real film distribution
    all_idx  = np.arange(n_total)
    i, j, k  = np.unravel_index(all_idx, arr.shape)
    x_all    = (i + 0.5) / nx
    y_all    = (j + 0.5) / ny
    z_all    = (k + 0.5) / nz
    a_norm   = (alpha_deg - 74.5) / 14.5
    full_data = np.column_stack([x_all, y_all, z_all,
                                 np.full(n_total, a_norm), flat_bin]).astype(np.float32)

    np.savez_compressed(out_dir / "train_samples.npz", data=train_data)
    np.savez_compressed(out_dir / "val_samples.npz",   data=val_data)
    np.savez_compressed(out_dir / "test_full_grid.npz", data=full_data)

    meta = {
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
        "alpha_deg":       alpha_deg,
        "alpha_norm":      float(a_norm),
        "voxel_shape":     list(arr.shape),
        "n_voxels_total":  n_total,
        "occ_frac":        round(occ_frac, 6),
        "n_occupied":      int(n_occ),
        "n_empty":         int(n_empty),
        "pos_weight_ref":  round(n_empty / max(n_occ, 1), 4),
        "train_samples":   len(train_data),
        "val_samples":     len(val_data),
        "test_full_grid":  n_total,
        "sampling":        "50/50 balanced occ/empty (with replacement if needed)",
        "loss_strategy":   "standard BCE on balanced batches (no pos_weight needed)",
        "voxel_source":    str(vf.relative_to(PROJECT_ROOT)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "normalization.json").write_text(json.dumps({
        "x_y_z":              "unit voxel coordinates [0,1]",
        "alpha_norm":         f"({alpha_deg}-74.5)/14.5 = {a_norm:.4f}  (fixed for this network)",
        "rho":                "binary {0,1} thresholded at 0.5",
        "global_occ_frac_train": round(0.5, 4),   # balanced → always 0.5 in train
    }, indent=2) + "\n", encoding="utf-8")

    print(f"  α={alpha_deg:3d}: occ={occ_frac:.4f}  n_occ={n_occ:7,}  pw_ref={n_empty/max(n_occ,1):.1f}"
          f"  train={len(train_data):,}  val={len(val_data):,}  full_grid={n_total:,}")
    return meta


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Building per-alpha NIF datasets → {OUT_BASE.relative_to(PROJECT_ROOT)}")
    print(f"{'α':>4}  {'occ_frac':>9}  {'n_occ':>8}  {'pos_weight':>11}  {'train':>8}  {'val':>6}  {'full_grid':>10}")
    print(f"----  ---------  --------  -----------  --------  ------  ----------")
    for alpha in ALPHAS:
        build_one(alpha, rng, force=args.force)

    print(f"\nDone. {len(ALPHAS)} datasets in {OUT_BASE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
