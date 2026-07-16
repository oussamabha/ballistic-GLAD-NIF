"""
Build full-grid spatial-block datasets for dense alpha films.

Dense alphas (occ_frac > 0.15) failed with 80k balanced sampling:
the model cannot find column boundaries from partial random samples
when the film is densely packed. Pred_spread ≈ 0.10 confirms the model
outputs nearly uniform probability (no spatial discrimination).

Root cause: 80k random samples = 17-28% of each voxel class. Insufficient
spatial context to locate column walls when most of the film is occupied.

Solution: train on ALL voxels in z = 0..0.8*nz (spatial training block),
test on z = 0.8*nz..nz (spatial holdout — never seen during training).
Use weighted BCE (pos_weight = n_empty_train/n_occ_train) to balance
gradient contributions without subsampling.

Overfit test reference: IoU=0.737 for α=65 with all 377,500 voxels in training.
This experiment uses 80% of the z-extent as training (302,500 voxels).

Gate: APPROVE_FULLGRID_DENSE_ALPHA_TRAINING
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
OUT_BASE      = PINN_NIF_ROOT / "pinn_training/datasets/nif_fullgrid_dense"

Z_TRAIN_FRAC = 0.80   # first 80% of z-slices → training
VAL_FRAC     = 0.10   # 10% of training voxels → val (monitoring only, same spatial region)
BINARY_THR   = 0.5
SEED         = 42
ALPHAS       = [65, 70, 75, 80]  # dense alphas: occ_frac > 0.15, trivial in per-alpha balanced


def build_one(alpha_deg: int, rng: np.random.Generator, force: bool = False) -> dict:
    vf = next(VOXEL_DIR.glob(f"rho_LA_helical_alpha_300nm_alpha{alpha_deg:03d}_seed000*.npy"))
    out_dir = OUT_BASE / f"alpha_{alpha_deg:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metadata.json").exists() and not force:
        print(f"  α={alpha_deg:3d}: SKIP (already exists, use --force to rebuild)")
        return json.loads((out_dir / "metadata.json").read_text())

    arr = np.load(vf).astype(np.float32)
    nx, ny, nz = arr.shape

    # Spatial block split on z-axis (deposition direction)
    # Helical pitch ≈ 150nm, film = 300nm → ~2 full helix turns in 151 z-slices
    # Both train and test blocks see parts of the helix → representative
    k_split = round(nz * Z_TRAIN_FRAC)  # 121 for nz=151

    arr_train = arr[:, :, :k_split]       # shape (nx, ny, k_split)
    arr_test  = arr[:, :, k_split:]       # shape (nx, ny, nz-k_split)

    a_norm = (alpha_deg - 74.5) / 14.5

    def block_to_data(block: np.ndarray, z_offset: int) -> np.ndarray:
        bx, by, bz = block.shape
        flat_bin = (block.ravel() > BINARY_THR).astype(np.float32)
        idx = np.arange(flat_bin.size)
        i, j, k = np.unravel_index(idx, (bx, by, bz))
        x_n = (i + 0.5) / nx
        y_n = (j + 0.5) / ny
        z_n = (k + z_offset + 0.5) / nz   # global z coordinate
        return np.column_stack([x_n, y_n, z_n, np.full(flat_bin.size, a_norm), flat_bin]).astype(np.float32)

    train_full = block_to_data(arr_train, z_offset=0)
    test_data  = block_to_data(arr_test,  z_offset=k_split)

    # Val: random 10% subset from training region (monitoring only — no spatial holdout)
    n_val = max(1, round(len(train_full) * VAL_FRAC))
    val_idx  = rng.choice(len(train_full), n_val, replace=False)
    val_data = train_full[val_idx]
    # train_data = all train voxels (optimizer sees everything in training region)
    train_data = train_full

    # Compute pos_weight from training region (what weighted_bce sees)
    y_train       = train_data[:, 4]
    n_occ_train   = int((y_train > 0.5).sum())
    n_empty_train = int((y_train < 0.5).sum())
    occ_frac_train = float(y_train.mean())
    pos_weight     = n_empty_train / max(n_occ_train, 1)

    # Full voxel grid — for comparison with per-alpha results
    flat_full  = (arr.ravel() > BINARY_THR).astype(np.float32)
    idx_full   = np.arange(flat_full.size)
    i_f, j_f, k_f = np.unravel_index(idx_full, arr.shape)
    full_data  = np.column_stack([
        (i_f + 0.5) / nx, (j_f + 0.5) / ny, (k_f + 0.5) / nz,
        np.full(flat_full.size, a_norm), flat_full
    ]).astype(np.float32)

    np.savez_compressed(out_dir / "train_samples.npz",  data=train_data)
    np.savez_compressed(out_dir / "val_samples.npz",    data=val_data)
    np.savez_compressed(out_dir / "test_holdout.npz",   data=test_data)
    np.savez_compressed(out_dir / "test_full_grid.npz", data=full_data)

    meta = {
        "generated_at":        datetime.now().isoformat(timespec="seconds"),
        "alpha_deg":           alpha_deg,
        "alpha_norm":          round(a_norm, 6),
        "voxel_shape":         list(arr.shape),
        "n_voxels_total":      int(flat_full.size),
        "occ_frac_full":       round(float(flat_full.mean()), 6),
        "split_strategy":      f"z_block: train=z[0:{k_split}], test=z[{k_split}:{nz}]",
        "z_train_slices":      k_split,
        "z_test_slices":       int(nz - k_split),
        "k_split":             k_split,
        "train_samples":       len(train_data),
        "val_samples":         len(val_data),
        "test_holdout_samples": len(test_data),
        "test_full_grid":      int(flat_full.size),
        "occ_frac_train":      round(occ_frac_train, 6),
        "n_occ_train":         n_occ_train,
        "n_empty_train":       n_empty_train,
        "pos_weight":          round(pos_weight, 4),
        "loss_strategy":       "weighted_bce: pos_weight = n_empty_train / n_occ_train",
        "voxel_source":        str(vf.relative_to(PROJECT_ROOT)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "normalization.json").write_text(json.dumps({
        "x_y_z":                  "unit voxel coordinates [0,1] normalized to global grid",
        "alpha_norm":             f"({alpha_deg}-74.5)/14.5 = {a_norm:.4f}  (fixed for this network)",
        "rho":                    "binary {0,1} thresholded at 0.5",
        "global_occ_frac_train":  round(occ_frac_train, 6),  # used by build_loss_fn → weighted_bce
        "pos_weight":             round(pos_weight, 4),
        "note": (
            f"pos_weight = n_empty_train / n_occ_train = {n_empty_train}/{n_occ_train} = {pos_weight:.4f}. "
            f"For dense films (occ>0.5), this DOWN-weights occupied class, emphasizing empty voxels "
            f"(column boundaries). Equivalent to upsampling the minority class."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    # Test block occupancy (may differ from global if cut mid-helix-turn)
    occ_test = float(test_data[:, 4].mean())
    print(f"  α={alpha_deg:3d}: occ_train={occ_frac_train:.4f}  occ_test={occ_test:.4f}  "
          f"pw={pos_weight:.3f}  train={len(train_data):,}  val={len(val_data):,}  "
          f"test_holdout={len(test_data):,}")
    return meta


def main():
    import argparse
    p = argparse.ArgumentParser(description="Build full-grid z-block datasets for dense alphas")
    p.add_argument("--force", action="store_true", help="Rebuild even if dataset exists")
    args = p.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Building full-grid dense NIF datasets → {OUT_BASE.relative_to(PROJECT_ROOT)}")
    print(f"  z_train_frac={Z_TRAIN_FRAC} (first 80% of z-slices)")
    print(f"  val_frac={VAL_FRAC} of train region")
    print()
    for alpha in ALPHAS:
        build_one(alpha, rng, force=args.force)
    print(f"\nDone. {len(ALPHAS)} datasets in {OUT_BASE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
