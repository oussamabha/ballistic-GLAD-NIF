"""
Build phase-conditioned NIF datasets — all 7 alpha angles, z-block spatial split.

Adds helix phase encoding (φ_sin, φ_cos) to every voxel coordinate.
This gives the model explicit knowledge of the helix rotation angle,
allowing it to generalize across z-positions never seen in training.

Dataset columns: (x, y, z, alpha_norm, phi_sin, phi_cos, label) — 7 columns total.
  load_npz reads arr[:, :-1] as X (6 features) and arr[:, -1] as y.

Split: z[0:121] train (80%), z[121:150] spatial holdout (20%).
Loss: weighted_bce with pos_weight = n_empty_train / n_occ_train.

Helix parameters (from filename: pitch150, 300nm film, nz=151):
  pitch_voxels = 150nm / (300nm/151) = 75.5 voxels per turn
  φ(k) = 2π × k / 75.5  for k = 0..150

Gate: APPROVE_PHASE_CONDITIONED_NIF_TRAINING
"""
from __future__ import annotations
import json
import math
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
OUT_BASE      = PINN_NIF_ROOT / "pinn_training/datasets/nif_phase_conditioned"

HELIX_PITCH_NM   = 150.0
FILM_THICKNESS_NM = 300.0
Z_TRAIN_FRAC = 0.80
VAL_FRAC     = 0.10
BINARY_THR   = 0.5
SEED         = 42
ALPHAS       = [65, 70, 75, 80, 85, 87, 89]


def compute_phase_features(k_indices: np.ndarray, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute sin/cos helix phase for z-slice indices k."""
    pitch_voxels = HELIX_PITCH_NM / (FILM_THICKNESS_NM / nz)
    phi = 2.0 * math.pi * k_indices / pitch_voxels
    return np.sin(phi).astype(np.float32), np.cos(phi).astype(np.float32)


def build_one(alpha_deg: int, rng: np.random.Generator, force: bool = False) -> dict:
    vf = next(VOXEL_DIR.glob(f"rho_LA_helical_alpha_300nm_alpha{alpha_deg:03d}_seed000*.npy"))
    out_dir = OUT_BASE / f"alpha_{alpha_deg:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "metadata.json").exists() and not force:
        print(f"  α={alpha_deg:3d}: SKIP (already exists)")
        return json.loads((out_dir / "metadata.json").read_text())

    arr = np.load(vf).astype(np.float32)
    nx, ny, nz = arr.shape
    pitch_voxels = HELIX_PITCH_NM / (FILM_THICKNESS_NM / nz)
    k_split = round(nz * Z_TRAIN_FRAC)          # 121 for nz=151
    a_norm  = (alpha_deg - 74.5) / 14.5

    arr_train = arr[:, :, :k_split]
    arr_test  = arr[:, :, k_split:]

    def block_to_data(block: np.ndarray, z_offset: int) -> np.ndarray:
        bx, by, bz = block.shape
        flat_bin = (block.ravel() > BINARY_THR).astype(np.float32)
        idx = np.arange(flat_bin.size)
        i, j, k = np.unravel_index(idx, (bx, by, bz))
        k_global = k + z_offset                  # global z-slice index
        phi_sin, phi_cos = compute_phase_features(k_global.astype(np.float32), nz)
        return np.column_stack([
            (i + 0.5) / nx,                      # x ∈ (0,1)
            (j + 0.5) / ny,                      # y ∈ (0,1)
            (k + z_offset + 0.5) / nz,           # z ∈ (0,1)  global
            np.full(flat_bin.size, a_norm),       # α_norm (fixed per network)
            phi_sin,                              # sin(2π k_global / pitch_voxels)
            phi_cos,                              # cos(2π k_global / pitch_voxels)
            flat_bin,                             # label (last column)
        ]).astype(np.float32)

    train_full = block_to_data(arr_train, z_offset=0)
    test_data  = block_to_data(arr_test,  z_offset=k_split)

    # Val: 10% random subset of training region (monitoring; same z-block)
    n_val    = max(1, round(len(train_full) * VAL_FRAC))
    val_idx  = rng.choice(len(train_full), n_val, replace=False)
    val_data = train_full[val_idx]
    train_data = train_full          # all training-region voxels for optimizer

    # Pos_weight from training distribution
    y_train       = train_data[:, -1]
    n_occ_train   = int((y_train > 0.5).sum())
    n_empty_train = int((y_train < 0.5).sum())
    occ_frac_train = float(y_train.mean())
    pos_weight     = n_empty_train / max(n_occ_train, 1)

    # Full voxel grid for cross-experiment comparison
    flat_full = (arr.ravel() > BINARY_THR).astype(np.float32)
    idx_full  = np.arange(flat_full.size)
    i_f, j_f, k_f = np.unravel_index(idx_full, arr.shape)
    phi_sin_f, phi_cos_f = compute_phase_features(k_f.astype(np.float32), nz)
    full_data = np.column_stack([
        (i_f + 0.5) / nx, (j_f + 0.5) / ny, (k_f + 0.5) / nz,
        np.full(flat_full.size, a_norm),
        phi_sin_f, phi_cos_f,
        flat_full,
    ]).astype(np.float32)

    np.savez_compressed(out_dir / "train_samples.npz",  data=train_data)
    np.savez_compressed(out_dir / "val_samples.npz",    data=val_data)
    np.savez_compressed(out_dir / "test_holdout.npz",   data=test_data)
    np.savez_compressed(out_dir / "test_full_grid.npz", data=full_data)

    meta = {
        "generated_at":         datetime.now().isoformat(timespec="seconds"),
        "alpha_deg":            alpha_deg,
        "alpha_norm":           round(a_norm, 6),
        "voxel_shape":          list(arr.shape),
        "n_voxels_total":       int(flat_full.size),
        "occ_frac_full":        round(float(flat_full.mean()), 6),
        "occ_frac_train":       round(occ_frac_train, 6),
        "occ_frac_test":        round(float(test_data[:, -1].mean()), 6),
        "pitch_voxels":         round(pitch_voxels, 4),
        "helix_pitch_nm":       HELIX_PITCH_NM,
        "split_strategy":       f"z_block: train=z[0:{k_split}], holdout=z[{k_split}:{nz}]",
        "k_split":              k_split,
        "z_train_slices":       k_split,
        "z_test_slices":        int(nz - k_split),
        "n_input_features":     6,
        "feature_columns":      ["x", "y", "z", "alpha_norm", "phi_sin", "phi_cos"],
        "train_samples":        len(train_data),
        "val_samples":          len(val_data),
        "test_holdout_samples": len(test_data),
        "test_full_grid":       int(flat_full.size),
        "n_occ_train":          n_occ_train,
        "n_empty_train":        n_empty_train,
        "pos_weight":           round(pos_weight, 4),
        "loss_strategy":        "weighted_bce: pos_weight = n_empty_train / n_occ_train",
        "voxel_source":         str(vf.relative_to(PROJECT_ROOT)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "normalization.json").write_text(json.dumps({
        "feature_columns":       ["x", "y", "z", "alpha_norm", "phi_sin", "phi_cos"],
        "x_y_z":                 "unit voxel coordinates [0,1] in global grid",
        "alpha_norm":            f"({alpha_deg}-74.5)/14.5 = {a_norm:.4f}  (fixed per network)",
        "phi_sin_phi_cos":       f"sin/cos(2π×k/{pitch_voxels:.1f})  k=z-slice-index",
        "global_occ_frac_train": round(occ_frac_train, 6),
        "pos_weight":            round(pos_weight, 4),
    }, indent=2) + "\n", encoding="utf-8")

    occ_test = float(test_data[:, -1].mean())
    print(f"  α={alpha_deg:3d}: occ_train={occ_frac_train:.4f}  occ_holdout={occ_test:.4f}  "
          f"pw={pos_weight:.3f}  train={len(train_data):,}  val={len(val_data):,}  "
          f"holdout={len(test_data):,}")
    return meta


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Building phase-conditioned NIF datasets → {OUT_BASE.relative_to(PROJECT_ROOT)}")
    print(f"  pitch_voxels = {HELIX_PITCH_NM/(FILM_THICKNESS_NM/151):.1f}  "
          f"z_train_frac={Z_TRAIN_FRAC}  val_frac={VAL_FRAC}")
    print(f"  n_features = 6: (x, y, z, α_norm, φ_sin, φ_cos)")
    print()
    for alpha in ALPHAS:
        build_one(alpha, rng, force=args.force)
    print(f"\nDone. {len(ALPHAS)} datasets in {OUT_BASE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
