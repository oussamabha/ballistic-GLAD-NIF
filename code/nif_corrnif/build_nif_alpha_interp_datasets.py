"""
Build leave-one-alpha-out datasets for AlphaInterpolatingFlatNIF (Option C).

For each held-out alpha, creates:
  - train_samples.npz : balanced data from the other 6 alphas
  - val_samples.npz   : balanced data from the other 6 alphas (val split)
  - test_holdout.npz  : all voxels of the held-out alpha (z-holdout AND full grid)

Dataset columns: (x, y, z, alpha_norm, phi_sin, phi_cos, label) — 7 columns.
  - All 6 alphas contribute balanced samples (40k occ + 40k empty each)
  - Combined train: 6 × 80k = 480k samples (shuffled across alphas)
  - Test: full z-block of held-out alpha (377,500 voxels)

The model must predict morphology at an alpha never seen during training.
This tests whether the NIF learned a continuous α-response surface.

Gate: APPROVE_ALPHA_INTERPOLATING_NIF_TRAINING
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
OUT_BASE      = PINN_NIF_ROOT / "pinn_training/datasets/nif_alpha_interp"

HELIX_PITCH_NM   = 150.0
FILM_THICKNESS_NM = 300.0
Z_TRAIN_FRAC = 0.80          # train on z[0:k_split], holdout z[k_split:]
N_TRAIN_EACH = 40_000        # occ + empty per alpha → 80k per alpha → 480k total for 6 alphas
N_VAL_EACH   = 5_000         # occ + empty per alpha → 10k val per alpha
BINARY_THR   = 0.5
SEED         = 42
ALPHAS       = [65, 70, 75, 80, 85, 87, 89]


def compute_phase_features(k_indices: np.ndarray, nz: int) -> tuple[np.ndarray, np.ndarray]:
    pitch_voxels = HELIX_PITCH_NM / (FILM_THICKNESS_NM / nz)
    phi = 2.0 * math.pi * k_indices / pitch_voxels
    return np.sin(phi).astype(np.float32), np.cos(phi).astype(np.float32)


def load_voxel_array(alpha_deg: int) -> np.ndarray:
    vf = next(VOXEL_DIR.glob(f"rho_LA_helical_alpha_300nm_alpha{alpha_deg:03d}_seed000*.npy"))
    return np.load(vf).astype(np.float32)


def build_alpha_samples(alpha_deg: int, arr: np.ndarray, rng: np.random.Generator,
                        n_each: int, z_slice_range: range | None = None) -> np.ndarray:
    """Build balanced samples for one alpha. Optionally restrict to z-slice range."""
    nx, ny, nz = arr.shape
    a_norm = (alpha_deg - 74.5) / 14.5

    if z_slice_range is not None:
        sub = arr[:, :, z_slice_range.start:z_slice_range.stop]
        z_offset = z_slice_range.start
    else:
        sub = arr
        z_offset = 0
    bx, by, bz = sub.shape
    flat_bin = (sub.ravel() > BINARY_THR).astype(np.float32)

    occupied = np.flatnonzero(flat_bin > 0.5)
    empty    = np.flatnonzero(flat_bin < 0.5)

    idx_occ   = rng.choice(occupied, n_each, replace=(len(occupied) < n_each))
    idx_empty = rng.choice(empty,    n_each, replace=(len(empty)    < n_each))
    idx = np.concatenate([idx_occ, idx_empty])
    rng.shuffle(idx)

    i, j, k = np.unravel_index(idx, (bx, by, bz))
    k_global = k + z_offset
    phi_sin, phi_cos = compute_phase_features(k_global.astype(np.float32), nz)

    return np.column_stack([
        (i + 0.5) / nx,
        (j + 0.5) / ny,
        (k_global + 0.5) / nz,
        np.full(len(idx), a_norm),
        phi_sin, phi_cos,
        flat_bin[idx],
    ]).astype(np.float32)


def build_full_block(alpha_deg: int, arr: np.ndarray, z_offset: int, block: np.ndarray) -> np.ndarray:
    """Build full voxel grid for one z-block (test/holdout)."""
    nx, ny, nz = arr.shape
    a_norm = (alpha_deg - 74.5) / 14.5
    flat_bin = (block.ravel() > BINARY_THR).astype(np.float32)
    idx = np.arange(flat_bin.size)
    i, j, k = np.unravel_index(idx, block.shape)
    k_global = k + z_offset
    phi_sin, phi_cos = compute_phase_features(k_global.astype(np.float32), nz)
    return np.column_stack([
        (i + 0.5) / nx, (j + 0.5) / ny, (k_global + 0.5) / nz,
        np.full(flat_bin.size, a_norm),
        phi_sin, phi_cos, flat_bin,
    ]).astype(np.float32)


def build_looa(held_out: int, rng: np.random.Generator, force: bool = False) -> dict:
    """Build leave-one-alpha-out dataset. Held-out alpha = test."""
    out_dir = OUT_BASE / f"looa_holdout_{held_out:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ((out_dir / "metadata.json").exists()
            and (out_dir / "test_train_region.npz").exists()
            and not force):
        print(f"  holdout α={held_out}: SKIP")
        return json.loads((out_dir / "metadata.json").read_text())

    train_alphas = [a for a in ALPHAS if a != held_out]
    train_parts, val_parts = [], []

    # Reference nz from first alpha (all have same grid)
    arr0 = load_voxel_array(train_alphas[0])
    nz = arr0.shape[2]
    k_split = round(nz * Z_TRAIN_FRAC)   # same split as phase-conditioned experiment

    for a in train_alphas:
        arr = load_voxel_array(a)
        z_range = range(0, k_split)
        train_parts.append(build_alpha_samples(a, arr, rng, N_TRAIN_EACH, z_range))
        val_parts.append(build_alpha_samples(a, arr, rng, N_VAL_EACH, z_range))

    train_data = np.concatenate(train_parts)
    val_data   = np.concatenate(val_parts)
    # Shuffle across alphas
    rng.shuffle(train_data)
    rng.shuffle(val_data)

    # Held-out alpha: full z-holdout (spatial block never seen)
    arr_ho        = load_voxel_array(held_out)
    holdout_block = arr_ho[:, :, k_split:]
    test_data     = build_full_block(held_out, arr_ho, k_split, holdout_block)

    # Training z-region of held-out alpha — PURE alpha-interpolation test.
    # z[0:k_split] is in the training z-range; z-generalization is NOT in play.
    # Only challenge: predicting morphology at an alpha never seen during training.
    train_region_block = arr_ho[:, :, :k_split]
    train_region_data  = build_full_block(held_out, arr_ho, 0, train_region_block)
    np.savez_compressed(out_dir / "test_train_region.npz", data=train_region_data)
    occ_tr = float(train_region_data[:, -1].mean())

    # Also save full grid of held-out alpha for comparison
    full_block = arr_ho
    full_flat  = (full_block.ravel() > BINARY_THR).astype(np.float32)
    nx, ny, nz_ho = arr_ho.shape
    idx_f = np.arange(full_flat.size)
    i_f, j_f, k_f = np.unravel_index(idx_f, arr_ho.shape)
    a_norm_ho = (held_out - 74.5) / 14.5
    phi_s, phi_c = compute_phase_features(k_f.astype(np.float32), nz_ho)
    full_data = np.column_stack([
        (i_f+0.5)/nx, (j_f+0.5)/ny, (k_f+0.5)/nz_ho,
        np.full(full_flat.size, a_norm_ho), phi_s, phi_c, full_flat,
    ]).astype(np.float32)

    np.savez_compressed(out_dir / "train_samples.npz",  data=train_data)
    np.savez_compressed(out_dir / "val_samples.npz",    data=val_data)
    np.savez_compressed(out_dir / "test_holdout.npz",   data=test_data)
    np.savez_compressed(out_dir / "test_full_grid.npz", data=full_data)

    occ_ho = float(test_data[:, -1].mean())
    meta = {
        "generated_at":              datetime.now().isoformat(timespec="seconds"),
        "held_out_alpha":            held_out,
        "train_alphas":              train_alphas,
        "k_split":                   k_split,
        "n_train_each":              N_TRAIN_EACH,
        "n_val_each":                N_VAL_EACH,
        "train_samples":             len(train_data),
        "val_samples":               len(val_data),
        "test_train_region_samples": len(train_region_data),
        "test_holdout_samples":      len(test_data),
        "test_full_grid":            int(full_flat.size),
        "occ_frac_train_region":     round(occ_tr, 6),
        "occ_frac_holdout":          round(occ_ho, 6),
        "trivial_iou_train_region":  round(occ_tr, 6),
        "trivial_iou_holdout":       round(occ_ho, 6),
        "n_input_features":          6,
        "loss_strategy":             "bce (balanced 50/50 per alpha across train alphas)",
        "gate":                      "APPROVE_ALPHA_INTERPOLATING_NIF_TRAINING",
        "_eval_note": (
            "Primary eval: test_train_region.npz (z[0:k_split] of held-out alpha). "
            "Tests PURE alpha-interpolation — z-generalization not in play. "
            "Secondary: test_holdout.npz (z[k_split:] of held-out alpha) — conflates "
            "alpha+z generalization; expected to fail per corotating frame diagnostic."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "normalization.json").write_text(json.dumps({
        "feature_columns":       ["x", "y", "z", "alpha_norm", "phi_sin", "phi_cos"],
        "alpha_norm":            "(alpha_deg - 74.5) / 14.5",
        "phi_sin_phi_cos":       f"sin/cos(2π×k/{HELIX_PITCH_NM/(FILM_THICKNESS_NM/nz):.1f})",
        "global_occ_frac_train": 0.5,   # balanced per alpha → global ≈ 0.5
    }, indent=2) + "\n", encoding="utf-8")

    print(f"  holdout α={held_out:3d}: train_alphas={train_alphas}  "
          f"train={len(train_data):,}  val={len(val_data):,}  "
          f"train_region={len(train_region_data):,}(occ={occ_tr:.4f})  "
          f"z_holdout={len(test_data):,}(occ={occ_ho:.4f})")
    return meta


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--held-out", type=int, default=None,
                   help="Build only one LOOA split (default: all 7)")
    args = p.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    targets = [args.held_out] if args.held_out else ALPHAS
    print(f"Building leave-one-alpha-out datasets → {OUT_BASE.relative_to(PROJECT_ROOT)}")
    print(f"  n_train_each={N_TRAIN_EACH}/alpha  k_split=z[0:~{round(151*Z_TRAIN_FRAC)}]")
    print()
    for ho in targets:
        build_looa(ho, rng, force=args.force)
    print(f"\nDone. {len(targets)} LOOA splits in {OUT_BASE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
