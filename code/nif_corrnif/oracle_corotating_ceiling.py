"""
Corotating-frame oracle: ceiling IoU achievable by any (x_rot, y_rot) → occupancy model.

For each alpha:
  1. Derotate z[0:121] by -φ(k) → accumulate mean occupancy map M_train(x_rot, y_rot)
  2. Use M_train as predictor on holdout z[121:150] (derotated the same way)
  3. Threshold-sweep → best IoU

This is an UPPER BOUND. No learned NIF can exceed this. If oracle Δ < 0.02:
the corotating-frame approach is not worth building for that alpha.

Also evaluates on the training region itself (oracle in-region ceiling).
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

HELIX_PITCH_NM    = 150.0
FILM_THICKNESS_NM = 300.0
BINARY_THR        = 0.5
ALPHAS            = [65, 70, 75, 80, 85, 87, 89]
THRESHOLDS        = np.concatenate([np.arange(0.02, 0.50, 0.02), np.arange(0.50, 0.96, 0.05)])

def find_project_root(start=Path(__file__).resolve()):
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("No project root")

PROJECT_ROOT = find_project_root()
VOXEL_DIR    = PROJECT_ROOT / "02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline/voxels"


def phi(k: np.ndarray, nz: int) -> np.ndarray:
    pitch_voxels = HELIX_PITCH_NM / (FILM_THICKNESS_NM / nz)
    return 2.0 * math.pi * k / pitch_voxels


def derotate_slice(sl: np.ndarray, angle: float) -> np.ndarray:
    """Derotate a 2D slice (nx, ny) by -angle using nearest-neighbour lookup."""
    nx, ny = sl.shape
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    # destination grid (i,j) → source grid (i_src, j_src)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    i_src = cos_a * (ii - cx) - sin_a * (jj - cy) + cx
    j_src = sin_a * (ii - cx) + cos_a * (jj - cy) + cy
    i_src = np.clip(np.round(i_src).astype(int), 0, nx - 1)
    j_src = np.clip(np.round(j_src).astype(int), 0, ny - 1)
    return sl[i_src, j_src]


def best_iou(pred_map_flat: np.ndarray, y_flat: np.ndarray) -> dict:
    best = {"iou": -1, "thr": 0.5, "delta": 0.0}
    occ = float(y_flat.mean())
    for thr in THRESHOLDS:
        pb = pred_map_flat >= thr
        tp = float((pb & y_flat).sum())
        fp = float((pb & ~y_flat).sum())
        fn = float((~pb & y_flat).sum())
        d  = tp + fp + fn
        iou = tp / d if d > 0 else 0.0
        if iou > best["iou"]:
            best = {"iou": iou, "thr": float(thr), "delta": iou - occ}
    return best


print("=" * 90)
print(" COROTATING-FRAME ORACLE — ceiling IoU for any (x_rot,y_rot)→occupancy model")
print(" M_train = mean derotated occupancy map from z[0:121]")
print(" Test: nearest-neighbour lookup on z[121:150] and z[0:121]")
print("=" * 90)
print()
print(f" {'α':>4}  {'occ_train':>10}  {'occ_hold':>9}  "
      f"{'in-region Δ':>12}  {'holdout Δ':>10}  {'peak/mean':>10}  {'verdict'}")
print(" " + "-" * 80)

for alpha in ALPHAS:
    vf   = next(VOXEL_DIR.glob(f"rho_LA_helical_alpha_300nm_alpha{alpha:03d}_seed000*.npy"))
    arr  = np.load(vf).astype(np.float32)
    binary = (arr > BINARY_THR)           # (nx, ny, nz) bool
    nx, ny, nz = binary.shape
    k_split    = round(nz * 0.80)         # 121 for nz=151

    # --- Build M_train: mean derotated occupancy across train z-slices ---
    M_sum  = np.zeros((nx, ny), dtype=np.float64)
    angles = phi(np.arange(k_split), nz)
    for k in range(k_split):
        sl = binary[:, :, k].astype(np.float64)
        M_sum += derotate_slice(sl, angles[k])
    M_train = (M_sum / k_split).astype(np.float32)   # (nx, ny) ∈ [0,1]

    peak_mean = float(M_train.max() / (M_train.mean() + 1e-9))

    # --- In-region oracle: for each voxel (ii,jj,k), find its corotating position ---
    # Corotating position = derotation by -φ(k), same convention as derotate_slice.
    # derotate_slice uses cos(-φ), sin(-φ) → i_src = cos(φ)*(i-cx)+sin(φ)*(j-cy)+cx
    train_preds, train_labels = [], []
    for k in range(k_split):
        angle = angles[k]
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)   # rotation by -φ → same as derotate_slice
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        i_src  = np.clip(np.round(cos_a*(ii-cx) - sin_a*(jj-cy) + cx).astype(int), 0, nx-1)
        j_src  = np.clip(np.round(sin_a*(ii-cx) + cos_a*(jj-cy) + cy).astype(int), 0, ny-1)
        train_preds.append(M_train[i_src, j_src].ravel())
        train_labels.append(binary[:, :, k].ravel())
    in_preds  = np.concatenate(train_preds)
    in_labels = np.concatenate(train_labels)
    in_best   = best_iou(in_preds, in_labels)

    # --- Holdout oracle ---
    hold_preds, hold_labels = [], []
    ho_angles = phi(np.arange(k_split, nz), nz)
    for idx, k in enumerate(range(k_split, nz)):
        angle  = ho_angles[idx]
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)   # rotation by -φ, same as derotate_slice
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cx, cy = (nx-1)/2.0, (ny-1)/2.0
        i_src  = np.clip(np.round(cos_a*(ii-cx) - sin_a*(jj-cy) + cx).astype(int), 0, nx-1)
        j_src  = np.clip(np.round(sin_a*(ii-cx) + cos_a*(jj-cy) + cy).astype(int), 0, ny-1)
        hold_preds.append(M_train[i_src, j_src].ravel())
        hold_labels.append(binary[:, :, k].ravel())
    ho_preds  = np.concatenate(hold_preds)
    ho_labels = np.concatenate(hold_labels)
    ho_best   = best_iou(ho_preds, ho_labels)

    occ_tr = float(in_labels.mean())
    occ_ho = float(ho_labels.mean())
    verdict = ("HEADROOM-TRAIN" if in_best["delta"] > 0.02 else "trivial-train") + \
              (" + HOLDOUT" if ho_best["delta"] > 0.02 else "")

    print(f" {alpha:>4}  {occ_tr:>10.4f}  {occ_ho:>9.4f}  "
          f"{in_best['delta']:>+12.4f}  {ho_best['delta']:>+10.4f}  "
          f"{peak_mean:>10.3f}  {verdict}")

print()
print(" Interpretation:")
print("  in-region Δ > 0.02  → spatial structure exists in 2D corotating map")
print("  holdout Δ   > 0.02  → structure consistent enough for z-block generalization")
print("  Δ ≤ 0.02 for both  → don't build CorotatingFrameNIF for that alpha")
print("=" * 90)
