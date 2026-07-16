"""
Spatial holdout evaluation for dense-alpha full-grid NIF networks.

Evaluates each best.pt on:
  1. test_holdout.npz  — z-block spatial holdout (z >= 80% of nz, never seen in training)
  2. test_full_grid.npz — full voxel grid (for comparison with per-alpha balanced results)

Trivial baseline:
  On holdout: predict-all-1 → IoU ≈ occ_frac_test
  On full grid: predict-all-1 → IoU = occ_frac_full
  A model is non-trivial if IoU > trivial_baseline at the best threshold.

Reference for comparison:
  Per-alpha balanced (80k): ALL dense alphas were trivial (pred_spread ≈ 0.09-0.20)
  Overfit test (α=65, all voxels): IoU=0.737
  This experiment targets: IoU > 0.65 for dense alphas on spatial holdout.
"""
from __future__ import annotations
import importlib.util, json
import numpy as np
import torch
from pathlib import Path


def find_project_root(start=Path(__file__).resolve()):
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("No project root")


PROJECT_ROOT  = find_project_root()
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
MODEL_PATH    = PINN_NIF_ROOT / "pinn_training/models/nif_levelA_helical.py"
RUNS_DIR      = PINN_NIF_ROOT / "pinn_training/runs"
DS_BASE       = PINN_NIF_ROOT / "pinn_training/datasets/nif_fullgrid_dense"
ALPHAS        = [65, 70, 75, 80]
BATCH         = 65536
THRESHOLDS    = np.concatenate([np.arange(0.02, 0.50, 0.02), np.arange(0.50, 0.96, 0.05)])

spec = importlib.util.spec_from_file_location("nif", MODEL_PATH)
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best(alpha: int):
    candidates = sorted(RUNS_DIR.glob(f"nif_fullgrid_dense_{alpha:03d}_*"), reverse=True)
    if not candidates:
        return None, None
    bp = candidates[0] / "checkpoints/best.pt"
    if not bp.exists():
        return None, None
    ckpt  = torch.load(bp, map_location="cpu", weights_only=False)
    model = mod.build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def predict(model, data: np.ndarray) -> np.ndarray:
    X = torch.tensor(data[:, :4])
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), BATCH):
            preds.append(model(X[s:s+BATCH].to(device)).cpu().numpy())
    return np.concatenate(preds)


def threshold_sweep(pred: np.ndarray, y: np.ndarray):
    best = {"iou": -1, "thr": 0.5, "prec": 0, "rec": 0, "f1": 0}
    for thr in THRESHOLDS:
        pb  = pred >= thr
        tp  = float((pb & y).sum()); fp = float((pb & ~y).sum()); fn = float((~pb & y).sum())
        d   = tp + fp + fn
        iou = tp / d if d > 0 else 0.0
        pr  = tp / max(tp+fp, 1); rc = tp / max(tp+fn, 1)
        f1  = 2*pr*rc / max(pr+rc, 1e-9)
        if iou > best["iou"]:
            best = {"iou": iou, "thr": float(thr), "prec": pr, "rec": rc, "f1": f1}
    return best


def eval_set(model, npz_path: Path) -> tuple[dict, np.ndarray, float]:
    data  = np.load(npz_path)["data"]
    y     = (data[:, 4] > 0.5)
    pred  = predict(model, data)
    spread = float(np.percentile(pred, 90) - np.percentile(pred, 10))
    best  = threshold_sweep(pred, y)
    return best, y, spread


print("=" * 90)
print(f" DENSE ALPHA FULL-GRID SPATIAL HOLDOUT EVALUATION   device={device}")
print("=" * 90)
print()

# ── Spatial holdout table ────────────────────────────────────────────────────
print(" SPATIAL HOLDOUT (z-block, never seen in training):")
print(f" {'α':>4}  {'occ_holdout':>11}  {'trivial':>8}  {'model':>7}  {'Δ':>7}  "
      f"{'prec':>6}  {'rec':>6}  {'F1':>6}  {'thr':>5}  {'spread':>7}  {'verdict':>12}")
print(f" {'----':>4}  {'-----------':>11}  {'--------':>8}  {'-------':>7}  {'-------':>7}  "
      f"{'------':>6}  {'------':>6}  {'------':>6}  {'-----':>5}  {'-------':>7}  {'------------':>12}")

holdout_results = {}
for alpha in ALPHAS:
    ds_dir = DS_BASE / f"alpha_{alpha:03d}"
    meta   = json.loads((ds_dir / "metadata.json").read_text())

    model, ckpt = load_best(alpha)
    if model is None:
        print(f" {alpha:>4}  {'NO MODEL':>60}")
        continue

    best_h, y_h, spread_h = eval_set(model, ds_dir / "test_holdout.npz")
    occ_h    = float(y_h.mean())
    trivial_h = occ_h
    delta_h  = best_h["iou"] - trivial_h
    verdict  = "NON-TRIVIAL" if delta_h > 0.02 else "= trivial"

    print(f" {alpha:>4}  {occ_h:>11.4f}  {trivial_h:>8.4f}  {best_h['iou']:>7.4f}  "
          f"{delta_h:>+7.4f}  {best_h['prec']:>6.4f}  {best_h['rec']:>6.4f}  "
          f"{best_h['f1']:>6.4f}  {best_h['thr']:>5.2f}  {spread_h:>7.4f}  {verdict:>12}")

    holdout_results[alpha] = {
        "occ_holdout": occ_h, "trivial_holdout": trivial_h,
        "holdout_iou": best_h["iou"], "holdout_delta": delta_h,
        "holdout_prec": best_h["prec"], "holdout_rec": best_h["rec"],
        "holdout_f1": best_h["f1"], "holdout_thr": best_h["thr"],
        "holdout_spread": spread_h,
        "best_epoch": ckpt["best_epoch"], "val_bce": float(ckpt["best_metric"]),
    }

print()

# ── Full grid table (for comparison with per-alpha balanced) ─────────────────
print(" FULL GRID (377,500 voxels — comparison with per-alpha balanced results):")
print(f" {'α':>4}  {'occ_full':>9}  {'trivial':>8}  {'model':>7}  {'Δ':>7}  "
      f"{'prec':>6}  {'rec':>6}  {'spread':>7}  {'verdict':>12}")
print(f" {'----':>4}  {'---------':>9}  {'--------':>8}  {'-------':>7}  {'-------':>7}  "
      f"{'------':>6}  {'------':>6}  {'-------':>7}  {'------------':>12}")

full_results = {}
for alpha in ALPHAS:
    ds_dir = DS_BASE / f"alpha_{alpha:03d}"
    meta   = json.loads((ds_dir / "metadata.json").read_text())
    occ_f  = meta["occ_frac_full"]

    if alpha not in holdout_results:
        continue
    model, _ = load_best(alpha)

    best_f, y_f, spread_f = eval_set(model, ds_dir / "test_full_grid.npz")
    delta_f = best_f["iou"] - occ_f
    verdict = "NON-TRIVIAL" if delta_f > 0.02 else "= trivial"

    print(f" {alpha:>4}  {occ_f:>9.4f}  {occ_f:>8.4f}  {best_f['iou']:>7.4f}  "
          f"{delta_f:>+7.4f}  {best_f['prec']:>6.4f}  {best_f['rec']:>6.4f}  "
          f"{spread_f:>7.4f}  {verdict:>12}")

    full_results[alpha] = {
        "occ_full": occ_f, "full_iou": best_f["iou"], "full_delta": delta_f,
        "full_prec": best_f["prec"], "full_rec": best_f["rec"],
        "full_spread": spread_f,
    }

print("=" * 90)

# Summary
n_nt = sum(1 for v in holdout_results.values() if v["holdout_delta"] > 0.02)
print(f" Holdout non-trivial: {n_nt}/{len(holdout_results)} dense alphas")
if holdout_results:
    mean_iou = np.mean([v["holdout_iou"] for v in holdout_results.values()])
    mean_rec = np.mean([v["holdout_rec"] for v in holdout_results.values()])
    mean_spr = np.mean([v["holdout_spread"] for v in holdout_results.values()])
    print(f" Mean holdout IoU: {mean_iou:.4f}   Mean recall: {mean_rec:.4f}   "
          f"Mean pred_spread: {mean_spr:.4f}")
print()
print(" Reference (per-alpha balanced 80k, full-grid eval):")
print("   α=65: trivial (spread=0.094)   α=70: trivial (0.124)")
print("   α=75: trivial (spread=0.110)   α=80: trivial (0.202)")
print("=" * 90)

# Save results
combined = {}
for alpha in ALPHAS:
    if alpha in holdout_results:
        combined[str(alpha)] = {**holdout_results[alpha],
                                 **(full_results.get(alpha, {}))}

out = RUNS_DIR / "fullgrid_dense_spatial_holdout_results.json"
out.write_text(json.dumps(combined, indent=2))
print(f"\nResults saved: {out.relative_to(PROJECT_ROOT)}")
