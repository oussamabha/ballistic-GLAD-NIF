"""
Full-grid evaluation for per-alpha NIF networks.

Evaluates each best.pt on the FULL VOXEL GRID (all 377,500 voxels, real film distribution).
This gives honest metrics — not inflated by balanced-sampling artefacts.

Trivial baseline:
  predict-all-1 → IoU = occ_frac (real film)
  predict-all-0 → IoU = 0
  A model is non-trivial only if IoU > occ_frac at the optimal threshold.
"""
from __future__ import annotations
import importlib.util, json
import numpy as np
import torch
from pathlib import Path

def find_project_root(start=Path(__file__).resolve()):
    for p in [start, *start.parents]:
        if (p/"00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("No project root")

PROJECT_ROOT  = find_project_root()
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
MODEL_PATH    = PINN_NIF_ROOT / "pinn_training/models/nif_levelA_helical.py"
RUNS_DIR      = PINN_NIF_ROOT / "pinn_training/runs"
DS_BASE       = PINN_NIF_ROOT / "pinn_training/datasets/nif_per_alpha"
ALPHAS        = [65, 70, 75, 80, 85, 87, 89]
BATCH         = 65536

spec = importlib.util.spec_from_file_location("nif", MODEL_PATH)
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best(alpha: int):
    candidates = sorted(RUNS_DIR.glob(f"nif_peralpha_{alpha:03d}_*"), reverse=True)
    if not candidates:
        return None, None
    run_dir = candidates[0]
    bp = run_dir / "checkpoints/best.pt"
    if not bp.exists():
        return None, None
    ckpt  = torch.load(bp, map_location="cpu", weights_only=False)
    model = mod.build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), ckpt


def threshold_sweep(pred: np.ndarray, y: np.ndarray, thresholds):
    best = {"iou": -1, "thr": 0.5, "prec": 0, "rec": 0, "f1": 0}
    for thr in thresholds:
        pb  = pred >= thr
        tp  = float((pb & y).sum()); fp = float((pb & ~y).sum()); fn = float((~pb & y).sum())
        d   = tp + fp + fn
        iou = tp / d if d > 0 else 0.0
        pr  = tp / max(tp+fp, 1); rc = tp / max(tp+fn, 1)
        f1  = 2*pr*rc / max(pr+rc, 1e-9)
        if iou > best["iou"]:
            best = {"iou": iou, "thr": float(thr), "prec": pr, "rec": rc, "f1": f1}
    return best


print("=" * 78)
print(f" PER-ALPHA FULL-GRID EVALUATION   device={device}")
print("=" * 78)
print(f" {'α':>4}  {'true_occ':>9}  {'trivial_IoU':>12}  {'model_IoU':>10}  "
      f"{'Δ':>8}  {'prec':>6}  {'rec':>6}  {'F1':>6}  {'thr':>5}  {'verdict':>12}")
print(f" {'----':>4}  {'---------':>9}  {'------------':>12}  {'----------':>10}  "
      f"{'--------':>8}  {'------':>6}  {'------':>6}  {'------':>6}  {'-----':>5}  {'------------':>12}")

results = {}
THRESHOLDS = np.concatenate([np.arange(0.02, 0.50, 0.02), np.arange(0.50, 0.96, 0.05)])

for alpha in ALPHAS:
    ds_dir = DS_BASE / f"alpha_{alpha:03d}"
    meta   = json.loads((ds_dir / "metadata.json").read_text())
    occ    = meta["occ_frac"]           # true film occupancy
    trivial_iou = occ                   # predict-all-1 on real film → IoU = occ_frac

    model, ckpt = load_best(alpha)
    if model is None:
        print(f" {alpha:>4}  {occ:>9.4f}  {'NO MODEL':>22}")
        continue

    # Full grid evaluation
    data = np.load(ds_dir / "test_full_grid.npz")["data"]
    X    = torch.tensor(data[:, :4])
    y    = (data[:, 4] > 0.5)

    preds = []
    with torch.no_grad():
        for s in range(0, len(X), BATCH):
            preds.append(model(X[s:s+BATCH].to(device)).cpu().numpy())
    pred = np.concatenate(preds)

    spread = float(np.percentile(pred, 90) - np.percentile(pred, 10))
    best   = threshold_sweep(pred, y, THRESHOLDS)
    delta  = best["iou"] - trivial_iou
    verdict = "NON-TRIVIAL" if delta > 0.01 else "= trivial"

    results[alpha] = {
        "true_occ": occ, "trivial_iou": trivial_iou,
        "model_iou": best["iou"], "delta": delta,
        "precision": best["prec"], "recall": best["rec"], "f1": best["f1"],
        "best_thr": best["thr"], "pred_spread": spread,
        "best_epoch": ckpt["best_epoch"], "val_bce": float(ckpt["best_metric"]),
        "run_dir": str(sorted(RUNS_DIR.glob(f"nif_peralpha_{alpha:03d}_*"), reverse=True)[0].relative_to(PROJECT_ROOT)),
    }
    print(f" {alpha:>4}  {occ:>9.4f}  {trivial_iou:>12.4f}  {best['iou']:>10.4f}  "
          f"{delta:>+8.4f}  {best['prec']:>6.4f}  {best['rec']:>6.4f}  {best['f1']:>6.4f}  "
          f"{best['thr']:>5.2f}  {verdict:>12}")

print("=" * 78)
n_nt = sum(1 for v in results.values() if v["delta"] > 0.01)
print(f" Non-trivial: {n_nt}/{len(results)} alphas")
if results:
    avg_iou = np.mean([v["model_iou"] for v in results.values()])
    avg_rec = np.mean([v["recall"]    for v in results.values()])
    print(f" Mean IoU (full grid): {avg_iou:.4f}   Mean Recall: {avg_rec:.4f}")
print("=" * 78)

out = PINN_NIF_ROOT / "pinn_training/runs/per_alpha_fullgrid_results.json"
out.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2))
print(f"\nResults saved: {out.relative_to(PROJECT_ROOT)}")
