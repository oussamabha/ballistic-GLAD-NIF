"""
Spatial holdout evaluation for phase-conditioned NIF networks.

Evaluates each best.pt on:
  1. test_holdout.npz — z-block holdout (z >= 80%, never seen in training)
  2. test_full_grid.npz — full 377,500 voxels (all z-positions)

Success criterion: holdout IoU > trivial baseline (occ_frac_holdout) by > 0.02.

Reference baselines:
  Full-grid z-block (no phase, 4 features): 0/4 dense alphas non-trivial on holdout
  Per-alpha balanced 80k (no phase):        3/7 non-trivial (sparse only)

Expected with phase conditioning:
  All 7 alphas non-trivial on holdout (helix rotation barrier removed).
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
DS_BASE       = PINN_NIF_ROOT / "pinn_training/datasets/nif_phase_conditioned"
ALPHAS        = [65, 70, 75, 80, 85, 87, 89]
BATCH         = 65536
THRESHOLDS    = np.concatenate([np.arange(0.02, 0.50, 0.02), np.arange(0.50, 0.96, 0.05)])

spec = importlib.util.spec_from_file_location("nif", MODEL_PATH)
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best(alpha: int):
    candidates = sorted(RUNS_DIR.glob(f"nif_phasecond_{alpha:03d}_*"), reverse=True)
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
    X = torch.tensor(data[:, :-1])    # all columns except last = features
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
    data   = np.load(npz_path)["data"]
    y      = (data[:, -1] > 0.5)
    pred   = predict(model, data)
    spread = float(np.percentile(pred, 90) - np.percentile(pred, 10))
    return threshold_sweep(pred, y), y, spread


print("=" * 95)
print(f" PHASE-CONDITIONED NIF — SPATIAL HOLDOUT EVALUATION   device={device}")
print(f" Architecture: PhaseConditionedFiLMNIF   inputs: (x,y,z,α,φ_sin,φ_cos)")
print("=" * 95)
print()

# ── Spatial holdout ─────────────────────────────────────────────────────────
print(" SPATIAL HOLDOUT (z >= 80%, phase seen in training but z-position novel):")
print(f" {'α':>4}  {'occ_hold':>9}  {'trivial':>8}  {'model':>7}  {'Δ':>7}  "
      f"{'prec':>6}  {'rec':>6}  {'F1':>6}  {'thr':>5}  {'spread':>7}  {'verdict':>12}")
print(f" {'----':>4}  {'---------':>9}  {'--------':>8}  {'-------':>7}  {'-------':>7}  "
      f"{'------':>6}  {'------':>6}  {'------':>6}  {'-----':>5}  {'-------':>7}  {'------------':>12}")

holdout_res = {}
for alpha in ALPHAS:
    ds_dir = DS_BASE / f"alpha_{alpha:03d}"
    meta   = json.loads((ds_dir / "metadata.json").read_text())

    model, ckpt = load_best(alpha)
    if model is None:
        print(f" {alpha:>4}  {'NO MODEL':>70}")
        continue

    best_h, y_h, spread_h = eval_set(model, ds_dir / "test_holdout.npz")
    occ_h   = float(y_h.mean())
    delta_h = best_h["iou"] - occ_h
    verdict = "NON-TRIVIAL" if delta_h > 0.02 else "= trivial"

    print(f" {alpha:>4}  {occ_h:>9.4f}  {occ_h:>8.4f}  {best_h['iou']:>7.4f}  "
          f"{delta_h:>+7.4f}  {best_h['prec']:>6.4f}  {best_h['rec']:>6.4f}  "
          f"{best_h['f1']:>6.4f}  {best_h['thr']:>5.2f}  {spread_h:>7.4f}  {verdict:>12}")

    holdout_res[alpha] = {
        "occ_holdout": occ_h, "holdout_iou": best_h["iou"], "holdout_delta": delta_h,
        "holdout_prec": best_h["prec"], "holdout_rec": best_h["rec"],
        "holdout_f1": best_h["f1"], "holdout_thr": best_h["thr"], "holdout_spread": spread_h,
        "best_epoch": ckpt["best_epoch"], "val_bce": float(ckpt["best_metric"]),
    }

print()

# ── Full grid ────────────────────────────────────────────────────────────────
print(" FULL GRID (377,500 voxels):")
print(f" {'α':>4}  {'occ_full':>9}  {'trivial':>8}  {'model':>7}  {'Δ':>7}  "
      f"{'prec':>6}  {'rec':>6}  {'spread':>7}  {'verdict':>12}")
print(f" {'----':>4}  {'---------':>9}  {'--------':>8}  {'-------':>7}  {'-------':>7}  "
      f"{'------':>6}  {'------':>6}  {'-------':>7}  {'------------':>12}")

full_res = {}
for alpha in ALPHAS:
    if alpha not in holdout_res:
        continue
    ds_dir = DS_BASE / f"alpha_{alpha:03d}"
    meta   = json.loads((ds_dir / "metadata.json").read_text())
    occ_f  = meta["occ_frac_full"]
    model, _ = load_best(alpha)

    best_f, _, spread_f = eval_set(model, ds_dir / "test_full_grid.npz")
    delta_f = best_f["iou"] - occ_f
    verdict = "NON-TRIVIAL" if delta_f > 0.02 else "= trivial"

    print(f" {alpha:>4}  {occ_f:>9.4f}  {occ_f:>8.4f}  {best_f['iou']:>7.4f}  "
          f"{delta_f:>+7.4f}  {best_f['prec']:>6.4f}  {best_f['rec']:>6.4f}  "
          f"{spread_f:>7.4f}  {verdict:>12}")

    full_res[alpha] = {
        "occ_full": occ_f, "full_iou": best_f["iou"], "full_delta": delta_f,
        "full_prec": best_f["prec"], "full_rec": best_f["rec"], "full_spread": spread_f,
    }

print("=" * 95)

n_nt = sum(1 for v in holdout_res.values() if v["holdout_delta"] > 0.02)
print(f" Holdout non-trivial: {n_nt}/{len(holdout_res)}")
if holdout_res:
    print(f" Mean holdout IoU:  {np.mean([v['holdout_iou']   for v in holdout_res.values()]):.4f}")
    print(f" Mean holdout Δ:    {np.mean([v['holdout_delta'] for v in holdout_res.values()]):+.4f}")
    print(f" Mean pred_spread:  {np.mean([v['holdout_spread'] for v in holdout_res.values()]):.4f}")
print()
print(" Comparison baselines (holdout non-trivial count):")
print("   Full-grid z-block (no phase, 4-feat): 0/4 dense, 3/7 total")
print("   Per-alpha balanced 80k (no phase):    3/7 (sparse only)")
print("   Phase-conditioned (this run):         ?/7  ← above")
print("=" * 95)

out = RUNS_DIR / "phase_conditioned_spatial_holdout_results.json"
combined = {str(a): {**holdout_res.get(a, {}), **full_res.get(a, {})} for a in ALPHAS}
out.write_text(json.dumps(combined, indent=2))
print(f"\nResults saved: {out.relative_to(PROJECT_ROOT)}")
