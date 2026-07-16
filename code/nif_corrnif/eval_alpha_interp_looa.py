"""
Leave-one-alpha-out evaluation for AlphaInterpolatingFlatNIF (Option C).

PRIMARY evaluation: test_train_region.npz — z[0:k_split] of the held-out alpha.
  Tests PURE alpha-interpolation: z-range is the training z-region, so z-generalization
  is NOT in play. Only challenge: predicting morphology at an unseen alpha.

SECONDARY evaluation: test_holdout.npz — z[k_split:] of the held-out alpha.
  Conflates alpha-interpolation with z-generalization (expected to fail per
  corotating-frame diagnostic showing column drift in sparse films).

Success criterion (primary): IoU > occ_frac + 0.02  (non-trivial spatial discrimination).
Gate: APPROVE_ALPHA_INTERPOLATING_NIF_TRAINING
"""
from __future__ import annotations
import importlib.util, json, argparse
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
DS_BASE       = PINN_NIF_ROOT / "pinn_training/datasets/nif_alpha_interp"
ALPHAS        = [65, 70, 75, 80, 85, 87, 89]
BATCH         = 65536
THRESHOLDS    = np.concatenate([np.arange(0.02, 0.50, 0.02), np.arange(0.50, 0.96, 0.05)])

spec = importlib.util.spec_from_file_location("nif", MODEL_PATH)
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = argparse.ArgumentParser()
p.add_argument("--held-out", type=int, default=None)
args = p.parse_args()

targets = [args.held_out] if args.held_out else ALPHAS


def load_best(held_out: int):
    ho3 = f"{held_out:03d}"
    candidates = sorted(RUNS_DIR.glob(f"nif_alpha_interp_looa{ho3}_*"), reverse=True)
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
    X = torch.tensor(data[:, :-1])
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), BATCH):
            preds.append(model(X[s:s+BATCH].to(device)).cpu().numpy())
    return np.concatenate(preds)


def threshold_sweep(pred: np.ndarray, y: np.ndarray):
    best = {"iou": -1, "thr": 0.5, "prec": 0, "rec": 0, "f1": 0}
    for thr in THRESHOLDS:
        pb = pred >= thr
        tp = float((pb & y).sum()); fp = float((pb & ~y).sum()); fn = float((~pb & y).sum())
        d = tp + fp + fn
        iou = tp/d if d > 0 else 0.0
        pr = tp/max(tp+fp, 1); rc = tp/max(tp+fn, 1)
        f1 = 2*pr*rc/max(pr+rc, 1e-9)
        if iou > best["iou"]:
            best = {"iou": iou, "thr": float(thr), "prec": pr, "rec": rc, "f1": f1}
    return best


def eval_npz(model, npz_path: Path):
    data   = np.load(npz_path)["data"]
    y      = (data[:, -1] > 0.5)
    pred   = predict(model, data)
    spread = float(np.percentile(pred, 90) - np.percentile(pred, 10))
    return threshold_sweep(pred, y), float(y.mean()), spread


W = 95
print("=" * W)
print(f" ALPHA-INTERPOLATING LOOA EVALUATION   device={device}")
print(f" Architecture: AlphaInterpolatingFlatNIF   (no FiLM, 6D joint Fourier)")
print(f" Training alphas: 6 of 7  |  Held-out: 1 alpha never seen during training")
print("=" * W)
print()
print(" PRIMARY METRIC — test_train_region.npz  [z in training zone, pure alpha interpolation]")
HEADER = (f" {'held-out α':>10}  {'train_αs':>26}  {'occ_tr':>7}  {'trivial':>8}  "
          f"{'model':>7}  {'Δ':>7}  {'prec':>6}  {'rec':>6}  {'spread':>7}  {'verdict':>12}")
SEP = " " + "-"*(W-2)
print(HEADER); print(SEP)

results = {}
for held_out in targets:
    ho3 = f"{held_out:03d}"
    ds_dir = DS_BASE / f"looa_holdout_{ho3}"
    if not ds_dir.exists():
        print(f" {held_out:>10}  {'DATASET MISSING':>70}")
        continue
    meta = json.loads((ds_dir / "metadata.json").read_text())
    model, ckpt = load_best(held_out)
    if model is None:
        print(f" {held_out:>10}  {'NO MODEL':>70}")
        continue

    tr_path = ds_dir / "test_train_region.npz"
    if not tr_path.exists():
        print(f" {held_out:>10}  {'REBUILD NEEDED (test_train_region.npz missing)':>70}")
        continue

    best_tr, occ_tr, spread_tr = eval_npz(model, tr_path)
    delta_tr = best_tr["iou"] - occ_tr
    verdict_tr = "NON-TRIVIAL" if delta_tr > 0.02 else "= trivial"
    train_str = str(meta["train_alphas"])

    print(f" {held_out:>10}  {train_str:>26}  {occ_tr:>7.4f}  {occ_tr:>8.4f}  "
          f"{best_tr['iou']:>7.4f}  {delta_tr:>+7.4f}  {best_tr['prec']:>6.4f}  "
          f"{best_tr['rec']:>6.4f}  {spread_tr:>7.4f}  {verdict_tr:>12}")

    # z-holdout: secondary (expected to fail — column drift)
    best_ho, occ_ho, spread_ho = eval_npz(model, ds_dir / "test_holdout.npz")
    delta_ho = best_ho["iou"] - occ_ho

    results[held_out] = {
        "train_alphas":       meta["train_alphas"],
        # Primary: training z-region of held-out alpha
        "occ_train_region":   occ_tr,
        "iou_train_region":   best_tr["iou"],
        "delta_train_region": delta_tr,
        "prec_train_region":  best_tr["prec"],
        "rec_train_region":   best_tr["rec"],
        "spread_train_region": spread_tr,
        # Secondary: z-holdout of held-out alpha
        "occ_holdout":        occ_ho,
        "iou_holdout":        best_ho["iou"],
        "delta_holdout":      delta_ho,
        "spread_holdout":     spread_ho,
        "best_epoch":         ckpt["best_epoch"],
        "val_bce":            float(ckpt["best_metric"]),
    }

print()
print(" SECONDARY — test_holdout.npz  [z-holdout of held-out alpha, both α+z unseen — expected fail]")
print(HEADER); print(SEP)
for held_out in targets:
    if held_out not in results:
        continue
    r = results[held_out]
    delta_ho = r["delta_holdout"]
    verdict_ho = "NON-TRIVIAL" if delta_ho > 0.02 else "= trivial"
    print(f" {held_out:>10}  {str(r['train_alphas']):>26}  {r['occ_holdout']:>7.4f}  "
          f"{r['occ_holdout']:>8.4f}  {r['iou_holdout']:>7.4f}  {delta_ho:>+7.4f}  "
          f"{'—':>6}  {'—':>6}  {r['spread_holdout']:>7.4f}  {verdict_ho:>12}")

print("=" * W)
n_nt_tr = sum(1 for v in results.values() if v["delta_train_region"] > 0.02)
n_nt_ho = sum(1 for v in results.values() if v["delta_holdout"] > 0.02)
print(f" Primary  (α-interp, z-in-range): Non-trivial = {n_nt_tr}/{len(results)}")
print(f" Secondary (α+z joint holdout):   Non-trivial = {n_nt_ho}/{len(results)}  (expected 0/N)")
if results:
    print(f" Mean primary Δ:   {np.mean([v['delta_train_region'] for v in results.values()]):+.4f}")
    print(f" Mean secondary Δ: {np.mean([v['delta_holdout']      for v in results.values()]):+.4f}")
print("=" * W)

out = RUNS_DIR / "alpha_interp_looa_results.json"
out.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2))
print(f"\nResults saved: {out.relative_to(PROJECT_ROOT)}")
