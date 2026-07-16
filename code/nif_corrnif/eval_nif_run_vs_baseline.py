"""
Generalized per-alpha threshold-swept IoU comparison: ANY trained NIF run
(given a --run-dir and its --dataset-dir) vs. a baseline run (default: the
ORIGINAL pooled-seed hard-label run behind Table tab:hard_iou in
P2_density_calibrated_NIF_GLAD.tex).

This is a generalization of
runs/nif_hardlabel_seed0_plainbce_20260705_222336/analysis/eval_seed_isolation_diagnostic.py
(tier-1 seed-isolation diagnostic, 2026-07-05), which hardcoded two specific
run directories. Same methodology, but takes model paths as CLI arguments so
it can be reused for tier-2 Fix A (hash-grid encoding) and Fix B (continuous
regression) without copy-pasting a new throwaway script for each.

Usage:
  python3 eval_nif_run_vs_baseline.py --run-dir runs/nif_hashgrid_levelA_helical_YYYYMMDD_HHMMSS \\
      --dataset-dir datasets/levelA_helical_nif_hardlabel

  python3 eval_nif_run_vs_baseline.py --run-dir runs/nif_continuous_regression_YYYYMMDD_HHMMSS \\
      --dataset-dir datasets/levelA_helical_nif --candidate-is-regression

  # Compare two arbitrary runs directly (skip the default baseline):
  python3 eval_nif_run_vs_baseline.py --run-dir <A> --dataset-dir <dsA> \\
      --baseline-run-dir <B> --baseline-dataset-dir <dsB>

Notes:
  - "--candidate-is-regression" only changes how the candidate's raw
    prediction is produced (still just model(x) — FiLMConditionedNIF's
    sigmoid output range is unchanged by loss choice) — the per-alpha
    threshold sweep and IoU definition are identical either way, since IoU
    against a binarized ground truth is well-defined regardless of whether
    the model was trained with BCE or MSE/L1. To compare against the soft
    (continuous) ground truth directly, pass --dataset-dir pointing at the
    continuous dataset; the ">= threshold" binarization is applied to BOTH
    prediction and target symmetrically, matching the paper's existing
    soft-label evaluation methodology (Table tab:soft_iou).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate /mnt/d/GLAD_PROJECT root")


PROJECT_ROOT = find_project_root()
PINN_ROOT    = PROJECT_ROOT / "02_PINN_NIF" / "pinn_training"
MODEL_PATH   = PINN_ROOT / "models" / "nif_levelA_helical.py"

DEFAULT_BASELINE_RUN     = PINN_ROOT / "runs" / "nif_hardlabel_levelA_helical_20260615_113351"
DEFAULT_BASELINE_DATASET = PINN_ROOT / "datasets" / "levelA_helical_nif_hardlabel" / "test_samples.npz"
ALPHAS = [60, 65, 70, 75, 80, 85, 87, 89]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve(p: str | Path) -> Path:
    """Resolve a path relative to pinn_training/ (where run-dir / dataset-dir
    are conventionally given, e.g. 'runs/foo_20260706_...' or
    'datasets/levelA_helical_nif'); absolute paths pass through unchanged."""
    p = Path(p)
    return p if p.is_absolute() else PINN_ROOT / p


def load_model_module():
    spec = importlib.util.spec_from_file_location("nif_levelA_helical", MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_model(run_dir: Path, mod):
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = mod.build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def per_alpha_eval(model, npz_path: Path, alphas_expected):
    arr = np.load(npz_path)["data"].astype(np.float32)
    x = arr[:, :-1]
    y = arr[:, -1]
    alpha_n = x[:, 3]
    alpha_deg = np.round(alpha_n * 14.5 + 74.5, 1)

    results = {}
    with torch.no_grad():
        xb = torch.tensor(x, device=device)
        pred = model(xb).cpu().numpy().ravel()

    for a in alphas_expected:
        mask = np.isclose(alpha_deg, a, atol=0.6)
        n = mask.sum()
        if n == 0:
            continue
        yb = (y[mask] >= 0.5).astype(np.float32)   # binarize ground truth for IoU (matches paper convention)
        pb = pred[mask]
        occ_frac = float(yb.mean())
        trivial_iou = occ_frac  # predict-all-1 baseline
        best_iou = -1.0
        best_thr = None
        for thr in np.arange(0.05, 0.96, 0.01):
            pbin = (pb >= thr).astype(np.float32)
            tp = float(((pbin == 1) & (yb == 1)).sum())
            fp = float(((pbin == 1) & (yb == 0)).sum())
            fn = float(((pbin == 0) & (yb == 1)).sum())
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_thr = float(thr)
        results[str(a)] = {
            "n": int(n),
            "occ_frac": occ_frac,
            "trivial_iou": trivial_iou,
            "model_iou": best_iou,
            "best_thr": best_thr,
            "delta": best_iou - trivial_iou,
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Candidate run directory (relative to pinn_training/ or absolute).")
    ap.add_argument("--dataset-dir", required=True, help="Dataset dir containing test_samples.npz for the candidate run.")
    ap.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE_RUN))
    ap.add_argument("--baseline-dataset-dir", default=None,
                     help="Defaults to the original pooled hard-label test set.")
    ap.add_argument("--candidate-is-regression", action="store_true",
                     help="Informational only (see module docstring); does not change the eval logic.")
    ap.add_argument("--out-json", default="/tmp/eval_tier2_vs_baseline_results.json")
    args = ap.parse_args()

    mod = load_model_module()

    cand_run = resolve(args.run_dir)
    cand_ds  = resolve(args.dataset_dir)
    cand_test = cand_ds / "test_samples.npz" if (cand_ds / "test_samples.npz").exists() else cand_ds
    base_run = resolve(args.baseline_run_dir)
    base_test = resolve(args.baseline_dataset_dir) if args.baseline_dataset_dir else DEFAULT_BASELINE_DATASET

    print("=" * 70)
    print(f"CANDIDATE: {cand_run}")
    model_c, cfg_c = load_model(cand_run, mod)
    print(f"  architecture={cfg_c.get('architecture')}  dataset_test={cand_test}")
    res_c = per_alpha_eval(model_c, cand_test, ALPHAS)
    print(json.dumps(res_c, indent=2))

    print("=" * 70)
    print(f"BASELINE: {base_run}")
    model_b, cfg_b = load_model(base_run, mod)
    print(f"  architecture={cfg_b.get('architecture')}  dataset_test={base_test}")
    res_b = per_alpha_eval(model_b, base_test, ALPHAS)
    print(json.dumps(res_b, indent=2))

    print("=" * 70)
    print("COMPARISON (delta = model_iou - trivial_iou; positive = non-trivial spatial learning)")
    print(f"{'alpha':>6} | {'baseline delta':>15} | {'candidate delta':>16} | candidate helps?")
    for a in ALPHAS:
        ka = str(a)
        db = res_b.get(ka, {}).get("delta")
        dc = res_c.get(ka, {}).get("delta")
        if db is None or dc is None:
            continue
        helps = "YES" if (dc > db + 0.01) else ("no" if dc < db - 0.01 else "~same")
        marker = " <-- multi-seed angle" if a in (80, 85) else ""
        print(f"{a:>6} | {db:>15.4f} | {dc:>16.4f} | {helps}{marker}")

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps({"candidate": res_c, "baseline": res_b,
                                     "candidate_run": str(cand_run), "baseline_run": str(base_run)}, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
