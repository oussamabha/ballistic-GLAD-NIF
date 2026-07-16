from __future__ import annotations

import argparse
import csv
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
ROOT = PROJECT_ROOT
GLAD_ROOT = PROJECT_ROOT / "01_GLAD_SIMULATION"
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
VOXEL_DESC_ROOT = PROJECT_ROOT / "03_VOXEL_DESCRIPTOR_ANALYSIS"
LIT_ROOT = PROJECT_ROOT / "04_LITERATURE_AND_PARAMETER_WORKFLOW"
THESIS_ROOT = PROJECT_ROOT / "05_THESIS_PAPER_ASSETS"
EXPERIMENTAL_ROOT = PROJECT_ROOT / "06_EXPERIMENTAL_VALIDATION"
UTIL_ROOT = PROJECT_ROOT / "07_UTILITIES_GENERATORS"
LEGACY_ROOT = PROJECT_ROOT / "08_LEGACY_REVIEW"


def resolve_project_path(value: str | Path, default_root: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    text = str(value).replace("\\", "/")
    marker = "deepseek_correction/"
    if marker in text:
        text = text.split(marker, 1)[1]
    prefix_map = {
        "pinn_training": PINN_NIF_ROOT / "pinn_training",
        "analysis": VOXEL_DESC_ROOT / "analysis",
        "descriptor_results_hybrid": VOXEL_DESC_ROOT / "descriptor_results_hybrid",
        "simulation_batch": GLAD_ROOT / "simulation_batch",
    }
    for prefix, base in sorted(prefix_map.items(), key=lambda item: len(item[0]), reverse=True):
        if text == prefix:
            return base
        if text.startswith(prefix + "/"):
            return base.joinpath(*[part for part in text[len(prefix) + 1:].split("/") if part])
    if text == ".":
        return default_root
    return default_root.joinpath(*[part for part in text.split("/") if part])


MODEL_PATH = PINN_NIF_ROOT / "pinn_training" / "models" / "nif_levelA_helical.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("nif_levelA_helical", MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_npz(path: Path):
    arr = np.load(path)["data"].astype(np.float32)
    return torch.tensor(arr[:, :4]), torch.tensor(arr[:, 4])


def metrics(pred: torch.Tensor, y: torch.Tensor) -> dict:
    pred = pred.detach().cpu()
    y = y.detach().cpu().clamp(0, 1)
    mse = float(torch.mean((pred - y) ** 2).item())
    pb = pred >= 0.5
    gb = y >= 0.5
    tp = int((pb & gb).sum().item()); fp = int((pb & ~gb).sum().item()); fn = int((~pb & gb).sum().item())
    return {"density_mse": mse, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Level A helical NIF run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default="pinn_training/datasets/levelA_helical_nif")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    out = run / "evaluation"; out.mkdir(parents=True, exist_ok=True)
    ckpt_path = run / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit("Missing best checkpoint; evaluation not run.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mod = load_model_module()
    model = mod.build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    rows = []
    dataset_dir = resolve_project_path(args.dataset_dir, PINN_NIF_ROOT)
    for split in ["train", "val", "test"]:
        x, y = load_npz(dataset_dir / f"{split}_samples.npz")
        with torch.no_grad():
            pred = model(x)
        rows.append({"split": split, **metrics(pred, y)})
    with (out / "per_alpha_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for name in ["per_seed_metrics.csv", "heldout_alpha_reconstruction_metrics.csv", "descriptor_comparison.csv"]:
        (out / name).write_text("status,note\nTODO,Requires completed Level A simulation metadata and voxel ground truth\n", encoding="utf-8")
    (out / "evaluation_report.md").write_text("# Level A Evaluation Report\n\nInitial split metrics written. Descriptor and held-out alpha evaluation require completed Level A dataset metadata.\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
