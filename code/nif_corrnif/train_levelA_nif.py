#!/usr/bin/env python
"""
Training skeleton for the Level-A 300 nm helical NIF.

DEFAULT BEHAVIOUR IS DRY-RUN. No optimisation, no checkpoints, no files
written unless you pass --run explicitly.

Dry-run (safe):
  python pinn_training/train_levelA_nif.py \
    --manifest pinn_training/datasets/levelA_300nm_nif_manifest.csv \
    --split-strategy conservative_interpolation \
    --model siren_small \
    --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate /mnt/d/GLAD_PROJECT root")


PROJECT_ROOT = find_project_root()
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"


def resolve_project_path(value: str | Path, default_root: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    text = str(value).replace("\\", "/")
    if text == ".":
        return default_root
    if text == "pinn_training" or text.startswith("pinn_training/"):
        rest = text[len("pinn_training"):].lstrip("/")
        return (PINN_NIF_ROOT / "pinn_training").joinpath(*[part for part in rest.split("/") if part])
    return default_root.joinpath(*[part for part in text.split("/") if part])


def build_model(model_type: str):
    import torch.nn as nn
    import torch

    class SIREN(nn.Module):
        def __init__(self, hidden, depth, w0=30.0):
            super().__init__()
            self.w0 = w0
            dims = [4] + [hidden] * (depth - 1) + [1]
            self.lin = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

        def forward(self, x):
            for k, layer in enumerate(self.lin[:-1]):
                x = torch.sin((self.w0 if k == 0 else 1.0) * layer(x))
            return torch.sigmoid(self.lin[-1](x))

    class MLP(nn.Module):
        def __init__(self, hidden, depth):
            super().__init__()
            dims = [4] + [hidden] * (depth - 1) + [1]
            seq = []
            for i in range(len(dims) - 1):
                seq.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    seq.append(nn.SiLU())
            seq.append(nn.Sigmoid())
            self.net = nn.Sequential(*seq)

        def forward(self, x):
            return self.net(x)

    if model_type == "siren_small":
        return SIREN(256, 4)
    if model_type == "siren_medium":
        return SIREN(512, 6)
    if model_type == "mlp_small":
        return MLP(128, 4)
    raise ValueError(f"unknown model {model_type}")


SPLIT_KEY = {
    "conservative_interpolation": "strategy_A_conservative_interpolation",
    "seed_reproducibility": "strategy_C_seed_reproducibility",
    "all_data": "strategy_D_all_data_pretraining",
}


def dry_run(args) -> int:
    import torch
    from levelA_nif_dataset import LevelANIFManifest, LevelANIFDataset
    from torch.utils.data import DataLoader

    args.base_dir = str(resolve_project_path(args.base_dir, PROJECT_ROOT))
    args.manifest = str(resolve_project_path(args.manifest, Path(args.base_dir)))
    args.splits = str(resolve_project_path(args.splits, Path(args.base_dir)))
    args.checkpoint_dir = str(resolve_project_path(args.checkpoint_dir, PINN_NIF_ROOT))

    print("=" * 64)
    print("DRY-RUN (no training, no checkpoints)")
    print("=" * 64)
    for k in ("manifest", "split_strategy", "model", "batch_size", "epochs",
              "learning_rate", "device", "checkpoint_dir"):
        print(f"  {k:16s}: {getattr(args, k.replace('-', '_'))}")

    man = LevelANIFManifest(args.manifest, args.base_dir)
    print(f"[OK] manifest loaded: {len(man)} jobs")

    splits = json.loads(Path(args.splits).read_text())
    strat = splits[SPLIT_KEY[args.split_strategy]]
    print(f"[OK] split strategy: {strat['name']}")
    if "train" in strat:
        print(f"     train={len(strat.get('train', []))} "
              f"val={len(strat.get('validation', []))} "
              f"test={len(strat.get('test', []))}")

    model = build_model(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] model '{args.model}' built: {n_params:,} params on {args.device}")

    train_ds = LevelANIFDataset(args.manifest, base_dir=args.base_dir,
                                split_role="train", num_occupied=64, num_empty=64, seed=args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    xb, yb = next(iter(loader))
    print(f"[OK] train data: {len(train_ds)} samples; batch x={tuple(xb.shape)} y={tuple(yb.shape)}")

    # one forward pass to prove the wiring (no grad, no optim step)
    with torch.no_grad():
        out = model(xb)
    print(f"[OK] forward pass ok: out={tuple(out.shape)} range=[{out.min():.3f},{out.max():.3f}]")

    print("\nDRY-RUN COMPLETE. To train for real, re-run with --run.")
    return 0


def run_training(args) -> int:
    print("ACTUAL TRAINING REQUESTED (--run).")
    print("This skeleton intentionally does not implement the optimisation loop yet.")
    print("Planned: Adam + (MSE on rho), grad-clip 1.0, val every N epochs,")
    print("checkpoints {latest,best,final} under --checkpoint-dir, CSV/JSONL logs,")
    print("resume via --resume. See reports/levelA_nif_training_plan.md.")
    print("No checkpoints written.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Level-A 300 nm helical NIF training (dry-run by default)")
    ap.add_argument("--manifest", default="pinn_training/datasets/levelA_300nm_nif_manifest.csv")
    ap.add_argument("--splits", default="pinn_training/datasets/levelA_300nm_nif_splits.json")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--split-strategy", default="conservative_interpolation",
                    choices=list(SPLIT_KEY))
    ap.add_argument("--model", default="siren_small",
                    choices=["siren_small", "siren_medium", "mlp_small"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--checkpoint-dir", default="pinn_training/checkpoints/levelA_300nm_nif")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED to actually train; otherwise dry-run only")
    args = ap.parse_args()

    if args.run:
        return run_training(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
