"""
Train Level A helical NIF with focal loss and configurable patience.

Changes vs train_nif_levelA_helical.py:
  - Focal loss (FL = -alpha_t * (1-p_t)^gamma * log(p_t)) with configurable gamma/alpha
  - Weighted BCE option (pos_weight = n_neg/n_pos, read from normalization.json)
  - Early stopping patience default raised to 80 (overfit test shows breakthrough at epoch 120+)
  - Permission gates: APPROVE_SEED_ISOLATED_FOCAL_RETRAIN | APPROVE_PER_ALPHA_SEPARATE_TRAINING

Diagnosis that motivated this script (2026-06-15):
  - Single-film overfit test: FiLMConditionedNIF achieves IoU=0.737 vs trivial=0.625 on α=65
  - Production hard-label run stopped at epoch 90 (best=50, patience=40)
  - Breakthrough in overfit test starts at epoch 120-140 → patience=40 killed the run
  - Multi-seed label conflicts at α=80,85 create contradictory targets
  - Fix: seed-0-only dataset + focal loss + patience=100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate /mnt/d/GLAD_PROJECT root")


PROJECT_ROOT  = find_project_root()
ROOT          = PROJECT_ROOT
GLAD_ROOT     = PROJECT_ROOT / "01_GLAD_SIMULATION"
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
MODEL_PATH    = PINN_NIF_ROOT / "pinn_training" / "models" / "nif_levelA_helical.py"


def resolve_project_path(value: str | Path, default_root: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    text = str(value).replace("\\", "/")
    prefix_map = {
        "pinn_training": PINN_NIF_ROOT / "pinn_training",
    }
    for prefix, base in sorted(prefix_map.items(), key=lambda item: len(item[0]), reverse=True):
        if text == prefix:
            return base
        if text.startswith(prefix + "/"):
            return base.joinpath(*[part for part in text[len(prefix) + 1:].split("/") if part])
    return default_root.joinpath(*[part for part in text.split("/") if part])


def load_model_module():
    spec = importlib.util.spec_from_file_location("nif_levelA_helical", MODEL_PATH)
    mod  = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_run_dir(cfg: dict, resume_dir: str | None) -> Path:
    if resume_dir:
        return Path(resume_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (resolve_project_path(cfg.get("output_root", "pinn_training/runs"), PINN_NIF_ROOT)
            / f"{cfg.get('run_name', 'nif_hardlabel_v2')}_{stamp}")


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    numbered = sorted(ckpt_dir.glob("epoch_*.pt"))
    return numbered[-1] if numbered else None


def load_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.load(path)["data"].astype(np.float32)
    # Label is always the last column; features are all preceding columns.
    # Supports 4-feature (x,y,z,α) and 6-feature (x,y,z,α,φ_sin,φ_cos) datasets.
    return torch.tensor(arr[:, :-1]), torch.tensor(arr[:, -1])


def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float, alpha: float) -> torch.Tensor:
    """
    Binary focal loss. pred must be sigmoid output in (0,1).
    FL = -alpha_t * (1-p_t)^gamma * log(p_t)
    gamma=2 (standard): focuses on hard examples, down-weights easy confident cases.
    alpha=0.70: up-weights positive class to address class imbalance → raises recall.
    """
    pred   = pred.clamp(1e-7, 1.0 - 1e-7)
    p_t    = torch.where(target > 0.5, pred, 1.0 - pred)
    alpha_t = torch.where(target > 0.5,
                          torch.full_like(pred, alpha),
                          torch.full_like(pred, 1.0 - alpha))
    return (alpha_t * (1.0 - p_t).pow(gamma) * (-torch.log(p_t))).mean()


def weighted_bce_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE with per-class weighting. pos_weight = n_neg/n_pos (>1 upweights positives)."""
    pred = pred.clamp(1e-7, 1.0 - 1e-7)
    return -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred)).mean()


def build_loss_fn(cfg: dict, dataset_path: Path):
    loss_type = cfg.get("loss", "focal")
    if loss_type == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))
        alpha = float(cfg.get("focal_alpha", 0.70))
        print(f"Loss: focal  gamma={gamma}  alpha={alpha}")
        return lambda pred, tgt: focal_loss(pred, tgt, gamma, alpha)
    if loss_type == "weighted_bce":
        norm_path = dataset_path / "normalization.json"
        if norm_path.exists():
            norm = json.loads(norm_path.read_text())
            occ = float(norm.get("global_occ_frac_train", 0.358))
        else:
            occ = float(cfg.get("occ_frac", 0.358))
        pos_weight = (1.0 - occ) / max(occ, 1e-7)
        print(f"Loss: weighted_bce  pos_weight={pos_weight:.4f}")
        return lambda pred, tgt: weighted_bce_loss(pred, tgt, pos_weight)
    print("Loss: bce (standard)")
    bce = torch.nn.BCELoss()
    return lambda pred, tgt: bce(pred, tgt)


def evaluate(model, x, y, batch_size, device) -> dict:
    model.eval()
    losses = []
    tp = fp = fn = correct = total = 0
    bce = torch.nn.BCELoss(reduction="mean")
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size].to(device)
            yb = y[s:s + batch_size].to(device).clamp(0, 1)
            pred = model(xb)
            losses.append(float(bce(pred, yb).item()))
            pb = pred >= 0.5
            gb = yb >= 0.5
            correct += int((pb == gb).sum().item())
            total   += int(len(yb))
            tp += int((pb & gb).sum().item())
            fp += int((pb & ~gb).sum().item())
            fn += int((~pb & gb).sum().item())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    iou       = tp / max(tp + fp + fn, 1)
    return {
        "loss": float(np.mean(losses)),
        "occupancy_accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "tp": tp, "fp": fp, "fn": fn,
    }


def backup_existing(path: Path) -> None:
    if path.exists() and path.name in {"best.pt", "final.pt"}:
        backup = path.with_name(f"{path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        shutil.copy2(path, backup)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_existing(path)
    torch.save(payload, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NIF v2 with focal loss. Permission-gated.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--allow-train", action="store_true", help="Pass training gate.")
    args = parser.parse_args()

    if not args.allow_train:
        print("TRAINING_NOT_STARTED: gate not passed.")
        print("  Re-run with --allow-train after user confirms gate.")
        return 0

    cfg_path = resolve_project_path(args.config, PROJECT_ROOT).resolve()
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg      = load_config(cfg_path)

    KNOWN_GATES = {"APPROVE_SEED_ISOLATED_FOCAL_RETRAIN", "APPROVE_PER_ALPHA_SEPARATE_TRAINING",
                   "APPROVE_FULLGRID_DENSE_ALPHA_TRAINING", "APPROVE_PHASE_CONDITIONED_NIF_TRAINING",
                   "APPROVE_PHASEONLY_NIF_TRAINING", "APPROVE_ALPHA_INTERPOLATING_NIF_TRAINING",
                   "APPROVE_SEED_ISOLATED_PLAIN_BCE_RETRAIN",
                   "APPROVE_HASHGRID_SPATIAL_ENCODING_TRAINING"}
    perm_required = cfg.get("permission_required", "")
    if perm_required and perm_required not in KNOWN_GATES:
        print(f"GATE MISMATCH: config requires '{perm_required}', "
              f"script knows: {sorted(KNOWN_GATES)}")
        return 1

    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # CPU thread configuration — applied before any torch compute
    n_threads = int(cfg.get("n_threads", 8))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(min(n_threads, 4))

    device = torch.device(
        "cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu"
    )

    run_dir  = make_run_dir(cfg, args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config_used.yaml")

    dataset     = resolve_project_path(cfg["dataset_dir"], PINN_NIF_ROOT)
    x_train, y_train = load_npz(dataset / "train_samples.npz")
    x_val,   y_val   = load_npz(dataset / "val_samples.npz")

    print(f"Dataset: {dataset.relative_to(PROJECT_ROOT)}")
    print(f"  train={len(x_train):,}  val={len(x_val):,}")
    print(f"  global pos_frac (train): {y_train.mean():.4f}")

    mod   = load_model_module()
    model = mod.build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.get('architecture')}  params={n_params:,}  device={device}")
    if device.type == "cpu":
        print(f"CPU mode: torch_threads={n_threads}  (set OMP_NUM_THREADS={n_threads} in env for full effect)")
    else:
        print(f"GPU mode: {torch.cuda.get_device_name(0)}  cpu_threads={n_threads}")

    loss_fn = build_loss_fn(cfg, dataset)

    opt   = torch.optim.AdamW(model.parameters(),
                               lr=float(cfg["learning_rate"]),
                               weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs"]))

    start_epoch = 1
    best_loss   = float("inf")
    best_epoch  = 0
    resume_status = "RESUME NOT FOUND"

    if args.resume:
        lp = latest_checkpoint(ckpt_dir)
        if lp:
            ckpt = torch.load(lp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            sched.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_loss   = float(ckpt.get("best_metric", best_loss))
            best_epoch  = int(ckpt.get("best_epoch", 0))
            resume_status = f"RESUME FOUND {lp}"

    (run_dir / "resume_status.txt").write_text(
        f"{resume_status}\nstart_epoch={start_epoch}\nbest_metric={best_loss}\nbest_epoch={best_epoch}\n",
        encoding="utf-8",
    )

    log_csv = run_dir / "training_log.csv"
    new_log = not log_csv.exists()
    batch   = int(cfg["batch_size"])
    epochs  = int(cfg["epochs"])
    patience = int(cfg.get("early_stopping_patience", 100))
    save_every = int(cfg.get("save_every", 20))
    stale   = 0

    print(f"\nTraining: epochs={epochs}  batch={batch}  patience={patience}")
    print(f"Run dir: {run_dir.relative_to(PROJECT_ROOT)}\n")

    with (log_csv.open("a", encoding="utf-8", newline="") as fcsv,
          (run_dir / "training_log.jsonl").open("a", encoding="utf-8") as fj):
        writer = csv.DictWriter(fcsv, fieldnames=["epoch", "train_loss", "val_loss",
                                                    "val_iou", "val_precision", "val_recall", "lr"])
        if new_log:
            writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            perm   = torch.randperm(len(x_train))
            losses = []
            for s in range(0, len(perm), batch):
                idx = perm[s:s + batch]
                xb  = x_train[idx].to(device)
                yb  = y_train[idx].to(device).clamp(0.0, 1.0)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            sched.step()

            metrics    = evaluate(model, x_val, y_val, batch, device)
            train_loss = float(np.mean(losses))
            row = {
                "epoch":        epoch,
                "train_loss":   train_loss,
                "val_loss":     metrics["loss"],
                "val_iou":      metrics["iou"],
                "val_precision": metrics["precision"],
                "val_recall":   metrics["recall"],
                "lr":           opt.param_groups[0]["lr"],
            }
            writer.writerow(row); fcsv.flush()
            fj.write(json.dumps(row) + "\n"); fj.flush()

            print(f"  ep={epoch:>4}  train_fl={train_loss:.4f}  val_bce={metrics['loss']:.4f}"
                  f"  iou={metrics['iou']:.4f}  prec={metrics['precision']:.4f}"
                  f"  rec={metrics['recall']:.4f}  lr={opt.param_groups[0]['lr']:.2e}")

            payload = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch":     epoch,
                "best_metric": min(best_loss, metrics["loss"]),
                "best_epoch":  best_epoch,
                "config":    cfg,
                "config_hash": sha256_text(cfg_text),
                "seed":      seed,
                "normalization": (json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
                                  if (dataset / "normalization.json").exists() else {}),
                "metrics":   metrics,
            }
            save_checkpoint(ckpt_dir / "latest.pt", payload)

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]; best_epoch = epoch; stale = 0
                payload["best_metric"] = best_loss; payload["best_epoch"] = best_epoch
                save_checkpoint(ckpt_dir / "best.pt", payload)
                print(f"  ** NEW BEST  epoch={epoch}  val_bce={best_loss:.6f}  "
                      f"iou={metrics['iou']:.4f}  prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}")
            else:
                stale += 1

            if epoch % save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", payload)

            if patience and stale >= patience:
                print(f"Early stop: no val improvement for {patience} epochs. Best epoch={best_epoch}.")
                break

    final_payload = torch.load(ckpt_dir / "latest.pt", map_location=device, weights_only=False)
    save_checkpoint(ckpt_dir / "final.pt", final_payload)

    best_payload = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    best_metrics = best_payload["metrics"]

    (run_dir / "validation_summary.json").write_text(
        json.dumps(best_metrics, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "training_summary.md").write_text(
        f"# NIF v2 Training Summary\n\n"
        f"loss={cfg.get('loss','focal')}  gamma={cfg.get('focal_gamma',2.0)}  alpha={cfg.get('focal_alpha',0.70)}\n"
        f"best_val_bce={best_loss:.6f}  best_epoch={best_epoch}\n"
        f"best_iou={best_metrics['iou']:.4f}  precision={best_metrics['precision']:.4f}  recall={best_metrics['recall']:.4f}\n",
        encoding="utf-8",
    )

    print(f"\nTRAINING_COMPLETE {run_dir}")
    print(f"  best_epoch={best_epoch}  val_bce={best_loss:.6f}")
    print(f"  iou={best_metrics['iou']:.4f}  precision={best_metrics['precision']:.4f}  recall={best_metrics['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
