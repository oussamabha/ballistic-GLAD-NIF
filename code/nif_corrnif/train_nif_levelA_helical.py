from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_run_dir(cfg: dict, resume_dir: str | None) -> Path:
    if resume_dir:
        return Path(resume_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_project_path(cfg.get("output_root", "pinn_training/runs"), PINN_NIF_ROOT) / f"{cfg.get('run_name', 'nif_levelA_helical')}_{stamp}"


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    numbered = sorted(ckpt_dir.glob("epoch_*.pt"))
    return numbered[-1] if numbered else None


def load_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.load(path)["data"].astype(np.float32)
    return torch.tensor(arr[:, :4]), torch.tensor(arr[:, 4])


def evaluate(model, x, y, batch_size, device) -> dict:
    model.eval()
    losses = []
    correct = total = tp = fp = fn = 0
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
            total += int(len(yb))
            tp += int((pb & gb).sum().item())
            fp += int((pb & ~gb).sum().item())
            fn += int((~pb & gb).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {"loss": float(np.mean(losses)), "occupancy_accuracy": correct / max(total, 1), "precision": precision, "recall": recall, "iou": iou}


def backup_existing(path: Path) -> None:
    if path.exists() and path.name in {"best.pt", "final.pt"}:
        backup = path.with_name(f"{path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        shutil.copy2(path, backup)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_existing(path)
    torch.save(payload, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Level A helical NIF. Permission-gated and resume-safe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--allow-train", action="store_true", help="Required to actually train.")
    args = parser.parse_args()
    if not args.allow_train:
        print("TRAINING_NOT_STARTED: pass --allow-train only after user says RUN LEVEL A TRAINING NOW")
        return 0
    cfg_path = resolve_project_path(args.config, PROJECT_ROOT).resolve()
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = load_config(cfg_path)
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg, args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config_used.yaml")
    dataset = resolve_project_path(cfg["dataset_dir"], PINN_NIF_ROOT)
    x_train, y_train = load_npz(dataset / "train_samples.npz")
    x_val, y_val = load_npz(dataset / "val_samples.npz")
    mod = load_model_module()
    model = mod.build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs"]))
    start_epoch = 1; best_loss = float("inf"); best_epoch = 0
    resume_status = "RESUME NOT FOUND"
    if args.resume:
        lp = latest_checkpoint(ckpt_dir)
        if lp:
            ckpt = torch.load(lp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            sched.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_loss = float(ckpt.get("best_metric", best_loss))
            best_epoch = int(ckpt.get("best_epoch", 0))
            resume_status = f"RESUME FOUND {lp}"
    (run_dir / "resume_status.txt").write_text(f"{resume_status}\nstart_epoch={start_epoch}\nbest_metric={best_loss}\nbest_epoch={best_epoch}\n", encoding="utf-8")
    log_csv = run_dir / "training_log.csv"
    new_log = not log_csv.exists()
    bce = torch.nn.BCELoss()
    batch = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    patience = int(cfg.get("early_stopping_patience", 0))
    stale = 0
    with log_csv.open("a", encoding="utf-8", newline="") as fcsv, (run_dir / "training_log.jsonl").open("a", encoding="utf-8") as fj:
        writer = csv.DictWriter(fcsv, fieldnames=["epoch", "train_loss", "val_loss", "iou", "lr"])
        if new_log:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            perm = torch.randperm(len(x_train))
            losses = []
            for s in range(0, len(perm), batch):
                idx = perm[s:s + batch]
                xb = x_train[idx].to(device); yb = y_train[idx].to(device).clamp(0, 1)
                opt.zero_grad(set_to_none=True)
                loss = bce(model(xb), yb)
                loss.backward(); opt.step()
                losses.append(float(loss.item()))
            sched.step()
            metrics = evaluate(model, x_val, y_val, batch, device)
            train_loss = float(np.mean(losses))
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": metrics["loss"], "iou": metrics["iou"], "lr": opt.param_groups[0]["lr"]}
            writer.writerow(row); fcsv.flush()
            fj.write(json.dumps(row) + "\n"); fj.flush()
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch": epoch,
                "best_metric": min(best_loss, metrics["loss"]),
                "best_epoch": best_epoch,
                "config": cfg,
                "config_hash": sha256_text(cfg_text),
                "seed": seed,
                "normalization": json.loads((dataset / "normalization.json").read_text(encoding="utf-8")) if (dataset / "normalization.json").exists() else {},
                "metrics": metrics,
            }
            save_checkpoint(ckpt_dir / "latest.pt", payload)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]; best_epoch = epoch; stale = 0
                payload["best_metric"] = best_loss; payload["best_epoch"] = best_epoch
                save_checkpoint(ckpt_dir / "best.pt", payload)
            else:
                stale += 1
            if epoch % int(cfg.get("save_every", 10)) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", payload)
            if patience and stale >= patience:
                break
    final_payload = torch.load(ckpt_dir / "latest.pt", map_location=device, weights_only=False)
    save_checkpoint(ckpt_dir / "final.pt", final_payload)
    for result_path in [run_dir / "validation_summary.json", run_dir / "training_summary.md"]:
        if result_path.exists():
            backup = result_path.with_name(f"{result_path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{result_path.suffix}")
            shutil.copy2(result_path, backup)
    (run_dir / "validation_summary.json").write_text(json.dumps(final_payload["metrics"], indent=2) + "\n", encoding="utf-8")
    (run_dir / "training_summary.md").write_text(f"# Level A Training Summary\n\nbest_val_loss={best_loss}\nbest_epoch={best_epoch}\n", encoding="utf-8")
    print(f"TRAINING_COMPLETE {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
