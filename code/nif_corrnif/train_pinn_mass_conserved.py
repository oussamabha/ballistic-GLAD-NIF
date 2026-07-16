"""
PINN mass-conservation fine-tuning for FiLMConditionedNIF.

Gate: --allow-train (requires APPROVE_NIF_PINN_TRAINING to have been issued).

Loss:
  L_total = L_bce + lambda_mass * L_mass
  L_mass  = MSE( mean(rho_pred | alpha), F_TARGET_OCC[alpha] )

Warm-start: loads model weights from cfg["init_checkpoint"] (NIF best.pt).
Optimizer/scheduler are reset (new loss function, new landscape).

Resume: --resume --run-dir <existing_pinn_run_dir> loads that run's latest.pt.
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
    raise RuntimeError("Cannot locate project root")


PROJECT_ROOT = find_project_root()
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
VOXEL_DESC_ROOT = PROJECT_ROOT / "03_VOXEL_DESCRIPTOR_ANALYSIS"
GLAD_ROOT = PROJECT_ROOT / "01_GLAD_SIMULATION"

PINN_SRC = PINN_NIF_ROOT / "src"
if str(PINN_SRC) not in sys.path:
    sys.path.insert(0, str(PINN_SRC))

from mass_conservation_loss import mass_conservation_loss, per_alpha_mass_errors  # noqa: E402

MODEL_PATH = PINN_NIF_ROOT / "pinn_training" / "models" / "nif_levelA_helical.py"


def resolve_project_path(value: str | Path, default_root: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    text = str(value).replace("\\", "/")
    prefix_map = {
        "pinn_training": PINN_NIF_ROOT / "pinn_training",
        "analysis": VOXEL_DESC_ROOT / "analysis",
        "simulation_batch": GLAD_ROOT / "simulation_batch",
    }
    for prefix, base in sorted(prefix_map.items(), key=lambda item: len(item[0]), reverse=True):
        if text == prefix:
            return base
        if text.startswith(prefix + "/"):
            return base.joinpath(*[part for part in text[len(prefix) + 1:].split("/") if part])
    return default_root.joinpath(*[part for part in text.split("/") if part])


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
    out_root = resolve_project_path(cfg.get("output_root", "pinn_training/runs"), PINN_NIF_ROOT)
    return out_root / f"pinn_mass_conserved_{stamp}"


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
    bce_losses: list[float] = []
    correct = total = tp = fp = fn = 0
    bce = torch.nn.BCELoss(reduction="mean")
    all_pred = []
    all_alpha = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size].to(device)
            yb = y[s:s + batch_size].to(device).clamp(0, 1)
            pred = model(xb)
            bce_losses.append(float(bce(pred, yb).item()))
            pb = pred >= 0.5
            gb = yb >= 0.5
            correct += int((pb == gb).sum().item())
            total += int(len(yb))
            tp += int((pb & gb).sum().item())
            fp += int((pb & ~gb).sum().item())
            fn += int((~pb & gb).sum().item())
            all_pred.append(pred.cpu())
            all_alpha.append(xb[:, 3].cpu())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    val_bce = float(np.mean(bce_losses))

    rho_all = torch.cat(all_pred)
    alpha_all = torch.cat(all_alpha)
    mass_errs = per_alpha_mass_errors(rho_all, alpha_all)
    val_mass = float(np.mean(list(mass_errs.values()))) if mass_errs else 0.0

    return {
        "val_bce": val_bce,
        "val_mass": val_mass,
        "occupancy_accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "per_alpha_mass_errors": mass_errs,
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
    parser = argparse.ArgumentParser(
        description="PINN mass-conservation fine-tuning for FiLMConditionedNIF. "
                    "Requires APPROVE_NIF_PINN_TRAINING gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing PINN run (loads its latest.pt).")
    parser.add_argument("--run-dir", default=None,
                        help="Existing PINN run dir for --resume.")
    parser.add_argument("--allow-train", action="store_true",
                        help="Required to actually train (gate: APPROVE_NIF_PINN_TRAINING).")
    args = parser.parse_args()

    if not args.allow_train:
        print("TRAINING_NOT_STARTED: pass --allow-train only after APPROVE_NIF_PINN_TRAINING issued")
        return 0

    cfg_path = resolve_project_path(args.config, PROJECT_ROOT).resolve()
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = load_config(cfg_path)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu"
    )

    run_dir = make_run_dir(cfg, args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config_used.yaml")

    dataset = resolve_project_path(cfg["dataset_dir"], PINN_NIF_ROOT)
    x_train, y_train = load_npz(dataset / "train_samples.npz")
    x_val, y_val = load_npz(dataset / "val_samples.npz")

    mod = load_model_module()
    model = mod.build_model(cfg).to(device)

    lambda_mass = float(cfg.get("lambda_mass", 10.0))
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    epochs = int(cfg["epochs"])
    batch = int(cfg["batch_size"])
    patience = int(cfg.get("early_stopping_patience", 40))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 1
    best_iou = 0.0
    best_epoch = 0
    stale = 0
    resume_status = "COLD_START"

    if args.resume and args.run_dir:
        lp = latest_checkpoint(ckpt_dir)
        if lp:
            ckpt = torch.load(lp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            sched.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_iou = float(ckpt.get("best_iou", 0.0))
            best_epoch = int(ckpt.get("best_epoch", 0))
            stale = int(ckpt.get("stale", 0))
            resume_status = f"RESUMED {lp}"
        else:
            resume_status = "RESUME_REQUESTED_BUT_NO_CHECKPOINT"
    else:
        # Warm-start: load model weights from NIF init_checkpoint
        init_ckpt_path = resolve_project_path(cfg["init_checkpoint"], PINN_NIF_ROOT)
        if init_ckpt_path.exists():
            init_ckpt = torch.load(init_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(init_ckpt["model_state_dict"])
            resume_status = f"WARM_START {init_ckpt_path}"
        else:
            resume_status = f"INIT_CKPT_NOT_FOUND: {init_ckpt_path}"
            print(f"WARNING: {resume_status} — starting from scratch")

    (run_dir / "resume_status.txt").write_text(
        f"{resume_status}\nstart_epoch={start_epoch}\nbest_iou={best_iou}\nbest_epoch={best_epoch}\n",
        encoding="utf-8",
    )

    log_csv = run_dir / "training_log.csv"
    new_log = not log_csv.exists()
    csv_fields = ["epoch", "train_loss", "bce_loss", "mass_loss", "val_bce", "val_mass", "val_iou", "val_recall", "lr"]
    bce_fn = torch.nn.BCELoss()

    with log_csv.open("a", encoding="utf-8", newline="") as fcsv, \
         (run_dir / "training_log.jsonl").open("a", encoding="utf-8") as fj:

        writer = csv.DictWriter(fcsv, fieldnames=csv_fields)
        if new_log:
            writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            perm = torch.randperm(len(x_train))
            bce_losses: list[float] = []
            mass_losses: list[float] = []
            total_losses: list[float] = []

            for s in range(0, len(perm), batch):
                idx = perm[s:s + batch]
                xb = x_train[idx].to(device)
                yb = y_train[idx].to(device).clamp(0, 1)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                l_bce = bce_fn(pred, yb)
                l_mass = mass_conservation_loss(pred, xb[:, 3])
                loss = l_bce + lambda_mass * l_mass
                loss.backward()
                opt.step()
                bce_losses.append(float(l_bce.item()))
                mass_losses.append(float(l_mass.item()))
                total_losses.append(float(loss.item()))

            sched.step()

            metrics = evaluate(model, x_val, y_val, batch, device)
            train_bce = float(np.mean(bce_losses))
            train_mass = float(np.mean(mass_losses))
            train_total = float(np.mean(total_losses))

            row = {
                "epoch": epoch,
                "train_loss": round(train_total, 6),
                "bce_loss": round(train_bce, 6),
                "mass_loss": round(train_mass, 6),
                "val_bce": round(metrics["val_bce"], 6),
                "val_mass": round(metrics["val_mass"], 6),
                "val_iou": round(metrics["iou"], 6),
                "val_recall": round(metrics["recall"], 6),
                "lr": opt.param_groups[0]["lr"],
            }
            writer.writerow(row)
            fcsv.flush()
            fj.write(json.dumps(row) + "\n")
            fj.flush()

            normalization = {}
            norm_path = dataset / "normalization.json"
            if norm_path.exists():
                normalization = json.loads(norm_path.read_text(encoding="utf-8"))

            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch": epoch,
                "best_iou": max(best_iou, metrics["iou"]),
                "best_epoch": best_epoch,
                "stale": stale,
                "config": cfg,
                "config_hash": sha256_text(cfg_text),
                "seed": seed,
                "lambda_mass": lambda_mass,
                "normalization": normalization,
                "metrics": metrics,
            }

            save_checkpoint(ckpt_dir / "latest.pt", payload)

            if epoch % int(cfg.get("save_every", 10)) == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", payload)

            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_epoch = epoch
                stale = 0
                payload["best_iou"] = best_iou
                payload["best_epoch"] = best_epoch
                save_checkpoint(ckpt_dir / "best.pt", payload)
            else:
                stale += 1

            if patience and stale >= patience:
                print(f"EARLY_STOP epoch={epoch} stale={stale} best_iou={best_iou:.4f}")
                break

    final_payload = torch.load(ckpt_dir / "latest.pt", map_location=device, weights_only=False)
    save_checkpoint(ckpt_dir / "final.pt", final_payload)

    summary = {
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "lambda_mass": lambda_mass,
        "f_target_source": cfg.get("f_target_source", "gt_occ_frac"),
        "final_metrics": final_payload["metrics"],
    }
    for result_path in [run_dir / "validation_summary.json", run_dir / "training_summary.md"]:
        if result_path.exists():
            backup = result_path.with_name(
                f"{result_path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{result_path.suffix}"
            )
            shutil.copy2(result_path, backup)

    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "training_summary.md").write_text(
        f"# PINN Mass-Conserved Training Summary\n\n"
        f"best_val_iou={best_iou}\nbest_epoch={best_epoch}\nlambda_mass={lambda_mass}\n",
        encoding="utf-8",
    )
    print(f"TRAINING_COMPLETE {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
