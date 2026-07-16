"""
Tier-2 Fix B: continuous-density regression reframing.

Adapted from train_nif_hardlabel_v2.py's structure (same run-dir/checkpoint/
resume/logging conventions and permission-gate pattern), but changes the loss
from BCE-on-a-continuous-target to a direct regression loss (MSE or L1)
against the continuous max-normalized density field, and points at the
CONTINUOUS ("soft-label") dataset rather than the binary-thresholded one.

Motivation (see AGENTS.md "P1/P2 STATUS SYNC", P2 tier 2, and
P2_density_calibrated_NIF_GLAD.tex Experiment 2, "soft-label equilibrium
trap"): the existing soft-label experiment applies BCE to a continuous [0,1]
density target as if it were a Bernoulli probability. BCE's gradient w.r.t.
a continuous target still pulls toward whatever value minimizes expected
cross-entropy given the *distribution* of targets at each input, which for a
9x/98%-in-the-"soft"-zone target distribution (per the manuscript's own
f_soft diagnostic) is dominated by matching the marginal mean, not the
spatial gradient. This is a loss-function/target mismatch, not necessarily a
capacity or architecture problem — MSE/L1 have no such probabilistic
target assumption and directly penalize deviation from the continuous field,
which could let the same architecture (FiLMConditionedNIF, UNCHANGED here)
recover more of the learnable spatial gradient in the continuous density
field.

Architecture: FiLMConditionedNIF, unchanged — this script isolates the loss
function as the only new variable, in the same spirit as the tier-1
seed-isolation diagnostic (change exactly one thing vs. an existing run).
Sigmoid output (already in [0,1]) is kept — the task note is correct that
sigmoid is fine paired with MSE/L1, it is only the BCE pairing that assumes a
Bernoulli target.

Gate: APPROVE_CONTINUOUS_REGRESSION_NIF_TRAINING
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import shutil
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
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
MODEL_PATH    = PINN_NIF_ROOT / "pinn_training" / "models" / "nif_levelA_helical.py"

KNOWN_GATES = {"APPROVE_CONTINUOUS_REGRESSION_NIF_TRAINING"}


def resolve_project_path(value: str | Path, default_root: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    text = str(value).replace("\\", "/")
    prefix_map = {"pinn_training": PINN_NIF_ROOT / "pinn_training"}
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
            / f"{cfg.get('run_name', 'nif_continuous_regression')}_{stamp}")


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    numbered = sorted(ckpt_dir.glob("epoch_*.pt"))
    return numbered[-1] if numbered else None


def load_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.load(path)["data"].astype(np.float32)
    # Label is always the last column (continuous density here, NOT
    # thresholded); features are all preceding columns (x, y, z, alpha_norm).
    return torch.tensor(arr[:, :-1]), torch.tensor(arr[:, -1])


def build_regression_loss_fn(cfg: dict):
    loss_type = cfg.get("loss", "mse")
    if loss_type == "mse":
        print("Loss: mse (continuous regression)")
        mse = torch.nn.MSELoss()
        return lambda pred, tgt: mse(pred, tgt)
    if loss_type == "l1":
        print("Loss: l1 (continuous regression)")
        l1 = torch.nn.L1Loss()
        return lambda pred, tgt: l1(pred, tgt)
    raise ValueError(f"Unsupported loss for continuous regression: {loss_type!r} (expected 'mse' or 'l1')")


def evaluate(model, x, y, batch_size, device, threshold: float) -> dict:
    """Regression metrics (mse/mae/r2) PLUS threshold-derived occupancy metrics
    (precision/recall/iou at `threshold`) so results stay comparable to the
    hard-label baseline's Table tab:hard_iou methodology."""
    model.eval()
    sq_errs, abs_errs = [], []
    tp = fp = fn = correct = total = 0
    sum_y = 0.0
    sum_y2 = 0.0
    sum_sq_resid = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size].to(device)
            yb = y[s:s + batch_size].to(device).clamp(0.0, 1.0)
            pred = model(xb)

            diff = pred - yb
            sq_errs.append(float((diff ** 2).mean().item()))
            abs_errs.append(float(diff.abs().mean().item()))

            sum_sq_resid += float((diff ** 2).sum().item())
            sum_y += float(yb.sum().item())
            sum_y2 += float((yb ** 2).sum().item())
            n += int(len(yb))

            pb = pred >= threshold
            gb = yb >= threshold
            correct += int((pb == gb).sum().item())
            total   += int(len(yb))
            tp += int((pb & gb).sum().item())
            fp += int((pb & ~gb).sum().item())
            fn += int((~pb & gb).sum().item())

    mean_y = sum_y / max(n, 1)
    ss_tot = sum_y2 - n * mean_y ** 2
    r2 = 1.0 - (sum_sq_resid / max(ss_tot, 1e-12)) if ss_tot > 1e-12 else float("nan")

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    iou       = tp / max(tp + fp + fn, 1)
    return {
        "mse": float(np.mean(sq_errs)),
        "mae": float(np.mean(abs_errs)),
        "r2": r2,
        "threshold": threshold,
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
    parser = argparse.ArgumentParser(description="Train NIF with continuous-density regression loss (MSE/L1). Permission-gated.")
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

    perm_required = cfg.get("permission_required", "")
    if perm_required and perm_required not in KNOWN_GATES:
        print(f"GATE MISMATCH: config requires '{perm_required}', "
              f"script knows: {sorted(KNOWN_GATES)}")
        return 1

    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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

    dataset = resolve_project_path(cfg["dataset_dir"], PINN_NIF_ROOT)
    x_train, y_train = load_npz(dataset / "train_samples.npz")
    x_val,   y_val   = load_npz(dataset / "val_samples.npz")

    print(f"Dataset: {dataset.relative_to(PROJECT_ROOT)}")
    print(f"  train={len(x_train):,}  val={len(x_val):,}")
    print(f"  target mean (train): {y_train.mean():.4f}  target std: {y_train.std():.4f}")
    frac_soft = float(((y_train > 0.1) & (y_train < 0.9)).float().mean().item())
    print(f"  fraction of train targets in soft zone (0.1,0.9): {frac_soft:.4f}")

    mod   = load_model_module()
    model = mod.build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.get('architecture')}  params={n_params:,}  device={device}")

    loss_fn = build_regression_loss_fn(cfg)
    threshold = float(cfg.get("eval_threshold", 0.5))

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
        writer = csv.DictWriter(fcsv, fieldnames=["epoch", "train_loss", "val_mse", "val_mae",
                                                    "val_r2", "val_iou", "val_precision", "val_recall", "lr"])
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

            metrics    = evaluate(model, x_val, y_val, batch, device, threshold)
            train_loss = float(np.mean(losses))
            row = {
                "epoch": epoch, "train_loss": train_loss,
                "val_mse": metrics["mse"], "val_mae": metrics["mae"], "val_r2": metrics["r2"],
                "val_iou": metrics["iou"], "val_precision": metrics["precision"], "val_recall": metrics["recall"],
                "lr": opt.param_groups[0]["lr"],
            }
            writer.writerow(row); fcsv.flush()
            fj.write(json.dumps(row) + "\n"); fj.flush()

            print(f"  ep={epoch:>4}  train_loss={train_loss:.4f}  val_mse={metrics['mse']:.4f}"
                  f"  val_r2={metrics['r2']:.4f}  iou@{threshold}={metrics['iou']:.4f}"
                  f"  prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}")

            val_loss_for_selection = metrics["mse"]
            payload = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch":     epoch,
                "best_metric": min(best_loss, val_loss_for_selection),
                "best_epoch":  best_epoch,
                "config":    cfg,
                "config_hash": sha256_text(cfg_text),
                "seed":      seed,
                "normalization": (json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
                                  if (dataset / "normalization.json").exists() else {}),
                "metrics":   metrics,
            }
            save_checkpoint(ckpt_dir / "latest.pt", payload)

            if val_loss_for_selection < best_loss:
                best_loss = val_loss_for_selection; best_epoch = epoch; stale = 0
                payload["best_metric"] = best_loss; payload["best_epoch"] = best_epoch
                save_checkpoint(ckpt_dir / "best.pt", payload)
                print(f"  ** NEW BEST  epoch={epoch}  val_mse={best_loss:.6f}  "
                      f"r2={metrics['r2']:.4f}  iou={metrics['iou']:.4f}")
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
        f"# NIF Continuous Regression Training Summary\n\n"
        f"loss={cfg.get('loss','mse')}\n"
        f"best_val_mse={best_loss:.6f}  best_epoch={best_epoch}\n"
        f"best_r2={best_metrics['r2']:.4f}  best_iou@{threshold}={best_metrics['iou']:.4f}  "
        f"precision={best_metrics['precision']:.4f}  recall={best_metrics['recall']:.4f}\n",
        encoding="utf-8",
    )

    print(f"\nTRAINING_COMPLETE {run_dir}")
    print(f"  best_epoch={best_epoch}  val_mse={best_loss:.6f}")
    print(f"  r2={best_metrics['r2']:.4f}  iou={best_metrics['iou']:.4f}  "
          f"precision={best_metrics['precision']:.4f}  recall={best_metrics['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
