"""Generate configs for PhaseOnlyFiLMNIF — reuses nif_phase_conditioned datasets."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "00_PHD_KNOWLEDGE_HUB").exists():
        PROJECT_ROOT = p; break

PINN    = PROJECT_ROOT / "02_PINN_NIF"
CFG_DIR = PINN / "pinn_training/configs"
DS_BASE = PINN / "pinn_training/datasets/nif_phase_conditioned"  # SAME datasets
ALPHAS  = [65, 70, 75, 80, 85, 87, 89]

for a in ALPHAS:
    meta = json.loads((DS_BASE / f"alpha_{a:03d}" / "metadata.json").read_text())
    cfg = {
        "run_name":    f"nif_phaseonly_{a:03d}",
        "dataset_dir": f"pinn_training/datasets/nif_phase_conditioned/alpha_{a:03d}",
        "output_root": "pinn_training/runs",
        "architecture": "PhaseOnlyFiLMNIF",   # ← key change: z excluded from Fourier
        "hidden": 256, "n_layers": 4, "n_freq": 32,
        "loss": "weighted_bce",
        "epochs": 300, "batch_size": 32768, "learning_rate": 0.0005,
        "weight_decay": 0.0, "seed": 42, "save_every": 20, "eval_every": 1,
        "early_stopping_patience": 80, "device": "auto", "n_threads": 8,
        "scope_label": f"phaseonly_zblock_wbce_alpha{a}",
        "permission_required": "APPROVE_PHASEONLY_NIF_TRAINING",
        "_comment": (
            f"PhaseOnly α={a}°: z excluded from 4D Fourier encoder (x,y,φ_sin,φ_cos). "
            f"z_norm appended raw after encoding. Fixes z-shortcut that prevented "
            f"holdout generalization in PhaseConditionedFiLMNIF. "
            f"occ_train={meta['occ_frac_train']:.4f}  pw={meta['pos_weight']:.3f}"
        ),
    }
    out = CFG_DIR / f"nif_phaseonly_{a:03d}_config.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"  wrote {out.name}  (pw={meta['pos_weight']:.3f}  train={meta['train_samples']:,})")

print(f"\nDone — {len(ALPHAS)} configs in {CFG_DIR.relative_to(PROJECT_ROOT)}")
