"""Generate one YAML config per alpha for per-alpha NIF training."""
from pathlib import Path
import yaml, json

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p/"00_PHD_KNOWLEDGE_HUB").exists():
        PROJECT_ROOT = p; break

PINN = PROJECT_ROOT / "02_PINN_NIF"
CFG_DIR = PINN / "pinn_training/configs"
ALPHAS  = [65, 70, 75, 80, 85, 87, 89]

for a in ALPHAS:
    ds_dir = f"pinn_training/datasets/nif_per_alpha/alpha_{a:03d}"
    cfg = {
        "run_name":   f"nif_peralpha_{a:03d}",
        "dataset_dir": ds_dir,
        "output_root": "pinn_training/runs",
        "architecture": "FiLMConditionedNIF",
        "hidden":   256,
        "n_layers": 4,
        "n_freq":   32,
        "loss":     "bce",              # standard BCE; train set is 50/50 balanced → no pos_weight needed
        "epochs":   300,
        "batch_size": 32768,
        "learning_rate": 0.0005,
        "weight_decay":  0.0,
        "seed": 42,
        "save_every": 20,
        "eval_every": 1,
        "early_stopping_patience": 80,
        "device": "auto",
        "n_threads": 8,
        "scope_label": f"per_alpha_single_network_balanced_bce_alpha{a}",
        "permission_required": "APPROVE_PER_ALPHA_SEPARATE_TRAINING",
        "_comment": (
            f"Single network for alpha={a}deg only. No FiLM competition. "
            f"50/50 balanced batches → BCE gradient is balanced per voxel. "
            f"Overfit test (2026-06-15) showed IoU=0.737 for alpha=65 under similar conditions."
        ),
    }
    out = CFG_DIR / f"nif_peralpha_{a:03d}_config.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"  wrote {out.name}")

print(f"Done — {len(ALPHAS)} configs in {CFG_DIR.relative_to(PROJECT_ROOT)}")
