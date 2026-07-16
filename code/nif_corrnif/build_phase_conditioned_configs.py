"""Generate 7 YAML configs for PhaseConditionedFiLMNIF — one per alpha."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "00_PHD_KNOWLEDGE_HUB").exists():
        PROJECT_ROOT = p; break

PINN    = PROJECT_ROOT / "02_PINN_NIF"
CFG_DIR = PINN / "pinn_training/configs"
DS_BASE = PINN / "pinn_training/datasets/nif_phase_conditioned"
ALPHAS  = [65, 70, 75, 80, 85, 87, 89]

for a in ALPHAS:
    ds_dir = DS_BASE / f"alpha_{a:03d}"
    meta_path = ds_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  α={a}: dataset not found — run build_nif_phase_conditioned_datasets.py first")
        continue
    meta = json.loads(meta_path.read_text())
    occ_train = meta["occ_frac_train"]

    cfg = {
        "run_name":    f"nif_phasecond_{a:03d}",
        "dataset_dir": f"pinn_training/datasets/nif_phase_conditioned/alpha_{a:03d}",
        "output_root": "pinn_training/runs",
        "architecture": "PhaseConditionedFiLMNIF",   # 5D Fourier (xyz + phi_sin,phi_cos) + FiLM alpha
        "hidden":    256,
        "n_layers":  4,
        "n_freq":    32,
        "loss":      "weighted_bce",    # pos_weight from normalization.json → global_occ_frac_train
        "epochs":    300,
        "batch_size":   32768,
        "learning_rate": 0.0005,
        "weight_decay":  0.0,
        "seed": 42,
        "save_every": 20,
        "eval_every": 1,
        "early_stopping_patience": 80,
        "device":    "auto",
        "n_threads": 8,
        "scope_label": f"phase_conditioned_zblock_wbce_alpha{a}",
        "permission_required": "APPROVE_PHASE_CONDITIONED_NIF_TRAINING",
        "_comment": (
            f"Phase-conditioned NIF for α={a}°. Adds φ_sin=sin(2π·k/75.5), φ_cos=cos(2π·k/75.5) "
            f"to spatial encoding, solving the helical rotation z-generalization barrier. "
            f"occ_train={occ_train:.4f}  pw={meta['pos_weight']:.3f}  "
            f"train={meta['train_samples']:,}  holdout={meta['test_holdout_samples']:,}"
        ),
    }
    out = CFG_DIR / f"nif_phasecond_{a:03d}_config.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"  wrote {out.name}  "
          f"(occ_train={occ_train:.4f}  pw={meta['pos_weight']:.3f}  "
          f"train={meta['train_samples']:,})")

print(f"\nDone — {len(ALPHAS)} configs in {CFG_DIR.relative_to(PROJECT_ROOT)}")
