"""Generate one YAML config per dense alpha for full-grid z-block NIF training."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "00_PHD_KNOWLEDGE_HUB").exists():
        PROJECT_ROOT = p; break

PINN      = PROJECT_ROOT / "02_PINN_NIF"
CFG_DIR   = PINN / "pinn_training/configs"
DS_BASE   = PINN / "pinn_training/datasets/nif_fullgrid_dense"
ALPHAS    = [65, 70, 75, 80]

for a in ALPHAS:
    ds_dir = DS_BASE / f"alpha_{a:03d}"
    meta_path = ds_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  α={a}: dataset not found — run build_nif_fullgrid_dense_datasets.py first")
        continue
    meta = json.loads(meta_path.read_text())
    occ_train = meta["occ_frac_train"]

    cfg = {
        "run_name":    f"nif_fullgrid_dense_{a:03d}",
        "dataset_dir": f"pinn_training/datasets/nif_fullgrid_dense/alpha_{a:03d}",
        "output_root": "pinn_training/runs",
        "architecture": "FiLMConditionedNIF",
        "hidden":   256,
        "n_layers": 4,
        "n_freq":   32,
        "loss":     "weighted_bce",   # pos_weight = (1-occ_train)/occ_train from normalization.json
        "epochs":   300,
        "batch_size":  32768,
        "learning_rate": 0.0005,
        "weight_decay":  0.0,
        "seed": 42,
        "save_every": 20,
        "eval_every": 1,
        "early_stopping_patience": 80,
        "device": "auto",
        "n_threads": 8,
        "scope_label": f"fullgrid_dense_zblock_weighted_bce_alpha{a}",
        "permission_required": "APPROVE_FULLGRID_DENSE_ALPHA_TRAINING",
        "_comment": (
            f"Full-grid z-block training for α={a}° (occ_train={occ_train:.4f}). "
            f"Per-alpha balanced 80k failed: trivial collapse (pred_spread≈0.10). "
            f"Now uses all {meta['train_samples']:,} train voxels (z[0:{meta['k_split']}]), "
            f"spatial holdout z[{meta['k_split']}:{meta['voxel_shape'][2]}] ({meta['test_holdout_samples']:,} voxels). "
            f"pos_weight={meta['pos_weight']:.4f} balances gradient without subsampling."
        ),
    }
    out = CFG_DIR / f"nif_fullgrid_dense_{a:03d}_config.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"  wrote {out.name}  (occ_train={occ_train:.4f}  pw={meta['pos_weight']:.3f}  "
          f"train={meta['train_samples']:,}  test={meta['test_holdout_samples']:,})")

print(f"\nDone — {len(ALPHAS)} configs in {CFG_DIR.relative_to(PROJECT_ROOT)}")
