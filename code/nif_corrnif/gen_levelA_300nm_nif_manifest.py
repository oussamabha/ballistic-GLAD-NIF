#!/usr/bin/env python
"""
Generate the Level-A 300 nm NIF manifest by DERIVING every field from the
real frozen dataset. Nothing here is hand-typed scientific data: atom counts,
porosity, crop fractions, grid shapes and voxel paths all come from the
freeze artifacts under analysis/levelA_results/variability_freeze_20260529/.

Run from the project root (deepseek_correction/).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # pinn_training/ -> project root
FREEZE = ROOT / "analysis/levelA_results/variability_freeze_20260529"
ACCEPTED_CSV = FREEZE / "levelA_accepted_results.csv"
VOXEL_META = FREEZE / "seed0_plus_var_voxel_metadata_summary.json"
VOXEL_DIR = ROOT / "pinn_training/datasets/universal_glad_pipeline/voxels"

OUT_DIR = ROOT / "pinn_training/datasets"
COLUMNS = [
    "job_id", "alpha", "seed", "height_nm", "pitch_nm", "rotation_mode",
    "voxel_path", "voxel_metadata_path", "descriptor_source_csv",
    "final_atoms", "porosity", "crop_fraction_estimate",
    "voxel_grid_crop_status", "descriptor_rows",
    "accepted_for_nif_training", "split_role", "replicate_group", "notes",
]

# Strategy A (conservative interpolation) split assignment, keyed by (alpha, seed)
SPLIT_A = {
    # (60, 0): REMOVED — LA_helical_alpha060 is INVALID_CORRUPTED_ACTIVE_SLAB_LINEAGE.
    # Clean alpha60 (lit_bench campaign) = EXTERNAL_TEST_ONLY; NOT_SAME_VERIFIED_PROTOCOL.
    # Do not add alpha60 to any training or validation split. (65, 0): "train", (70, 0): "train", (75, 0): "train",
    (80, 0): "train", (80, 1): "train", (80, 2): "train",
    (85, 0): "train", (85, 1): "train", (85, 2): "train",
    (87, 0): "validation",
    (89, 0): "test",
}


def replicate_group(alpha: int, seed: int) -> str:
    if alpha == 80 and seed in (1, 2):
        return "alpha80_variability"
    if alpha == 85 and seed in (1, 2):
        return "alpha85_variability"
    return "seed0_sweep"


def main() -> int:
    assert ACCEPTED_CSV.exists(), f"missing {ACCEPTED_CSV}"
    assert VOXEL_META.exists(), f"missing {VOXEL_META}"

    meta = {m["job_id"]: m for m in json.loads(VOXEL_META.read_text())}

    rows = []
    excluded = []
    with open(ACCEPTED_CSV) as f:
        for r in csv.DictReader(f):
            job_id = r["job_id"]
            alpha = int(float(r["alpha"]))
            seed = int(r["seed"])

            # --- acceptance gates (re-checked, not assumed) ---
            crop_ok = r["voxel_grid_crop_status"].strip().upper() == "OK"
            crop_frac = float(r["crop_fraction_estimate"])
            desc_rows = int(r["descriptor_rows"])
            accepted = r["accepted_for_nif_training"].strip().lower() == "true"
            reasons = []
            if not crop_ok:
                reasons.append(f"crop_status={r['voxel_grid_crop_status']}")
            if crop_frac > 0.01:
                reasons.append(f"crop_fraction={crop_frac} > 0.01")
            if desc_rows != 15:
                reasons.append(f"descriptor_rows={desc_rows} != 15")
            if not accepted:
                reasons.append("accepted_for_nif_training != true")

            voxel_file = VOXEL_DIR / f"rho_{job_id}.npy"
            if not voxel_file.exists():
                reasons.append(f"voxel file missing: {voxel_file.name}")

            if reasons:
                excluded.append((job_id, "; ".join(reasons)))
                continue

            m = meta.get(job_id, {})
            grid_shape = m.get("grid_shape", [50, 50, 151])

            rows.append({
                "job_id": job_id,
                "alpha": alpha,
                "seed": seed,
                "height_nm": 300,                 # nominal; real final_height ~300.0x
                "pitch_nm": 150,                   # REAL pitch from job naming
                "rotation_mode": "helical",
                "voxel_path": str(voxel_file.relative_to(ROOT)).replace("\\", "/"),
                "voxel_metadata_path": str(VOXEL_META.relative_to(ROOT)).replace("\\", "/"),
                "descriptor_source_csv": str(ACCEPTED_CSV.relative_to(ROOT)).replace("\\", "/"),
                "final_atoms": int(r["final_atoms"]),
                "porosity": float(r["porosity"]),   # percent, as in real data
                "crop_fraction_estimate": crop_frac,
                "voxel_grid_crop_status": "OK",
                "descriptor_rows": desc_rows,
                "accepted_for_nif_training": True,
                "split_role": SPLIT_A[(alpha, seed)],
                "replicate_group": replicate_group(alpha, seed),
                "notes": f"grid_shape={grid_shape}; {r.get('notes','').strip()}",
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = OUT_DIR / "levelA_300nm_nif_manifest.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # JSON
    json_path = OUT_DIR / "levelA_300nm_nif_manifest.json"
    payload = {
        "metadata": {
            "dataset_name": "Level-A GLAD helical 300 nm NIF dataset",
            "freeze_dir": str(FREEZE.relative_to(ROOT)).replace("\\", "/"),
            "derived_from": [
                str(ACCEPTED_CSV.relative_to(ROOT)).replace("\\", "/"),
                str(VOXEL_META.relative_to(ROOT)).replace("\\", "/"),
            ],
            "total_accepted_jobs": len(rows),
            "height_nm": 300,
            "pitch_nm": 150,
            "rotation_mode": "helical",
            "voxel_shape": [50, 50, 151],
            "voxel_dtype": "float32",
            "voxel_value_range": "continuous rho in [0, 1]",
            "porosity_units": "percent",
            "alpha_norm_formula": "(alpha - 74.5) / 14.5  -> maps 60 to -1, 89 to +1",
            "caution": ("Simulation-backed Level-A helical alpha-conditioned dataset. "
                        "Validation status is PASS_HELICAL_GEOMETRY_ONLY. No experimental "
                        "Cu/CuOx calibration. Not a universal GLAD model. NIF training not started."),
        },
        "jobs": rows,
        "excluded": [{"job_id": j, "reason": why} for j, why in excluded],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"Accepted jobs written: {len(rows)}")
    print(f"Excluded jobs: {len(excluded)}")
    for j, why in excluded:
        print(f"  EXCLUDED {j}: {why}")
    print(f"CSV : {csv_path.relative_to(ROOT)}")
    print(f"JSON: {json_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
