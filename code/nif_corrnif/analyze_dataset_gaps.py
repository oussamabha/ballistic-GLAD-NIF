from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "hybrid_planning" / "local_numerical_inventory.json"
OUT_CSV = ROOT / "pinn_training" / "dataset_gap_analysis.csv"
OUT_MD = ROOT / "pinn_training" / "dataset_gap_analysis.md"

LEVELS = [
    {
        "training_level": "A_alpha_helical_fixed_pitch",
        "required_conditioning_variables": "x,y,z,alpha_deg",
        "required_alpha_values": [60, 65, 70, 75, 80, 85, 87, 89],
        "required_modes": ["helical"],
        "required_pitch_values": [150],
        "required_seeds": "seed0 all alphas; seeds1-2 for alpha80/85 minimum",
        "required_heights": [300],
    },
    {
        "training_level": "B_alpha_pitch_helical",
        "required_conditioning_variables": "x,y,z,alpha_deg,pitch_nm",
        "required_alpha_values": [70, 75, 80, 85, 87],
        "required_modes": ["helical"],
        "required_pitch_values": [50, 100, 150, 250],
        "required_seeds": "1 minimum; 3 preferred",
        "required_heights": [300],
    },
    {
        "training_level": "C_alpha_mode_conditioned",
        "required_conditioning_variables": "x,y,z,alpha_deg,rotation_mode_encoded,pitch_nm_optional",
        "required_alpha_values": [70, 75, 80, 85, 87],
        "required_modes": ["static_tilted", "helical", "zigzag_180"],
        "required_pitch_values": ["mode-dependent"],
        "required_seeds": "1 minimum; 3 preferred",
        "required_heights": [300],
    },
    {
        "training_level": "D_process_conditioned",
        "required_conditioning_variables": "x,y,z,alpha,mode,pitch,R_over_MFP,Ts_over_Tm,rate,distance,height,material,process_stage",
        "required_alpha_values": ["staged"],
        "required_modes": ["staged"],
        "required_pitch_values": ["staged"],
        "required_seeds": "staged",
        "required_heights": [300, 700],
    },
]


def main() -> int:
    inv = json.loads(INV.read_text(encoding="utf-8")) if INV.exists() else {"summary": {}}
    alphas_sim = set(float(x) for x in inv.get("summary", {}).get("available_alphas_with_sim", []))
    alphas_vox = set(float(x) for x in inv.get("summary", {}).get("available_alphas_with_voxel", []))
    rows = []
    for level in LEVELS:
        required = level["required_alpha_values"]
        req_nums = [float(x) for x in required if isinstance(x, (int, float))]
        existing_sim = sorted(a for a in req_nums if a in alphas_sim)
        existing_vox = sorted(a for a in req_nums if a in alphas_vox)
        missing_sim = sorted(a for a in req_nums if a not in alphas_sim)
        missing_vox = sorted(a for a in req_nums if a not in alphas_vox)
        can_train = level["training_level"].startswith("A") and not missing_vox and 60 in existing_vox and 89 in existing_vox
        if level["training_level"].startswith("A"):
            reason = "Level A cannot train now: alpha 60/65/70/89 300 nm fixed-pitch helical voxels are not present as a verified dataset."
            next_action = "Prepare and run Level A simulations only after user says RUN LEVEL A SIMULATIONS NOW."
        else:
            reason = "Plan only: required variables are absent from current training dataset."
            next_action = "Do not train; create staged simulations after Level A is validated."
        rows.append({
            **level,
            "required_alpha_values": ";".join(map(str, level["required_alpha_values"])),
            "required_modes": ";".join(map(str, level["required_modes"])),
            "required_pitch_values": ";".join(map(str, level["required_pitch_values"])),
            "required_heights": ";".join(map(str, level["required_heights"])),
            "existing_matching_simulations": ";".join(map(str, existing_sim)) or "NA",
            "existing_matching_voxels": ";".join(map(str, existing_vox)) or "NA",
            "missing_simulations": ";".join(map(str, missing_sim)) or "NA",
            "missing_voxels": ";".join(map(str, missing_vox)) or "NA",
            "can_train_now": str(bool(can_train)),
            "reason": reason,
            "next_action": next_action,
        })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    md = ["# Dataset Gap Analysis", "", "| level | can train now | missing simulations | missing voxels | next action |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['training_level']} | {r['can_train_now']} | {r['missing_simulations']} | {r['missing_voxels']} | {r['next_action']} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
