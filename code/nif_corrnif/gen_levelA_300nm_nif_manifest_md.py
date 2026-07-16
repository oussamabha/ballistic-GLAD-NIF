#!/usr/bin/env python
"""
Emit the human-readable manifest .md from the generated CSV/JSON, so the three
manifest formats stay in sync. Run after gen_levelA_300nm_nif_manifest.py.
Reads only generated artifacts + derives a couple of summary stats.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "pinn_training/datasets"
CSV = DS / "levelA_300nm_nif_manifest.csv"
JSON = DS / "levelA_300nm_nif_manifest.json"
OUT = DS / "levelA_300nm_nif_manifest.md"


def main() -> int:
    meta = json.loads(JSON.read_text())["metadata"]
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    lines = []
    lines.append("# Level-A GLAD Helical 300 nm NIF Dataset Manifest")
    lines.append("")
    lines.append(f"- Total accepted jobs: **{len(rows)}**")
    lines.append(f"- Height: {meta['height_nm']} nm  |  Pitch: **{meta['pitch_nm']} nm**  |  "
                 f"Rotation: {meta['rotation_mode']}")
    lines.append(f"- Voxel: shape {meta['voxel_shape']}, dtype {meta['voxel_dtype']}, "
                 f"{meta['voxel_value_range']}")
    lines.append(f"- Porosity units: **{meta['porosity_units']}**")
    lines.append(f"- Alpha norm: `{meta['alpha_norm_formula']}`")
    lines.append(f"- Derived from: {', '.join('`'+p+'`' for p in meta['derived_from'])}")
    lines.append("")
    lines.append(f"> {meta['caution']}")
    lines.append("")
    lines.append("## Jobs")
    lines.append("")
    lines.append("| job_id | alpha | seed | final_atoms | porosity % | crop_frac | "
                 "rows | split | replicate_group |")
    lines.append("|--------|-------|------|-------------|-----------|-----------|------|-------|-----------------|")
    for r in rows:
        lines.append(
            f"| {r['job_id']} | {r['alpha']} | {r['seed']} | {int(r['final_atoms']):,} | "
            f"{r['porosity']} | {float(r['crop_fraction_estimate']):.2e} | "
            f"{r['descriptor_rows']} | {r['split_role']} | {r['replicate_group']} |"
        )
    lines.append("")
    lines.append("## Acceptance gates (re-checked at generation time)")
    lines.append("")
    lines.append("- `voxel_grid_crop_status == OK`")
    lines.append("- `crop_fraction_estimate <= 0.01`")
    lines.append("- `descriptor_rows == 15`")
    lines.append("- `accepted_for_nif_training == true`")
    lines.append("- voxel `.npy` file present on disk")
    lines.append("")
    excl = json.loads(JSON.read_text()).get("excluded", [])
    lines.append(f"Excluded jobs: **{len(excl)}**"
                 + ("" if not excl else " — " + "; ".join(f"{e['job_id']} ({e['reason']})" for e in excl)))
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(rows)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
