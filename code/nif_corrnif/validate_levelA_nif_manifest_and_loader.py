#!/usr/bin/env python
"""
Validate the Level-A 300 nm NIF manifest and loader against the REAL voxels.
Writes a markdown + json report under pinn_training/reports/.
ASCII-only output for cross-platform terminals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from levelA_nif_dataset import (
    GRID_SHAPE, PINN_NIF_ROOT, PROJECT_ROOT, LevelANIFManifest, LevelANIFDataset,
    normalize_alpha, resolve_project_path, sample_points_from_voxel,
)

EXPECTED_SPLITS = {
    "train": 10, "validation": 1, "test": 1,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pinn_training/datasets/levelA_300nm_nif_manifest.csv")
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    base = resolve_project_path(args.base_dir, PROJECT_ROOT)
    checks = []  # (name, ok, detail)

    def rec(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    man = LevelANIFManifest(args.manifest, args.base_dir)

    # 1 row count
    rec("1 manifest row count == 12", len(man) == 12, f"got {len(man)}")
    # 2 all accepted
    rec("2 all accepted_for_nif_training", all(j["accepted_for_nif_training"] for j in man.jobs))
    # 3 voxel files exist
    miss_v = [j["job_id"] for j in man.jobs if not resolve_project_path(j["voxel_path"], base).exists()]
    rec("3 voxel files exist", not miss_v, f"missing={miss_v}")
    # 4 metadata file exists
    miss_m = [j["job_id"] for j in man.jobs if not resolve_project_path(j["voxel_metadata_path"], base).exists()]
    rec("4 voxel metadata exists", not miss_m, f"missing={miss_m}")

    # load voxels for shape/finite/range checks
    bad_shape, non_finite, ranges = [], [], []
    for j in man.jobs:
        arr = np.load(resolve_project_path(j["voxel_path"], base), allow_pickle=False)
        if arr.shape != GRID_SHAPE:
            bad_shape.append((j["job_id"], arr.shape))
        if not np.all(np.isfinite(arr)):
            non_finite.append(j["job_id"])
        ranges.append((float(arr.min()), float(arr.max())))
    # 5 shape
    rec("5 voxel shape == (50,50,151)", not bad_shape, f"bad={bad_shape}")
    # 6 finite
    rec("6 all finite (no NaN/Inf)", not non_finite, f"bad={non_finite}")
    # 7 density range within [0,1]
    rmin = min(r[0] for r in ranges); rmax = max(r[1] for r in ranges)
    rec("7 rho range within [0,1]", rmin >= -1e-6 and rmax <= 1 + 1e-6,
        f"global min={rmin}, max={rmax}")

    # 8 alpha normalization in [-1,1] and endpoints correct
    an = {j["alpha"]: normalize_alpha(j["alpha"]) for j in man.jobs}
    ok_alpha = all(-1 - 1e-9 <= v <= 1 + 1e-9 for v in an.values())
    rec("8 alpha_norm in [-1,1]", ok_alpha,
        f"a60={an.get(60):.3f} a89={an.get(89):.3f}")

    # 9 coordinate normalization in (-1,1)
    arr0 = np.load(resolve_project_path(man.jobs[0]["voxel_path"], base), allow_pickle=False)
    pts, rho = sample_points_from_voxel(arr0, 200, 200, 0, seed=0)
    coord_ok = bool(np.all(pts > -1) and np.all(pts < 1))
    rho_ok = bool(np.all(rho >= -1e-6) and np.all(rho <= 1 + 1e-6))
    rec("9 coord norm in (-1,1)", coord_ok,
        f"min={pts.min():.3f} max={pts.max():.3f}")

    # 10 balanced sampler smoke (counts honored)
    pts2, rho2 = sample_points_from_voxel(arr0, 100, 100, 50, seed=1)
    rec("10 balanced sampler smoke", pts2.shape == (250, 3) and rho2.shape == (250,),
        f"pts={pts2.shape} rho={rho2.shape}, target_rho_ok={rho_ok}")

    # 11 dataset batch shape via torch
    ds = LevelANIFDataset(args.manifest, base_dir=args.base_dir, split_role="train",
                          num_occupied=64, num_empty=64, seed=42)
    from torch.utils.data import DataLoader
    xb, yb = next(iter(DataLoader(ds, batch_size=32, shuffle=False)))
    rec("11 batch shapes [32,4]/[32,1]",
        tuple(xb.shape) == (32, 4) and tuple(yb.shape) == (32, 1),
        f"x={tuple(xb.shape)} y={tuple(yb.shape)}")

    # 12 split strategy consistency
    counts = {k: len(man.filter_by_split(k)) for k in EXPECTED_SPLITS}
    rec("12 split counts (train10/val1/test1)", counts == EXPECTED_SPLITS, f"got {counts}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\nSUMMARY: {passed}/{total} passed")

    report = {
        "manifest": args.manifest,
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "global_rho_min": rmin,
        "global_rho_max": rmax,
        "split_counts": counts,
    }
    rdir = PINN_NIF_ROOT / "pinn_training" / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "levelA_nif_manifest_validation_report.json").write_text(json.dumps(report, indent=2))

    md = [f"# Level-A 300 nm NIF Manifest + Loader Validation",
          "", f"Result: **{passed}/{total} checks passed**",
          f"(manifest: `{args.manifest}`)", "", "| # | Check | Result | Detail |",
          "|---|-------|--------|--------|"]
    for n, ok, d in checks:
        md.append(f"| | {n} | {'PASS' if ok else 'FAIL'} | {d} |")
    md += ["", f"- global rho range: [{rmin}, {rmax}]", f"- split counts: {counts}"]
    (rdir / "levelA_nif_manifest_validation_report.md").write_text("\n".join(md))

    print(f"Report: pinn_training/reports/levelA_nif_manifest_validation_report.(md|json)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
