#!/usr/bin/env python
"""
Generate split strategies (A/B/C/D) for the Level-A 300 nm NIF dataset,
deriving real job_ids from the manifest so nothing is hand-typed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "pinn_training/datasets/levelA_300nm_nif_manifest.csv"
OUT = ROOT / "pinn_training/datasets"


def load_jobs():
    with open(MAN, newline="") as f:
        return [(r["job_id"], int(float(r["alpha"])), int(r["seed"])) for r in csv.DictReader(f)]


def main() -> int:
    jobs = load_jobs()
    by_id = {jid: (a, s) for jid, a, s in jobs}
    ids = [jid for jid, _, _ in jobs]
    alphas = sorted({a for _, a, _ in jobs})

    def with_alpha(a):
        return [jid for jid, aa, _ in jobs if aa == a]

    def seed0():
        return [jid for jid, _, s in jobs if s == 0]

    # Strategy A: conservative interpolation
    A = {
        "name": "Conservative interpolation",
        "train": [j for j in ids if by_id[j][0] <= 85],
        "validation": with_alpha(87),
        "test": with_alpha(89),
        "purpose": "Test high-alpha generalization (val=87, test=89 held out of train).",
    }
    # Strategy B: leave-one-alpha-out
    B = {
        "name": "Leave-one-alpha-out",
        "purpose": "Evaluate interpolation across alpha; one alpha held out per fold.",
        "folds": [
            {"fold": i + 1, "held_out_alpha": a,
             "train": [j for j in ids if by_id[j][0] != a],
             "validation": with_alpha(a)}
            for i, a in enumerate(alphas)
        ],
    }
    # Strategy C: seed reproducibility
    C = {
        "name": "Seed reproducibility",
        "purpose": "Train on seed0 sweep; validate on stochastic replicas.",
        "train": seed0(),
        "validation_replicate_alpha80": [j for j in ids if by_id[j] in [(80, 1), (80, 2)]],
        "validation_replicate_alpha85": [j for j in ids if by_id[j] in [(85, 1), (85, 2)]],
    }
    # Strategy D: all-data pretraining
    D = {
        "name": "All-data pretraining",
        "purpose": "Final training on all accepted voxels AFTER A/B/C are evaluated.",
        "train": ids, "validation": [], "test": [],
        "caution": "Do not use until evaluation protocol (A/B/C) is agreed.",
    }

    payload = {
        "metadata": {
            "dataset": "Level-A GLAD helical 300 nm",
            "total_jobs": len(ids),
            "alphas": alphas,
            "note": "Multiple strategies; no final choice until approved.",
            "recommended_first": "strategy_A_conservative_interpolation",
        },
        "strategy_A_conservative_interpolation": A,
        "strategy_B_leave_one_alpha_out": B,
        "strategy_C_seed_reproducibility": C,
        "strategy_D_all_data_pretraining": D,
    }
    (OUT / "levelA_300nm_nif_splits.json").write_text(json.dumps(payload, indent=2))

    md = ["# Level-A 300 nm NIF Split Strategies", "",
          f"Derived from `{MAN.relative_to(ROOT)}` ({len(ids)} jobs). "
          "No final choice until approved.", "",
          "## Strategy A - Conservative interpolation (recommended first)",
          f"- train ({len(A['train'])}): {A['train']}",
          f"- validation ({len(A['validation'])}): {A['validation']}",
          f"- test ({len(A['test'])}): {A['test']}", "",
          "## Strategy B - Leave-one-alpha-out",
          f"- {len(B['folds'])} folds, one per alpha in {alphas}", "",
          "## Strategy C - Seed reproducibility",
          f"- train ({len(C['train'])}): seed0 sweep",
          f"- val a80 replicas: {C['validation_replicate_alpha80']}",
          f"- val a85 replicas: {C['validation_replicate_alpha85']}", "",
          "## Strategy D - All-data pretraining",
          f"- train ({len(D['train'])}): all jobs; use only after A/B/C.", ""]
    (OUT / "levelA_300nm_nif_splits.md").write_text("\n".join(md))

    print(f"Strategy A: train={len(A['train'])} val={len(A['validation'])} test={len(A['test'])}")
    print(f"Strategy B: {len(B['folds'])} folds")
    print("Wrote levelA_300nm_nif_splits.json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
