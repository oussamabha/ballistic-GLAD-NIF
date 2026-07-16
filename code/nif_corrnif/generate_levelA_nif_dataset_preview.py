#!/usr/bin/env python
"""
Lightweight preview figures for the Level-A 300 nm helical NIF dataset.
matplotlib only (no seaborn, no heavy 3D). Reads the real voxels via the manifest.

Outputs to pinn_training/reports/levelA_nif_dataset_preview/:
  - <job_id>_slices.png           central XY / XZ / YZ slices (per seed0 job)
  - alpha_vs_occupied_fraction.png
  - alpha_vs_mean_density.png
  - alpha_vs_porosity_from_metadata.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_manifest(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_slices(voxel: np.ndarray, job_id: str, out_dir: Path) -> None:
    cx, cy, cz = (s // 2 for s in voxel.shape)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(voxel[:, :, cz].T, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[0].set_title(f"XY  z={cz}"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    ax[1].imshow(voxel[:, cy, :].T, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax[1].set_title(f"XZ  y={cy}"); ax[1].set_xlabel("x"); ax[1].set_ylabel("z")
    im = ax[2].imshow(voxel[cx, :, :].T, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax[2].set_title(f"YZ  x={cx}"); ax[2].set_xlabel("y"); ax[2].set_ylabel("z")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rho")
    fig.suptitle(job_id, fontsize=9)
    fig.savefig(out_dir / f"{job_id}_slices.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def line_plot(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = np.argsort(x)
    ax.plot(np.array(x)[order], np.array(y)[order], "o-", lw=2, ms=7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pinn_training/datasets/levelA_300nm_nif_manifest.csv")
    ap.add_argument("--base-dir", default=".")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = base / "pinn_training/reports/levelA_nif_dataset_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(base / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest))

    alphas, occ_frac, mean_den, poros = [], [], [], []
    for r in rows:
        if int(r["seed"]) != 0:
            continue
        voxel = np.load(base / r["voxel_path"], allow_pickle=False)
        plot_slices(voxel, r["job_id"], out_dir)
        alphas.append(int(float(r["alpha"])))
        occ_frac.append(float((voxel > 0.5).mean()))
        mean_den.append(float(voxel.mean()))
        poros.append(float(r["porosity"]))
        print(f"  slices: {r['job_id']}  occ>0.5={occ_frac[-1]:.3f}  mean_rho={mean_den[-1]:.4f}")

    line_plot(alphas, occ_frac, "alpha (deg)", "occupied fraction (rho>0.5)",
              "Alpha vs occupied fraction", out_dir / "alpha_vs_occupied_fraction.png")
    line_plot(alphas, mean_den, "alpha (deg)", "mean rho",
              "Alpha vs mean density", out_dir / "alpha_vs_mean_density.png")
    line_plot(alphas, poros, "alpha (deg)", "porosity (%) [metadata]",
              "Alpha vs porosity (from metadata)", out_dir / "alpha_vs_porosity_from_metadata.png")

    print(f"Wrote figures to {out_dir.relative_to(base) if not out_dir.is_absolute() else out_dir}")
    print(f"  seed0 jobs plotted: {len(alphas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
