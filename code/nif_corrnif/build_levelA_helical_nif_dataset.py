from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate /mnt/d/GLAD_PROJECT root")


PROJECT_ROOT = find_project_root()
ROOT = PROJECT_ROOT
GLAD_ROOT = PROJECT_ROOT / "01_GLAD_SIMULATION"
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
VOXEL_DESC_ROOT = PROJECT_ROOT / "03_VOXEL_DESCRIPTOR_ANALYSIS"
LIT_ROOT = PROJECT_ROOT / "04_LITERATURE_AND_PARAMETER_WORKFLOW"
THESIS_ROOT = PROJECT_ROOT / "05_THESIS_PAPER_ASSETS"
EXPERIMENTAL_ROOT = PROJECT_ROOT / "06_EXPERIMENTAL_VALIDATION"
UTIL_ROOT = PROJECT_ROOT / "07_UTILITIES_GENERATORS"
LEGACY_ROOT = PROJECT_ROOT / "08_LEGACY_REVIEW"


def resolve_project_path(value: str | Path, default_root: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    text = str(value).replace("\\", "/")
    marker = "deepseek_correction/"
    if marker in text:
        text = text.split(marker, 1)[1]
    prefix_map = {
        "pinn_training": PINN_NIF_ROOT / "pinn_training",
        "analysis": VOXEL_DESC_ROOT / "analysis",
        "descriptor_results_hybrid": VOXEL_DESC_ROOT / "descriptor_results_hybrid",
        "simulation_batch": GLAD_ROOT / "simulation_batch",
    }
    for prefix, base in sorted(prefix_map.items(), key=lambda item: len(item[0]), reverse=True):
        if text == prefix:
            return base
        if text.startswith(prefix + "/"):
            return base.joinpath(*[part for part in text[len(prefix) + 1:].split("/") if part])
    if text == ".":
        return default_root
    return default_root.joinpath(*[part for part in text.split("/") if part])


DEFAULT_VOX = PINN_NIF_ROOT / "pinn_training" / "datasets" / "levelA_helical_alpha_sweep" / "voxels"
DEFAULT_OUT = PINN_NIF_ROOT / "pinn_training" / "datasets" / "levelA_helical_nif"


def alpha_seed_from_name(path: Path) -> tuple[float, int]:
    parts = path.stem.split("_")
    alpha_part = next(p for p in parts if p.startswith("alpha") and p[5:].isdigit())
    seed_part = next(p for p in parts if p.startswith("seed") and p[4:].isdigit())
    alpha = float(alpha_part[5:])
    seed = int(seed_part[4:])
    return alpha, seed


def sample_points(arr: np.ndarray, alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    nx, ny, nz = arr.shape
    flat = arr.ravel()
    occupied = np.flatnonzero(flat > np.quantile(flat[flat > 0], 0.55)) if (flat > 0).any() else np.array([], dtype=int)
    empty = np.flatnonzero(flat <= np.quantile(flat, 0.50))
    n_occ = min(len(occupied), n // 3)
    n_empty = min(len(empty), n // 3)
    n_rand = n - n_occ - n_empty
    idx = []
    if n_occ:
        idx.append(rng.choice(occupied, n_occ, replace=len(occupied) < n_occ))
    if n_empty:
        idx.append(rng.choice(empty, n_empty, replace=len(empty) < n_empty))
    idx.append(rng.integers(0, flat.size, n_rand))
    idx = np.concatenate(idx)
    i, j, k = np.unravel_index(idx, arr.shape)
    x = (i + 0.5) / nx
    y = (j + 0.5) / ny
    z = (k + 0.5) / nz
    rho = flat[idx]
    alpha_norm = (alpha - 74.5) / 14.5
    return np.column_stack([x, y, z, np.full_like(x, alpha_norm), rho, np.full_like(x, alpha)]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Level A helical NIF NPZ dataset from voxels.")
    parser.add_argument("--voxels-dir", default=str(DEFAULT_VOX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--samples-per-voxel", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = resolve_project_path(args.out_dir, PINN_NIF_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists() and not args.force:
        print("SKIP existing dataset; use --force to rebuild")
        return 0
    rng = np.random.default_rng(args.seed)
    files = sorted(resolve_project_path(args.voxels_dir, PINN_NIF_ROOT).glob("rho_LA_*.npy"))
    if not files:
        raise SystemExit("No Level A voxel files found. Do not train.")
    samples = []
    meta_rows = []
    for path in files:
        alpha, seed = alpha_seed_from_name(path)
        arr = np.load(path).astype(np.float32)
        samples.append(sample_points(arr, alpha, args.samples_per_voxel, rng))
        meta_rows.append({"file": str(path.relative_to(ROOT)), "alpha": alpha, "seed": seed, "shape": list(arr.shape)})
    data = np.concatenate(samples, axis=0)
    rng.shuffle(data)
    n = len(data)
    train, val = int(0.8 * n), int(0.9 * n)
    np.savez_compressed(out / "train_samples.npz", data=data[:train, :5])
    np.savez_compressed(out / "val_samples.npz", data=data[train:val, :5])
    np.savez_compressed(out / "test_samples.npz", data=data[val:, :5])
    normalization = {"x_y_z": "unit voxel coordinates [0,1]", "alpha_norm": "(alpha_deg-74.5)/14.5", "rho": "as stored"}
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "source_voxels": meta_rows, "morphology_mode": "helical", "pitch_nm": 150, "warnings": ["Do not train if voxels are audit copies rather than Level A simulations."]}, indent=2) + "\n", encoding="utf-8")
    (out / "split_report.md").write_text(f"# Level A Split Report\n\ntrain={train}, val={val-train}, test={n-val}\n", encoding="utf-8")
    print(f"Wrote dataset to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
