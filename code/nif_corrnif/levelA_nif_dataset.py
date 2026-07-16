"""
Level-A GLAD helical 300 nm NIF dataset loader (real data).

Targets the frozen voxels under
  pinn_training/datasets/universal_glad_pipeline/voxels/rho_<job_id>.npy
which are float32 grids of shape (50, 50, 151) holding continuous density
rho in [0, 1].

Conventions (consistent with build_levelA_helical_nif_dataset.py):
  coordinate norm : index i in [0, n)  ->  2*(i + 0.5)/n - 1   in (-1, 1)
  alpha norm      : (alpha - 74.5) / 14.5   -> 60 -> -1, 89 -> +1
  target          : continuous rho value at the sampled voxel
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # torch optional for pure-numpy use
    _HAS_TORCH = False
    Dataset = object  # type: ignore


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate /mnt/d/GLAD_PROJECT root")


PROJECT_ROOT = find_project_root()
GLAD_ROOT = PROJECT_ROOT / "01_GLAD_SIMULATION"
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
VOXEL_DESC_ROOT = PROJECT_ROOT / "03_VOXEL_DESCRIPTOR_ANALYSIS"


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
        "descriptor_results": VOXEL_DESC_ROOT / "descriptor_results",
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


GRID_SHAPE = (50, 50, 151)
ALPHA_CENTER = 74.5
ALPHA_HALF_RANGE = 14.5


def normalize_alpha(alpha: float) -> float:
    """Map alpha (deg) to [-1, 1]: 60 -> -1, 89 -> +1."""
    return (alpha - ALPHA_CENTER) / ALPHA_HALF_RANGE


def normalize_indices(ijk: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Map integer voxel indices to (-1, 1) per axis via cell centers."""
    out = np.empty(ijk.shape, dtype=np.float32)
    for ax in range(3):
        out[:, ax] = 2.0 * (ijk[:, ax] + 0.5) / shape[ax] - 1.0
    return out


class LevelANIFManifest:
    """Parse and filter the Level-A NIF manifest (CSV or JSON)."""

    def __init__(self, manifest_path: str, base_dir: str = "."):
        self.base_dir = resolve_project_path(base_dir, PROJECT_ROOT)
        self.jobs: List[Dict] = []
        p = resolve_project_path(manifest_path, self.base_dir)
        if p.suffix == ".csv":
            self._load_csv(p)
        elif p.suffix == ".json":
            self.jobs = json.loads(p.read_text())["jobs"]
            self._coerce_types()
        else:
            raise ValueError(f"Unsupported manifest format: {p.suffix}")

    def _load_csv(self, path: Path) -> None:
        with open(path, newline="") as f:
            self.jobs = list(csv.DictReader(f))
        self._coerce_types()

    def _coerce_types(self) -> None:
        for j in self.jobs:
            j["alpha"] = int(float(j["alpha"]))
            j["seed"] = int(j["seed"])
            j["final_atoms"] = int(j["final_atoms"])
            j["porosity"] = float(j["porosity"])
            j["crop_fraction_estimate"] = float(j["crop_fraction_estimate"])
            j["descriptor_rows"] = int(j["descriptor_rows"])
            acc = j["accepted_for_nif_training"]
            j["accepted_for_nif_training"] = str(acc).strip().lower() == "true"

    def filter_by_split(self, split_role: str) -> "LevelANIFManifest":
        out = LevelANIFManifest.__new__(LevelANIFManifest)
        out.base_dir = self.base_dir
        out.jobs = [j for j in self.jobs if j["split_role"] == split_role]
        return out

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i: int) -> Dict:
        return self.jobs[i]


def sample_points_from_voxel(
    voxel: np.ndarray,
    num_occupied: int = 2500,
    num_empty: int = 2500,
    num_random: int = 0,
    occupied_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced sampling from a continuous-rho voxel grid.

    Returns:
      points : (N, 3) float32 normalized coords in (-1, 1)
      rho    : (N,)   float32 continuous density target at each point
    """
    rng = np.random.default_rng(seed)
    flat = voxel.reshape(-1)
    occ_idx = np.flatnonzero(flat > occupied_threshold)
    emp_idx = np.flatnonzero(flat <= occupied_threshold)

    chosen: List[np.ndarray] = []
    if num_occupied and occ_idx.size:
        chosen.append(rng.choice(occ_idx, min(num_occupied, occ_idx.size),
                                 replace=occ_idx.size < num_occupied))
    if num_empty and emp_idx.size:
        chosen.append(rng.choice(emp_idx, min(num_empty, emp_idx.size),
                                 replace=emp_idx.size < num_empty))
    if num_random:
        chosen.append(rng.integers(0, flat.size, num_random))

    if not chosen:
        raise ValueError("No samples produced; check thresholds / counts.")

    idx = np.concatenate(chosen)
    ijk = np.stack(np.unravel_index(idx, voxel.shape), axis=1).astype(np.float32)
    points = normalize_indices(ijk, voxel.shape)
    rho = flat[idx].astype(np.float32)
    return points, rho


class LevelANIFDataset(Dataset):
    """
    Map-style dataset of (input, target) pairs.

      input  = [x_norm, y_norm, z_norm, alpha_norm]   (4,)
      target = [rho]                                   (1,)

    Voxels are loaded lazily from disk (not preloaded to GPU). A small per-job
    voxel cache on CPU is optional.
    """

    def __init__(
        self,
        manifest_path: str,
        base_dir: str = ".",
        split_role: Optional[str] = None,
        num_occupied: int = 2500,
        num_empty: int = 2500,
        num_random: int = 0,
        occupied_threshold: float = 0.5,
        seed: Optional[int] = 42,
        cache_voxels: bool = True,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for LevelANIFDataset")
        self.manifest = LevelANIFManifest(manifest_path, base_dir)
        if split_role:
            self.manifest = self.manifest.filter_by_split(split_role)
        self.base_dir = resolve_project_path(base_dir, PROJECT_ROOT)
        self.num_occupied = num_occupied
        self.num_empty = num_empty
        self.num_random = num_random
        self.occupied_threshold = occupied_threshold
        self.seed = seed
        self.cache_voxels = cache_voxels
        self._cache: Dict[str, np.ndarray] = {}

        self.per_job = num_occupied + num_empty + num_random
        self._index: List[Tuple[int, int]] = [
            (job_i, s) for job_i in range(len(self.manifest))
            for s in range(self.per_job)
        ]

    def _load_voxel(self, job_i: int) -> np.ndarray:
        job = self.manifest[job_i]
        jid = job["job_id"]
        if jid in self._cache:
            return self._cache[jid]
        arr = np.load(resolve_project_path(job["voxel_path"], self.base_dir), allow_pickle=False)
        if arr.shape != GRID_SHAPE:
            raise ValueError(f"{jid}: shape {arr.shape} != {GRID_SHAPE}")
        if self.cache_voxels:
            self._cache[jid] = arr
        return arr

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        job_i, s = self._index[idx]
        job = self.manifest[job_i]
        voxel = self._load_voxel(job_i)
        # deterministic per (job, sample-block) sampling
        pts, rho = sample_points_from_voxel(
            voxel, self.num_occupied, self.num_empty, self.num_random,
            self.occupied_threshold,
            seed=None if self.seed is None else self.seed + 1000 * job_i,
        )
        p = pts[s]
        r = rho[s]
        a = normalize_alpha(job["alpha"])
        x = np.array([p[0], p[1], p[2], a], dtype=np.float32)
        y = np.array([r], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


if __name__ == "__main__":
    print("Level-A helical 300 nm NIF loader. Import LevelANIFDataset / "
          "LevelANIFManifest / sample_points_from_voxel.")
