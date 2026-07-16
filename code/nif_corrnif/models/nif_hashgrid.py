"""
Tier-2 Fix A: multi-resolution hash-grid positional encoding (Instant-NGP style)
for the Level-A helical NIF, as an alternative to the global Fourier encoding in
FiLMConditionedNIF (nif_levelA_helical.py).

Motivation (see AGENTS.md "P1/P2 STATUS SYNC", P2 tier 2, 2026-07-05):
  FiLMConditionedNIF's FourierXYZ encoder uses a fixed random Fourier basis with
  GLOBAL support over the whole [0,1]^3 coordinate domain. To represent a sharp,
  LOCAL column boundary it must combine many high-frequency sinusoids, each of
  which oscillates over the entire domain — there is no mechanism biasing the
  network toward local spatial structure. A multi-resolution hash grid instead
  stores a small learned feature vector per grid cell at each of several
  resolutions, trilinearly interpolated — locality is architectural, not
  something the optimizer has to discover from a global basis.

Design notes:
  - Coarse levels here always hash (no dense/hashed switch as in the original
    Instant-NGP paper, where levels with (N+1)^3 <= table_size use a direct
    dense index instead of a spatial hash). This is a deliberate simplification
    for a first implementation — a direct dense index is a strict special case
    of a "hash" that never collides, so the encoding is still architecturally
    valid; it just does not exploit the (small) extra safety margin against
    collisions at coarse resolutions. Flagged here for a future revision if the
    coarse-level collision rate turns out to matter empirically.
  - Table size is kept modest (2**12 = 4096 entries/level x 2 features/level x
    8 levels = 65,536 embedding parameters) specifically so the total parameter
    count stays in the same ballpark as FiLMConditionedNIF (347,521 params at
    hidden=256/n_layers=4/n_freq=32) — see HashGridConditionedNIF docstring for
    the exact total. This isolates "does local encoding help" from "does more
    capacity help".
  - The FiLM alpha-conditioning mechanism (proven to work for angle
    conditioning in FiLMConditionedNIF) is kept byte-for-byte identical in
    structure; only the spatial encoder feeding the trunk MLP is replaced.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# Instant-NGP spatial hash primes (Muller et al. 2022, Sec 3.1). pi_1 = 1 so the
# first coordinate is not permuted (harmless, standard choice).
_HASH_PRIMES = (1, 2_654_435_761, 805_459_861)


class HashGridEncoder3D(nn.Module):
    """Multi-resolution hash-grid encoder for 3D coordinates in [0,1]^3.

    For each of `n_levels` resolutions (geometrically spaced between
    `base_resolution` and `finest_resolution`), the input point is scaled to
    that grid, its 8 surrounding integer grid-cell corners are looked up in a
    learned per-level embedding table (spatial hash, mod `table_size`), and the
    8 corner features are trilinearly interpolated. Levels are concatenated.

    Output dim = n_levels * feature_dim.
    """

    def __init__(
        self,
        n_levels: int = 8,
        feature_dim: int = 2,
        log2_table_size: int = 12,
        base_resolution: int = 16,
        finest_resolution: int = 256,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.feature_dim = feature_dim
        self.table_size = 1 << log2_table_size

        if n_levels > 1:
            growth = math.exp(
                (math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1)
            )
        else:
            growth = 1.0
        resolutions = [round(base_resolution * (growth ** lvl)) for lvl in range(n_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.float32))

        # One embedding table per level: (table_size, feature_dim).
        # Small init (matches Instant-NGP's U(-1e-4, 1e-4)) so the encoding
        # starts near-zero and does not dominate the FiLM-conditioned trunk
        # before training shapes it.
        tables = torch.empty(n_levels, self.table_size, feature_dim).uniform_(-1e-4, 1e-4)
        self.embeddings = nn.Parameter(tables)

        primes = torch.tensor(_HASH_PRIMES, dtype=torch.int64)
        self.register_buffer("primes", primes)

        # Precompute the 8 corner offsets of a unit cube: (8, 3) in {0,1}.
        corner_offsets = torch.tensor(
            [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
            dtype=torch.int64,
        )
        self.register_buffer("corner_offsets", corner_offsets)

    def _hash(self, coords_int: torch.Tensor) -> torch.Tensor:
        """coords_int: (..., 3) int64 grid coordinates -> (...,) table indices in [0, table_size)."""
        h = coords_int[..., 0] * self.primes[0]
        h = h ^ (coords_int[..., 1] * self.primes[1])
        h = h ^ (coords_int[..., 2] * self.primes[2])
        # Python-style modulo on possibly-negative int64 via remainder (not fmod).
        return torch.remainder(h, self.table_size)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, 3) in [0,1]. Returns (B, n_levels * feature_dim)."""
        B = xyz.shape[0]
        level_outputs = []
        for lvl in range(self.n_levels):
            res = self.resolutions[lvl]
            scaled = xyz * res  # (B, 3), continuous grid coords at this resolution
            floor_xyz = torch.floor(scaled)
            frac = scaled - floor_xyz  # (B, 3) in [0,1), trilinear weights
            floor_int = floor_xyz.to(torch.int64)  # (B, 3)

            corner_coords = floor_int.unsqueeze(1) + self.corner_offsets.unsqueeze(0)  # (B, 8, 3)
            idx = self._hash(corner_coords)  # (B, 8)
            corner_feats = self.embeddings[lvl][idx]  # (B, 8, feature_dim)

            # Trilinear interpolation weights matching corner_offsets ordering
            # (dx,dy,dz) each in {0,1}: weight = prod over axis of (frac if bit else 1-frac).
            w = torch.ones(B, 8, device=xyz.device, dtype=xyz.dtype)
            for axis in range(3):
                bit = self.corner_offsets[:, axis].to(xyz.dtype)  # (8,)
                axis_frac = frac[:, axis:axis + 1]  # (B, 1)
                w = w * (bit.unsqueeze(0) * axis_frac + (1 - bit.unsqueeze(0)) * (1 - axis_frac))

            interp = (corner_feats * w.unsqueeze(-1)).sum(dim=1)  # (B, feature_dim)
            level_outputs.append(interp)

        return torch.cat(level_outputs, dim=-1)  # (B, n_levels*feature_dim)


class HashGridConditionedNIF(nn.Module):
    """Hash-grid analogue of FiLMConditionedNIF. Input: (x, y, z, alpha_norm) — 4 features.

    Identical FiLM trunk/head/alpha_mlp structure to FiLMConditionedNIF; only the
    spatial encoder is swapped (global Fourier -> local multi-resolution hash
    grid). Default settings (n_levels=8, feature_dim=2, log2_table_size=12,
    base_resolution=16, finest_resolution=256) give:
      - encoding output dim = 8*2 = 16 (vs. FourierXYZ's 2*32=64)
      - embedding params = 8 * 4096 * 2 = 65,536
      - total model params ~ 400,769 (vs. FiLMConditionedNIF's 347,521) —
        same ballpark, ~15% larger, dominated by the shared trunk/alpha_mlp,
        not by the encoding table.
    """

    def __init__(
        self,
        hidden: int = 256,
        n_layers: int = 4,
        n_levels: int = 8,
        feature_dim: int = 2,
        log2_table_size: int = 12,
        base_resolution: int = 16,
        finest_resolution: int = 256,
        **_ignored,
    ):
        super().__init__()
        self.encoder = HashGridEncoder3D(
            n_levels=n_levels,
            feature_dim=feature_dim,
            log2_table_size=log2_table_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )
        self.hidden = hidden
        self.n_layers = n_layers
        in_dim = n_levels * feature_dim
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(n_layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.alpha_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2 * n_layers * hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4) = [x, y, z, alpha_norm]
        h = self.encoder(x[:, :3])
        film = self.alpha_mlp(x[:, 3:4])
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


def build_hashgrid_model(config: dict) -> nn.Module:
    kwargs = {
        "hidden":            int(config.get("hidden", 256)),
        "n_layers":          int(config.get("n_layers", 4)),
        "n_levels":          int(config.get("hg_n_levels", 8)),
        "feature_dim":       int(config.get("hg_feature_dim", 2)),
        "log2_table_size":   int(config.get("hg_log2_table_size", 12)),
        "base_resolution":   int(config.get("hg_base_resolution", 16)),
        "finest_resolution": int(config.get("hg_finest_resolution", 256)),
    }
    return HashGridConditionedNIF(**kwargs)
