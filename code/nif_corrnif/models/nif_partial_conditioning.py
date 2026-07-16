"""
Tier-2 Fix C: partial-realization conditioning.

Rationale (see AGENTS.md "P1/P2 STATUS SYNC", P2 tier 2, Fix C, 2026-07-06):
Fix A (hash-grid encoding) and Fix B (continuous regression) both still ask
the network to predict a SPECIFIC realization's exact occupancy from
(x,y,z,alpha) alone. P1's corotating-frame oracle already established that a
fixed spatial template gives zero gain over the bulk average at every angle
— column (x,y) positions are an independent per-realization random draw
(nucleation-site randomization), not a deterministic function of alpha. No
amount of encoding or loss-function change can make (x,y,z,alpha) determine
something the physics itself does not determine from those variables alone.

Fix C sidesteps this by conditioning on part of the SAME realization instead:
an "observed" near-substrate region (which DOES carry the seed-specific
nucleation information) is encoded into a latent vector and used, via the
same FiLM mechanism already proven for alpha-conditioning, to condition the
prediction of the "target" region (the rest of the film, strictly disjoint
in z from the observed region). This reframes the task from "predict a
random draw from unrelated inputs" (ill-posed) to "complete a partially
observed instance of this specific structure" (well-posed).

Components:
  - ObservedMapEncoder: small CNN over the (1, 50, 50) observed-region binary
    occupancy footprint (see build_levelA_helical_nif_partial_conditioning_
    dataset.py for how this map is constructed: max-pooled occupancy over
    the bottom 15% of z-slices, per (x,y) column). Two conv+SiLU blocks with
    stride-2 downsampling, then global average pool, then a linear
    projection to `latent_dim`. Deliberately simple (this is a first test of
    the hypothesis, not a tuned architecture).
  - PartialConditionedNIF: same FourierXYZ spatial encoder + FiLM trunk/head
    structure as FiLMConditionedNIF, but the FiLM gamma/beta parameters are
    now produced from concat(alpha_norm, observed_latent) instead of
    alpha_norm alone — i.e. the conditioning signal used to modulate the
    trunk carries BOTH the macroscopic deposition angle AND per-realization
    structural information extracted from that realization's own early
    growth.

Forward signature deliberately differs from the other models in
nif_levelA_helical.py / nif_hashgrid.py (which all take a single (B, F)
tensor): PartialConditionedNIF.forward(x, obs_map) takes the observed map as
a SEPARATE argument, since it is realization-level, not point-level, data.
This is intentionally NOT wired into nif_levelA_helical.py's build_model() —
it needs a dedicated training script (train_nif_partial_conditioning.py)
that knows to look up each sample's observed_map by its realization_id
column before calling the model, which the other architectures' shared
training script (train_nif_hardlabel_v2.py) has no way to do generically.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ObservedMapEncoder(nn.Module):
    """CNN encoder: (B, 1, H, W) binary occupancy footprint -> (B, latent_dim)."""

    def __init__(self, latent_dim: int = 32, in_hw: tuple[int, int] = (50, 50)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   # (B,16,25,25)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # (B,32,13,13)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B,32,1,1)
        self.proj = nn.Linear(32, latent_dim)

    def forward(self, obs_map: torch.Tensor) -> torch.Tensor:
        # obs_map: (B, H, W) or (B, 1, H, W)
        if obs_map.dim() == 3:
            obs_map = obs_map.unsqueeze(1)
        h = self.act(self.conv1(obs_map))
        h = self.act(self.conv2(h))
        h = self.pool(h).flatten(1)  # (B, 32)
        return self.proj(h)          # (B, latent_dim)


class PartialConditionedNIF(nn.Module):
    """Partial-realization-conditioned NIF. Spatial input: (x, y, z, alpha_norm)
    — same 4 features as FiLMConditionedNIF. Conditioning input: the
    realization's own observed-region occupancy map (B, 50, 50), encoded via
    `ObservedMapEncoder` and concatenated with alpha_norm before the FiLM MLP.
    """

    def __init__(
        self,
        hidden: int = 256,
        n_layers: int = 4,
        n_freq: int = 32,
        sigma: float = 6.0,
        latent_dim: int = 32,
        obs_map_hw: tuple[int, int] = (50, 50),
        film_scale: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        # 2026-07-08: film_scale exposed (was hardcoded 0.1) after the permutation
        # test diagnosis found the observed-map conditioning pathway's influence on
        # predictions was ~1e-3 in magnitude at alpha=85 even where the input carried
        # real seed-distinguishing signal -- consistent with the 0.1x FiLM damping
        # (inherited from the alpha-only baseline, for stability) capping how much
        # this pathway could ever matter. Default unchanged (0.1) for backward
        # compatibility with the already-trained run; film_scale=1.0 gives an
        # unclamped variant to test the diagnosis directly.
        self.film_scale = float(film_scale)
        # Local import to avoid a hard dependency between sibling model
        # modules at import time in every context; nif_hashgrid.py already
        # establishes the pattern of self-contained sibling modules loaded
        # lazily by nif_levelA_helical.py's build_model(). Here we just need
        # FourierXYZ, so we re-implement the (tiny) encoder inline instead of
        # importing across files, keeping this module fully self-contained.
        import math

        class _FourierXYZ(nn.Module):
            def __init__(self, n_freq: int, sigma: float, seed: int = 123):
                super().__init__()
                g = torch.Generator().manual_seed(seed)
                self.register_buffer("B", torch.randn(3, n_freq, generator=g) * sigma)

            def forward(self, xyz: torch.Tensor) -> torch.Tensor:
                proj = 2 * math.pi * xyz @ self.B
                return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        self.encoder = _FourierXYZ(n_freq=n_freq, sigma=sigma)
        self.obs_encoder = ObservedMapEncoder(latent_dim=latent_dim, in_hw=obs_map_hw)

        self.hidden = hidden
        self.n_layers = n_layers
        in_dim = 2 * n_freq
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(n_layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        # Conditioning MLP: input = alpha_norm (1) + observed latent (latent_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(1 + latent_dim, 64), nn.SiLU(), nn.Linear(64, 2 * n_layers * hidden)
        )

    def forward(self, x: torch.Tensor, obs_map: torch.Tensor) -> torch.Tensor:
        # x: (B, 4) = [x, y, z, alpha_norm]; obs_map: (B, 50, 50) or (B, 1, 50, 50)
        h = self.encoder(x[:, :3])
        obs_latent = self.obs_encoder(obs_map)             # (B, latent_dim)
        cond_in = torch.cat([x[:, 3:4], obs_latent], dim=-1)  # (B, 1+latent_dim)
        film = self.cond_mlp(cond_in)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta = beta.view(-1, self.n_layers, self.hidden)
        s = self.film_scale
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + s * gamma[:, i]) * h + s * beta[:, i])
        return self.head(h).squeeze(-1)


def build_partial_conditioning_model(config: dict) -> nn.Module:
    kwargs = {
        "hidden":     int(config.get("hidden", 256)),
        "n_layers":   int(config.get("n_layers", 4)),
        "n_freq":     int(config.get("n_freq", 32)),
        "sigma":      float(config.get("sigma", 6.0)),
        "latent_dim": int(config.get("pc_latent_dim", 32)),
        "film_scale": float(config.get("film_scale", 0.1)),
    }
    return PartialConditionedNIF(**kwargs)
