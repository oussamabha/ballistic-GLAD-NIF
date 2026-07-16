from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierXYZ(nn.Module):
    """3D Fourier encoder: (x,y,z) → 2*n_freq features."""
    def __init__(self, n_freq: int = 32, sigma: float = 6.0, seed: int = 123):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(3, n_freq, generator=g) * sigma)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * xyz @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierXYZPhi(nn.Module):
    """5D Fourier encoder: (x, y, z, φ_sin, φ_cos) → 2*n_freq features.
    Encodes helix phase alongside spatial coordinates. The random projection
    mixes cross-frequency terms between xyz and phase, allowing the model to
    learn column position as a function of helix rotation angle.
    """
    def __init__(self, n_freq: int = 32, sigma: float = 6.0, seed: int = 123):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(5, n_freq, generator=g) * sigma)

    def forward(self, xyzphi: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * xyzphi @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierXYPhi(nn.Module):
    """4D Fourier encoder: (x, y, φ_sin, φ_cos) → 2*n_freq features.
    z is EXCLUDED — prevents model from using z as a training-sample identifier.
    Forces the model to localize columns via helix phase φ, which IS consistent
    across z-positions (same phase → same column → holdout generalization).
    """
    def __init__(self, n_freq: int = 32, sigma: float = 6.0, seed: int = 123):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(4, n_freq, generator=g) * sigma)

    def forward(self, xyphi: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * xyphi @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierAll6(nn.Module):
    """6D Fourier encoder: (x, y, z, α_norm, φ_sin, φ_cos) → 2*n_freq features.
    Used by AlphaInterpolatingFlatNIF — all features encoded jointly, no FiLM.
    """
    def __init__(self, n_freq: int = 32, sigma: float = 6.0, seed: int = 123):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(6, n_freq, generator=g) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class BaselineConcatNIF(nn.Module):
    def __init__(self, hidden: int = 256, n_layers: int = 4, n_freq: int = 32, sigma: float = 6.0):
        super().__init__()
        self.encoder = FourierXYZ(n_freq=n_freq, sigma=sigma)
        in_dim = 2 * n_freq + 1
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz = x[:, :3]
        alpha = x[:, 3:4]
        return self.net(torch.cat([self.encoder(xyz), alpha], dim=-1)).squeeze(-1)


class FiLMConditionedNIF(nn.Module):
    """Original architecture. Input: (x, y, z, α_norm) — 4 features."""
    def __init__(self, hidden: int = 256, n_layers: int = 4, n_freq: int = 32, sigma: float = 6.0):
        super().__init__()
        self.encoder = FourierXYZ(n_freq=n_freq, sigma=sigma)
        self.hidden = hidden
        self.n_layers = n_layers
        in_dim = 2 * n_freq
        self.layers = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.alpha_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2 * n_layers * hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x[:, :3])
        film = self.alpha_mlp(x[:, 3:4])
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


class PhaseConditionedFiLMNIF(nn.Module):
    """Phase-conditioned NIF. Input: (x, y, z, α_norm, φ_sin, φ_cos) — 6 features.

    Helix phase φ = 2π × k / pitch_voxels (k = z-slice index).
    φ_sin = sin(φ), φ_cos = cos(φ) — periodic encoding of helix rotation.

    Architecture:
      - Spatial + phase: (x,y,z,φ_sin,φ_cos) → FourierXYZPhi → 2*n_freq features
      - Alpha: α_norm → FiLM MLP → gamma/beta for each MLP layer
      - FiLM conditioning on per-alpha network: gamma/beta are fixed rescalings,
        no multi-alpha competition, no collapse risk.

    Why this solves the z-generalization barrier:
      At phase φ=0.60×2π (z=121, holdout), the model has seen this phase during
      training at z≈45 and z≈120. It learned 'at phase φ_1, column is at (x_1,y_1)'
      and generalizes correctly to new z-positions with the same phase.
    """
    def __init__(self, hidden: int = 256, n_layers: int = 4, n_freq: int = 32, sigma: float = 6.0):
        super().__init__()
        self.encoder = FourierXYZPhi(n_freq=n_freq, sigma=sigma)   # 5D → 2*n_freq
        self.hidden = hidden
        self.n_layers = n_layers
        in_dim = 2 * n_freq
        self.layers = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.alpha_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2 * n_layers * hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6) = [x, y, z, alpha_norm, phi_sin, phi_cos]
        xyzphi = torch.cat([x[:, :3], x[:, 4:6]], dim=-1)   # (B, 5)
        h = self.encoder(xyzphi)
        film = self.alpha_mlp(x[:, 3:4])
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta  = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


class PhaseOnlyFiLMNIF(nn.Module):
    """Phase-only NIF: z excluded from Fourier encoder, helix phase is sole position signal.

    Input: (x, y, z, α_norm, φ_sin, φ_cos) — same 6-column format as PhaseConditionedFiLMNIF.

    Key difference from PhaseConditionedFiLMNIF:
      FourierXYPhi receives (x, y, φ_sin, φ_cos) — z is NOT in the Fourier basis.
      z_norm is appended as a raw scalar AFTER the Fourier encoding (captures
      column-width growth along deposition depth without enabling z-indexing).

    Why this solves the holdout failure:
      Without z in the Fourier encoder, the model cannot build a z-specific lookup.
      It must learn f(x, y, φ) → occupancy. Since the same helix phase φ appears
      at z≈45 and z≈120 in training, and at z=121 in the holdout, the model
      can apply the learned phase→column mapping to the holdout correctly.
    """
    def __init__(self, hidden: int = 256, n_layers: int = 4, n_freq: int = 32, sigma: float = 6.0):
        super().__init__()
        self.encoder = FourierXYPhi(n_freq=n_freq, sigma=sigma)   # 4D → 2*n_freq
        self.hidden   = hidden
        self.n_layers = n_layers
        in_dim = 2 * n_freq + 1   # +1 for raw z_norm appended after Fourier
        self.layers    = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden)
                                        for i in range(n_layers)])
        self.head      = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.alpha_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(),
                                        nn.Linear(64, 2 * n_layers * hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6) = [x, y, z, alpha_norm, phi_sin, phi_cos]
        xyphi = torch.cat([x[:, :2], x[:, 4:6]], dim=-1)   # (B,4): x,y,φ_sin,φ_cos
        z_raw = x[:, 2:3]                                    # (B,1): z depth scalar
        h = torch.cat([self.encoder(xyphi), z_raw], dim=-1)  # (B, 2*n_freq+1)
        film = self.alpha_mlp(x[:, 3:4])
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta  = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


class AlphaInterpolatingFlatNIF(nn.Module):
    """Multi-alpha flat NIF for leave-one-alpha-out generalization (Option C).

    Input: (x, y, z, α_norm, φ_sin, φ_cos) — all 6 features through joint Fourier encoder.
    No FiLM conditioning → no FiLM collapse risk when training on multiple alphas.

    The joint Fourier encoder creates cross-frequency terms between α and (xyz,φ),
    allowing the model to learn how morphology changes continuously with alpha.
    At test time, an unseen alpha between two training angles is interpolated by
    the smooth response surface learned across all training alpha values.

    Gate: APPROVE_ALPHA_INTERPOLATING_NIF_TRAINING
    """
    def __init__(self, hidden: int = 256, n_layers: int = 4, n_freq: int = 32, sigma: float = 6.0):
        super().__init__()
        self.encoder = FourierAll6(n_freq=n_freq, sigma=sigma)   # 6D → 2*n_freq
        in_dim = 2 * n_freq
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6) = [x, y, z, alpha_norm, phi_sin, phi_cos]
        return self.net(self.encoder(x)).squeeze(-1)


def build_model(config: dict) -> nn.Module:
    name = config.get("architecture", "FiLMConditionedNIF")
    kwargs = {
        "hidden":   int(config.get("hidden",   256)),
        "n_layers": int(config.get("n_layers", 4)),
        "n_freq":   int(config.get("n_freq",   32)),
        "sigma":    float(config.get("sigma",  6.0)),
    }
    if name == "BaselineConcatNIF":
        return BaselineConcatNIF(**kwargs)
    if name == "FiLMConditionedNIF":
        return FiLMConditionedNIF(**kwargs)
    if name == "PhaseConditionedFiLMNIF":
        return PhaseConditionedFiLMNIF(**kwargs)
    if name == "PhaseOnlyFiLMNIF":
        return PhaseOnlyFiLMNIF(**kwargs)
    if name == "AlphaInterpolatingFlatNIF":
        return AlphaInterpolatingFlatNIF(**kwargs)
    if name == "HashGridConditionedNIF":
        # Tier-2 Fix A (spatial inductive bias): defined in the sibling module
        # nif_hashgrid.py, not in this file — loaded lazily so this file (and
        # the other architectures in it) has zero new import-time dependency.
        import importlib.util
        from pathlib import Path

        hg_path = Path(__file__).resolve().parent / "nif_hashgrid.py"
        spec = importlib.util.spec_from_file_location("nif_hashgrid", hg_path)
        hg_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(hg_mod)
        return hg_mod.build_hashgrid_model(config)
    raise ValueError(f"Unknown architecture: {name}")
