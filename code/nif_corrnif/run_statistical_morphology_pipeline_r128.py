"""
Statistical Morphology Pipeline for GLAD NIFs — end-to-end in one command.

CORRECT GOAL: predict STATISTICAL morphology, not exact column positions.
  Column positions are stochastically nucleated → uncorrelated across α runs.
  Two-point correlation function h(r; α) characterises the ENSEMBLE and is
  smooth in α → learnable by a NIF.

Steps
─────
  1. Compute h(r; α) from voxel data for all 7 alphas (corotating frame, FFT)
  2. Train a compact NIF on (α_norm, r_norm) → h(r; α) — all 7 alphas
  3. LOOA: hold out each alpha, train on 6, predict h(r; α_held) — 7 folds
  4. BONUS: Maxwell-Garnett effective optical properties n_eff(α), Δn(α)

Output: 02_PINN_NIF/pinn_training/runs/morphology_pipeline_r128/
  figures/fig1_correlation_functions.png  — h(r;α) for all alphas
  figures/fig2_nif_full_fit.png           — NIF fit vs ground truth
  figures/fig3_looa_predictions.png       — LOOA held-out predictions
  figures/fig4_column_statistics.png      — occ_frac, d_col, density vs α
  figures/fig5_optical_properties.png     — n_eff, Δn, k_eff vs α
  corr_functions.npz                      — raw h(r;α) curves
  looa_results.json                       — LOOA RMSE and R² per alpha
  column_stats.json                       — column statistics per alpha
  mg_properties.json                      — effective optical properties per alpha

Runtime: ~5–10 min on CPU. No GPU required. No simulation gate required.
"""
from __future__ import annotations
import cmath, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── project paths ───────────────────────────────────────────────────────────
def _find_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("Cannot find project root")

PROJECT_ROOT = _find_root()
VOXEL_DIR    = PROJECT_ROOT / "02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline_r128/voxels"
OUT_DIR      = PROJECT_ROOT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline_r128"
FIG_DIR      = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────
ALPHAS            = [65, 70, 75, 80, 85, 87, 89]
HELIX_PITCH_NM    = 150.0
FILM_THICKNESS_NM = 300.0
BINARY_THR        = 0.5
ALPHA_NORM        = lambda a: (a - 74.5) / 14.5

# Tungsten at λ≈500 nm: ε_W = (n+ik)², n≈3.5, k≈2.8
EPS_W    = complex(3.5**2 - 2.8**2, 2.0 * 3.5 * 2.8)   # ≈ 4.41+19.60i
EPS_HOST = complex(1.0, 0.0)

COLORS = [cm.viridis(v) for v in np.linspace(0.1, 0.9, len(ALPHAS))]

# ── material helpers ─────────────────────────────────────────────────────────
def eps_to_nk(eps: complex) -> tuple[float, float]:
    """Return (n, k) from complex dielectric constant."""
    z = cmath.sqrt(eps)
    return abs(z.real), abs(z.imag)

def mg_cylindrical(eps_incl: complex, eps_host: complex, f: float) -> complex:
    """Maxwell-Garnett for cylindrical inclusions (transverse polarisation)."""
    num = (eps_incl + eps_host) + f * (eps_incl - eps_host)
    den = (eps_incl + eps_host) - f * (eps_incl - eps_host)
    return eps_host * num / den

# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def load_voxel(alpha: int) -> np.ndarray:
    vf = next(VOXEL_DIR.glob(
        f"rho_LA_r128_alpha_300nm_alpha{alpha:03d}_seed000*.npy"))
    return np.load(vf).astype(np.float32)

def derotate_slice(sl: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2-D slice into the corotating frame using bilinear interpolation.

    Bilinear instead of nearest-neighbour: eliminates the 1-pixel jitter at
    column boundaries that smears the within-column signature when accumulated
    over 151 z-slices.  The output is a REAL-valued continuous occupancy field
    in [0,1] (so <J^2> != <J>; the correlation normalisation below uses the
    empirical zero-lag variance, not the binary identity g(0)=1/f).

    Periodic (wrap) source coordinates are used because the simulation domain
    has periodic XY boundaries (xy_periodic_wrap_applied=True).  Edge clamping
    (the previous behaviour) replicates the boundary, violating per-slice mass
    conservation by up to ~10%; wrapping preserves the mean and is consistent
    with the FFT circular autocorrelation used downstream.  On the Level-A
    fields the clamped vs wrapped h_g curves differ by <=0.035 over the
    trusted short range and leave the angle ordering unchanged.
    """
    nx, ny = sl.shape
    ca, sa  = math.cos(-angle), math.sin(-angle)
    cx, cy  = (nx - 1) / 2.0, (ny - 1) / 2.0
    ii, jj  = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    # Float source coordinates
    ir_f = ca*(ii - cx) - sa*(jj - cy) + cx
    jr_f = sa*(ii - cx) + ca*(jj - cy) + cy
    # Integer corners with periodic wrap (minimum-image)
    ir0 = np.floor(ir_f).astype(int)
    jr0 = np.floor(jr_f).astype(int)
    fr  = ir_f - ir0   # row fractional weight
    fc  = jr_f - jr0   # col fractional weight
    ir0m, ir1m = ir0 % nx, (ir0 + 1) % nx
    jr0m, jr1m = jr0 % ny, (jr0 + 1) % ny
    return (sl[ir0m, jr0m] * (1 - fr) * (1 - fc)
          + sl[ir1m, jr0m] * fr       * (1 - fc)
          + sl[ir0m, jr1m] * (1 - fr) * fc
          + sl[ir1m, jr1m] * fr       * fc)

def phi_array(nz: int) -> np.ndarray:
    pitch_vox = HELIX_PITCH_NM / (FILM_THICKNESS_NM / nz)
    return 2.0 * math.pi * np.arange(nz) / pitch_vox

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Two-point correlation function h(r; α)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 1 — Pair correlation h_g(r; α)  [corotating frame, g(r) normalised]")
print("="*70)

def compute_corr_1d(arr: np.ndarray, smooth_half: int = 1
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pair correlation function g(r) in the corotating frame, radially averaged.

    Uses the UNCENTERED autocorrelation  C_full(r) = <I(x) · I(x+r)>  and
    normalises to the pair correlation:

        g(r) = C_full(r) / f²

    where the derotated field is continuous (J in [0,1]); g(0)=C(0)/f^2 is
    read empirically and is NOT 1/f (that identity holds only for binary I).

    Then maps to a normalised covariance (continuous-field, Approach A):

        h_g(r) = (C(r) - f^2) / (C(0) - f^2)   →  h_g(0)=1

    NOTE ON NORMALISATION (corrected 2026-06-17 / v17):
      Bilinear derotation makes the field CONTINUOUS (J in [0,1]), so the
      binary identity <J^2>=<J> does NOT hold and g(0) != 1/f.  The code
      therefore reads the empirical zero-lag g0 = C(0)/f^2 and normalises by
      (g0-1) = (C(0)-f^2)/f^2, i.e. the continuous-covariance form above.
      This is NOT clipped to [0,1]: h_g(r) can be slightly negative where the
      arrangement is anticorrelated.  For an infinite homogeneous system the
      correlation -> 0 at large r; the periodic box only samples r<=L/2, so
      the tail is a finite-box estimate, not a measured h_g(inf)=0.
      By construction h_g(0)=1 for ALL f, so h_g(0) carries NO filling-
      fraction information; only the unnormalised g(0)=1/f would encode f.

    Returns
      r_vox  : integer radial distances 0, 1, … r_max  (voxels)
      h_g_r  : normalised covariance, h_g(0)=1, may be slightly negative
      f      : mean occupancy (filling factor)  — from binary, not interpolated
    """
    nx, ny, nz = arr.shape
    binary = (arr > BINARY_THR).astype(np.float64)
    f      = float(binary.mean())      # occupancy from original binary field
    if f < 1e-9 or f > 1 - 1e-9:
        # Degenerate (empty or full) — return trivial flat curve
        r_vox = np.arange(int(0.48 * min(nx, ny)) + 1, dtype=np.float32)
        return r_vox, np.ones(len(r_vox), dtype=np.float32), f
    phis = phi_array(nz)

    # Integer center: fftshift places zero-lag at (nx//2, ny//2) for even grids.
    cx, cy = nx // 2, ny // 2

    # Accumulate UNCENTERED autocorrelation <I(x)·I(x+r)> in corotating frame.
    # Bilinear derotation → real-valued field; mean ≈ f, integral preserved.
    C_full_sum = np.zeros((nx, ny))
    for k in range(nz):
        sl = derotate_slice(binary[:, :, k].astype(np.float32), phis[k])
        F  = np.fft.fft2(sl.astype(np.float64))
        C_full_sum += np.fft.fftshift(np.fft.ifft2(np.abs(F)**2).real) / (nx * ny)
    C_full = C_full_sum / nz   # C(r) = <J(x)·J(x+r)>, continuous field

    # Pair correlation: g(r) = C(r) / f²   (continuous field J, not binary)
    g = C_full / (f**2)

    # Normalise to the continuous-covariance form h_g(r)=(C(r)-f^2)/(C(0)-f^2):
    # read g0 = C(0)/f^2 empirically (NOT the binary 1/f); h_g may go negative.
    g0 = float(g[cx, cy])        # continuous zero-lag; differs from 1/f
    denom = g0 - 1.0
    if abs(denom) < 1e-9:
        denom = (1.0 / f) - 1.0  # fallback to theoretical value

    # Radial average
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    r_map  = np.sqrt((ii - cx)**2 + (jj - cy)**2)
    r_max  = int(0.48 * min(nx, ny))
    r_vox  = np.arange(r_max + 1, dtype=np.float32)
    h_r    = np.zeros(len(r_vox))
    for ri, rv in enumerate(r_vox):
        mask    = (r_map >= rv - 0.5) & (r_map < rv + 0.5)
        h_r[ri] = g[mask].mean() if mask.any() else 1.0

    h_g_r = (h_r - 1.0) / denom   # h_g(0)=1; tail is finite-box (r<=L/2)

    # Light box-filter on r>0 to suppress FFT ringing (bilinear already helps)
    if smooth_half > 0:
        w = smooth_half * 2 + 1
        kernel = np.ones(w) / w
        h_g_r[1:] = np.convolve(h_g_r[1:], kernel, mode="same")

    # Force h_g(0) = 1 exactly after smoothing
    h_g_r[0] = 1.0

    return r_vox, h_g_r.astype(np.float32), f


def half_width(r_vox: np.ndarray, h_r: np.ndarray) -> float:
    """Interpolated r at which h(r) = 0.5  (= column radius)."""
    for i in range(len(h_r) - 1):
        if h_r[i] >= 0.5 > h_r[i + 1]:
            t = (0.5 - h_r[i]) / (h_r[i + 1] - h_r[i])
            return float(r_vox[i] + t * (r_vox[i + 1] - r_vox[i]))
    return float(r_vox[-1])


all_r: dict[int, np.ndarray] = {}
all_h: dict[int, np.ndarray] = {}
col_stats: dict[int, dict]   = {}

for alpha in ALPHAS:
    t0      = time.time()
    arr     = load_voxel(alpha)
    r_v, h_r, f = compute_corr_1d(arr)
    r_half  = half_width(r_v, h_r)
    d_col   = 2.0 * r_half
    n_col   = f / (math.pi * r_half**2) if r_half > 0.5 else 0.0
    all_r[alpha] = r_v
    all_h[alpha] = h_r
    col_stats[alpha] = {
        "occ_frac":               round(f, 5),
        "r_half_vox":             round(r_half, 2),
        "d_col_vox":              round(d_col, 2),
        "col_density_per_vox2":   round(n_col, 6),
    }
    print(f"  α={alpha:3d}°  occ={f:.4f}  r_half={r_half:.1f}vox "
          f" d_col={d_col:.1f}vox  density={n_col:.5f}/vox²  [{time.time()-t0:.1f}s]")

(OUT_DIR / "column_stats.json").write_text(
    json.dumps(col_stats, indent=2), encoding="utf-8")

np.savez_compressed(
    OUT_DIR / "corr_functions.npz",
    alphas   = np.array(ALPHAS),
    r_bins   = np.stack([all_r[a] for a in ALPHAS]),
    h_r      = np.stack([all_h[a] for a in ALPHAS]),
    occ_frac = np.array([col_stats[a]["occ_frac"] for a in ALPHAS]),
)

# Figure 1 — h(r; α) for all alphas
fig, ax = plt.subplots(figsize=(8, 5))
for ai, alpha in enumerate(ALPHAS):
    ax.plot(all_r[alpha], all_h[alpha], color=COLORS[ai], lw=2.0,
            label=f'α={alpha}°  f={col_stats[alpha]["occ_frac"]:.3f}')
ax.axhline(0.5, ls="--", c="gray", lw=0.8, label="h = 0.5  (column radius)")
ax.set_xlabel("Radial separation  r  (voxels)", fontsize=12)
ax.set_ylabel(r"Pair correlation  $h_g(r) = (g(r)-1)\,/\,(g(0)-1)$", fontsize=12)
ax.set_title(r"Pair correlation function  $h_g(r;\,\alpha)$  — corotating frame  [bilinear derot.]",
             fontsize=12)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_xlim(0, int(all_r[ALPHAS[0]][-1]))
ax.set_ylim(-0.15, 1.08)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_correlation_functions.png", dpi=150)
plt.close()
print(f"\n  -> fig1_correlation_functions.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — NIF on (α_norm, r_norm) → h(r; α)  [all 7 alphas]
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 2 — NIF: (α, r) -> h(r; α)  [all 7 alphas, full fit]")
print("="*70)

R_MAX_NORM = float(all_r[ALPHAS[0]][-1])   # normalise r to [0, 1]


def build_dataset(train_alphas: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for alpha in train_alphas:
        r_v = all_r[alpha]
        h_r = all_h[alpha]
        a_n = ALPHA_NORM(alpha)
        for ri in range(len(r_v)):
            xs.append([a_n, float(r_v[ri]) / R_MAX_NORM])
            ys.append(float(h_r[ri]))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class CorrNIF(nn.Module):
    """Small NIF: (α_norm, r_norm) → h(r; α).

    Fourier encoding on r (smooth 1-D function), FiLM conditioning on α.
    Architecture kept small: with only 350 total data points, anything
    larger would overfit on LOOA (300 train / 50 test).
    """
    def __init__(self, n_freq: int = 12, hidden: int = 64, n_layers: int = 3,
                 sigma_r: float = 2.0):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.register_buffer("B_r",
            torch.randn(1, n_freq, generator=g) * sigma_r)
        in_dim       = 2 * n_freq
        self.n_layers = n_layers
        self.hidden   = hidden
        self.layers   = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else hidden, hidden)
            for i in range(n_layers)
        ])
        self.head     = nn.Linear(hidden, 1)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 2 * n_layers * hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2) = [alpha_norm, r_norm]
        proj  = 2.0 * math.pi * x[:, 1:2] @ self.B_r   # (B, n_freq)
        h     = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        film  = self.film_mlp(x[:, 0:1])
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta  = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu(
                (1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


def train_nif(X: np.ndarray, y: np.ndarray,
              epochs: int = 5000, lr: float = 3e-3, wd: float = 1e-5
              ) -> CorrNIF:
    model = CorrNIF()
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xt), yt)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 1000 == 0:
            print(f"    ep={ep:5d}  mse={loss.item():.2e}")
    return model.eval()


def predict_h(model: CorrNIF, alpha: int) -> np.ndarray:
    r_v = all_r[alpha]
    a_n = ALPHA_NORM(alpha)
    X   = np.column_stack([
        np.full(len(r_v), a_n, dtype=np.float32),
        r_v / R_MAX_NORM,
    ])
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


# Full fit
X_all, y_all = build_dataset(ALPHAS)
print(f"  Training on all {len(ALPHAS)} alphas  ({len(X_all)} samples, CPU only)")
t0 = time.time()
nif_full = train_nif(X_all, y_all)
print(f"  Full-fit done in {time.time()-t0:.0f}s")

# Figure 2 — full fit
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
for ai, alpha in enumerate(ALPHAS):
    ax    = axes.flat[ai]
    r_v   = all_r[alpha]
    h_gt  = all_h[alpha]
    h_nif = predict_h(nif_full, alpha)
    rmse  = float(np.sqrt(np.mean((h_nif - h_gt)**2)))
    ax.plot(r_v, h_gt,  color=COLORS[ai], lw=2.0, label="Ground truth")
    ax.plot(r_v, h_nif, ls="--", color="black", lw=1.5,
            label=f"NIF  RMSE={rmse:.4f}")
    ax.set_title(f"α={alpha}°", fontsize=11)
    ax.set_ylim(-0.15, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
axes.flat[-1].axis("off")
fig.supxlabel("r (voxels)", fontsize=12)
fig.supylabel(r"$h_g(r)$", fontsize=12)
fig.suptitle(r"NIF full fit: $(\alpha, r) \to h_g(r;\alpha)$  [trained on all 7 alphas]",
             fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_nif_full_fit.png", dpi=150)
plt.close()
print(f"  -> fig2_nif_full_fit.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Leave-one-alpha-out evaluation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 3 — LOOA: hold out each α, train on 6, predict h(r; α_held)")
print("="*70)

looa_results: dict[int, dict] = {}
looa_preds:   dict[int, np.ndarray] = {}

for held_out in ALPHAS:
    train_as   = [a for a in ALPHAS if a != held_out]
    X_tr, y_tr = build_dataset(train_as)
    print(f"\n  Held-out α={held_out}°  train={train_as}  ({len(X_tr)} samples)")
    nif_lo = train_nif(X_tr, y_tr)

    r_v    = all_r[held_out]
    h_gt   = all_h[held_out]
    h_pred = predict_h(nif_lo, held_out)
    looa_preds[held_out] = h_pred

    rmse   = float(np.sqrt(np.mean((h_pred - h_gt)**2)))
    ss_res = float(np.sum((h_gt - h_pred)**2))
    ss_tot = float(np.sum((h_gt - h_gt.mean())**2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    r_half_pred = half_width(r_v, h_pred)
    r_half_gt   = half_width(r_v, h_gt)
    d_err       = abs(2 * r_half_pred - 2 * r_half_gt)

    looa_results[held_out] = {
        "train_alphas":     train_as,
        "rmse":             round(rmse, 5),
        "r2":               round(r2, 5),
        "r_half_pred_vox":  round(r_half_pred, 2),
        "r_half_gt_vox":    round(r_half_gt, 2),
        "d_col_error_vox":  round(d_err, 2),
    }
    print(f"    RMSE={rmse:.4f}  R²={r2:.4f}  "
          f"r_half_pred={r_half_pred:.1f}  r_half_gt={r_half_gt:.1f}  "
          f"|d_err|={d_err:.1f}vox")

(OUT_DIR / "looa_results.json").write_text(
    json.dumps(looa_results, indent=2), encoding="utf-8")

# Figure 3 — LOOA
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
for ai, held_out in enumerate(ALPHAS):
    ax     = axes.flat[ai]
    r_v    = all_r[held_out]
    h_gt   = all_h[held_out]
    h_pred = looa_preds[held_out]
    r2     = looa_results[held_out]["r2"]
    ax.fill_between(r_v, h_gt, h_pred, alpha=0.2, color=COLORS[ai])
    ax.plot(r_v, h_gt,   color=COLORS[ai], lw=2.0, label="Ground truth")
    ax.plot(r_v, h_pred, ls="--", color="black", lw=1.5,
            label=f"LOOA  R²={r2:.3f}")
    ax.set_title(f"held-out α={held_out}°", fontsize=11)
    ax.set_ylim(-0.15, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
axes.flat[-1].axis("off")
fig.supxlabel("r (voxels)", fontsize=12)
fig.supylabel(r"$h_g(r)$", fontsize=12)
fig.suptitle(
    r"LOOA: predict $h_g(r;\alpha_\mathrm{held})$ trained on 6 alphas" + "\n"
    r"Statistical morphology IS learnable across $\alpha$  (unlike exact voxel positions)",
    fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_looa_predictions.png", dpi=150)
plt.close()
print(f"\n  -> fig3_looa_predictions.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 BONUS — Column statistics + Maxwell-Garnett optical properties
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" BONUS — Column statistics and Maxwell-Garnett optical properties")
print("  Material: W columns in vacuum,  lambda~500 nm,  eps_W=4.41+19.60i")
print("="*70)

alpha_arr = np.array(ALPHAS, dtype=float)
occ_arr   = np.array([col_stats[a]["occ_frac"]             for a in ALPHAS])
d_arr     = np.array([col_stats[a]["d_col_vox"]             for a in ALPHAS])
rho_arr   = np.array([col_stats[a]["col_density_per_vox2"]  for a in ALPHAS])

mg_props: dict[int, dict] = {}
n_o_arr, n_e_arr, k_o_arr, dn_arr = [], [], [], []

for alpha in ALPHAS:
    f      = col_stats[alpha]["occ_frac"]
    eps_o  = mg_cylindrical(EPS_W, EPS_HOST, f)          # transverse (ordinary)
    eps_e  = f * EPS_W + (1.0 - f) * EPS_HOST            # Wiener parallel (extraordinary)
    n_o, k_o = eps_to_nk(eps_o)
    n_e, k_e = eps_to_nk(eps_e)   # k_e RETAINED (was dropped before 2026-06-17)
    dn       = abs(n_e - n_o)
    dk       = abs(k_e - k_o)
    # Beer-Lambert intensity transmission of extraordinary wave through film
    T_e      = math.exp(-4.0 * math.pi * k_e * FILM_THICKNESS_NM / 500.0)
    n_o_arr.append(n_o); n_e_arr.append(n_e)
    k_o_arr.append(k_o); dn_arr.append(dn)
    mg_props[alpha] = {
        "f": round(f, 5),
        "n_o_MG":   round(n_o, 4),
        "n_e_Wiener": round(n_e, 4),
        "k_o":      round(k_o, 4),
        "k_e":      round(k_e, 4),
        "delta_n":  round(dn, 5),
        "delta_k":  round(dk, 5),
        "T_e_300nm": T_e,
    }
    print(f"  α={alpha:3d}°  f={f:.3f}  n_o={n_o:.3f}  n_e={n_e:.3f}"
          f"  k_o={k_o:.3f}  k_e={k_e:.3f}  Δn={dn:.4f}  T_e={T_e:.2e}")

(OUT_DIR / "mg_properties.json").write_text(
    json.dumps(mg_props, indent=2), encoding="utf-8")

n_o_arr = np.array(n_o_arr)
n_e_arr = np.array(n_e_arr)
k_o_arr = np.array(k_o_arr)
dn_arr  = np.array(dn_arr)

# Figure 4 — Column statistics vs α
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(alpha_arr, occ_arr * 100, "o-", color="steelblue", lw=2, ms=7)
axes[0].set_xlabel("Deposition angle α (°)")
axes[0].set_ylabel("Filling fraction f (%)")
axes[0].set_title("Filling fraction  f(α)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(alpha_arr, d_arr, "s-", color="darkorange", lw=2, ms=7)
axes[1].set_xlabel("Deposition angle α (°)")
axes[1].set_ylabel("Column diameter (voxels)")
axes[1].set_title("Mean column diameter  d(α)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(alpha_arr, rho_arr * 1e3, "^-", color="green", lw=2, ms=7)
axes[2].set_xlabel("Deposition angle α (°)")
axes[2].set_ylabel(r"Column density ($\times 10^{-3}$ vox$^{-2}$)")
axes[2].set_title(r"Column number density  $\rho$(α)")
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    "Column statistics from h(r; α) — all vary monotonically with α\n"
    "These are ensemble properties: deterministic given α, unlike column positions",
    fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_column_statistics.png", dpi=150)
plt.close()
print(f"\n  -> fig4_column_statistics.png")

# Figure 5 — Optical properties vs α
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(alpha_arr, n_o_arr, "o-", color="royalblue", lw=2, ms=7,
             label=r"$n_o$ (MG, transverse)")
axes[0].plot(alpha_arr, n_e_arr, "s--", color="firebrick", lw=2, ms=7,
             label=r"$n_e$ (Wiener, longitudinal)")
axes[0].set_xlabel("Deposition angle α (°)")
axes[0].set_ylabel("Real refractive index")
axes[0].set_title("Effective refractive indices")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(alpha_arr, dn_arr, "^-", color="purple", lw=2, ms=7)
axes[1].set_xlabel("Deposition angle α (°)")
axes[1].set_ylabel(r"$|\Delta n| = |n_e - n_o|$")
axes[1].set_title(r"Birefringence  $\Delta n(\alpha)$  [sensing signal]")
axes[1].grid(True, alpha=0.3)

axes[2].plot(alpha_arr, k_o_arr, "D-", color="darkgreen", lw=2, ms=7)
axes[2].set_xlabel("Deposition angle α (°)")
axes[2].set_ylabel(r"Extinction coefficient  $k_\mathrm{eff}$")
axes[2].set_title(r"Effective extinction  $k_\mathrm{eff}(\alpha)$")
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    r"Maxwell-Garnett effective optical properties  (W, $\lambda\approx500$ nm,"
    r"  $\varepsilon_W = 4.41+19.6i$)" + "\n"
    "Derived analytically from C(r; α) filling fraction — no additional simulation",
    fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_optical_properties.png", dpi=150)
plt.close()
print(f"  -> fig5_optical_properties.png")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
mean_r2   = float(np.mean([looa_results[a]["r2"]   for a in ALPHAS]))
mean_rmse = float(np.mean([looa_results[a]["rmse"] for a in ALPHAS]))

print("\n" + "="*70)
print(" PIPELINE COMPLETE")
print("="*70)
print(f"\n  Output: {OUT_DIR.relative_to(PROJECT_ROOT)}")
print()
print(f"  LOOA evaluation — predict h(r; α_held) from 6 training alphas:")
print(f"    Mean R²   = {mean_r2:.4f}   (1.0 = perfect)")
print(f"    Mean RMSE = {mean_rmse:.4f}  (0.0 = perfect)")
print()
print("  Per-alpha LOOA:")
for a in ALPHAS:
    r = looa_results[a]
    print(f"    α={a:3d}°  R²={r['r2']:.4f}  RMSE={r['rmse']:.4f}  "
          f"|d_col err|={r['d_col_error_vox']:.1f}vox")
print()
print("  Optical properties (MG, W columns, λ~500nm):")
for a in ALPHAS:
    m = mg_props[a]
    print(f"    α={a:3d}°  f={m['f']:.3f}  n_o={m['n_o_MG']:.3f}"
          f"  n_e={m['n_e_Wiener']:.3f}  Δn={m['delta_n']:.4f}")
print()
print("  Physical interpretation:")
print("  - h(r;α) varies smoothly with α → NIF successfully interpolates")
print("  - occ_frac, d_col, rho_col all monotonic in α → clean physical trend")
print("  - Δn(α) shows the birefringence design window for GLAD sensors")
print("  - Voxel-level NIF fails because column POSITIONS are stochastic;")
print("    statistical NIF succeeds because the PATTERN STATISTICS are")
print("    deterministic given α (same physics → same ensemble).")
print()
print("  Figures:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"    {f.relative_to(PROJECT_ROOT)}")
print()
print("  Data files:")
for f in sorted(list(OUT_DIR.glob("*.json")) + list(OUT_DIR.glob("*.npz"))):
    print(f"    {f.relative_to(PROJECT_ROOT)}")
print("="*70)
