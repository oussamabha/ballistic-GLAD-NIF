"""
alpha72_prospective_validation.py
==================================
Prospective prediction of h_g(r; 72°) using CorrNIF trained on the 7
existing alpha values [65, 70, 75, 80, 85, 87, 89].

α=72° is a genuine interpolation point (not in the training set, lying
between 70° and 75°).  After the GLAD simulation for α=72° completes and
is voxelized, set SIMULATION_DONE=True to include ground-truth comparison.

Outputs (written to morphology_pipeline/):
  figures/fig8_alpha72_prospective.png  — main prospective prediction figure
  alpha72_prospective_results.json      — CorrNIF, linear, NN predictions + (optionally) ground truth
"""
from __future__ import annotations
import json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
def _find_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("Cannot find project root")

PROJECT_ROOT  = _find_root()
VOXEL_DIR     = PROJECT_ROOT / "02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline/voxels"
OUT_DIR       = PROJECT_ROOT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline"
FIG_DIR       = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
ALPHAS         = [65, 70, 75, 80, 85, 87, 89]  # training set
TARGET_ALPHA   = 72                              # unseen prospective point
ALPHA_NORM     = lambda a: (a - 74.5) / 14.5
HELIX_PITCH_NM = 150.0
FILM_THICKNESS = 300.0
BINARY_THR     = 0.5

# ── load existing corr_functions ─────────────────────────────────────────────
print("Loading corr_functions.npz ...")
npz = np.load(OUT_DIR / "corr_functions.npz")
stored_alphas = [int(a) for a in npz["alphas"]]
assert stored_alphas == ALPHAS, f"Alpha mismatch: {stored_alphas} vs {ALPHAS}"

all_r = {a: npz["r_bins"][i].astype(np.float32)  for i, a in enumerate(ALPHAS)}
all_h = {a: npz["h_r"][i].astype(np.float32)     for i, a in enumerate(ALPHAS)}
occ   = {a: float(npz["occ_frac"][i])            for i, a in enumerate(ALPHAS)}

R_MAX_NORM = float(all_r[ALPHAS[0]][-1])
print(f"  7 alphas loaded.  R_MAX_NORM = {R_MAX_NORM:.1f} vox")

# ── CorrNIF (identical architecture to run_statistical_morphology_pipeline) ───
class CorrNIF(nn.Module):
    def __init__(self, n_freq: int = 12, hidden: int = 64, n_layers: int = 3,
                 sigma_r: float = 2.0):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.register_buffer("B_r", torch.randn(1, n_freq, generator=g) * sigma_r)
        in_dim        = 2 * n_freq
        self.n_layers = n_layers
        self.hidden   = hidden
        self.layers   = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else hidden, hidden)
            for i in range(n_layers)
        ])
        self.head    = nn.Linear(hidden, 1)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 2 * n_layers * hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj  = 2.0 * math.pi * x[:, 1:2] @ self.B_r
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


def build_dataset(alphas: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for a in alphas:
        for ri, rv in enumerate(all_r[a]):
            xs.append([ALPHA_NORM(a), float(rv) / R_MAX_NORM])
            ys.append(float(all_h[a][ri]))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_nif(X: np.ndarray, y: np.ndarray,
              epochs: int = 5000, lr: float = 3e-3, wd: float = 1e-5) -> CorrNIF:
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


def query_at_alpha(model: CorrNIF, alpha_float: float,
                   r_vox: np.ndarray) -> np.ndarray:
    a_n = ALPHA_NORM(alpha_float)
    X   = np.column_stack([
        np.full(len(r_vox), a_n, dtype=np.float32),
        r_vox.astype(np.float32) / R_MAX_NORM,
    ])
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


# ── train CorrNIF on all 7 training alphas ───────────────────────────────────
print("\nTraining CorrNIF on all 7 alphas ...")
t0   = time.time()
Xall, yall = build_dataset(ALPHAS)
nif  = train_nif(Xall, yall)
print(f"Training done in {time.time()-t0:.0f}s")

# ── prospective prediction at α=72° ──────────────────────────────────────────
r_ref = all_r[70]   # use α=70° r-grid (same for all alphas)

h_corrnif = query_at_alpha(nif, float(TARGET_ALPHA), r_ref)

# Linear interpolation baseline: h_lin(72) = 0.6 * h(70) + 0.4 * h(75)
w70 = (75 - TARGET_ALPHA) / (75 - 70)   # = 0.6
w75 = (TARGET_ALPHA - 70) / (75 - 70)   # = 0.4
h_linear = w70 * all_h[70] + w75 * all_h[75]

# Nearest-neighbour: α=70° (nearest to 72)
h_nn70 = all_h[70]

print(f"\nProspective prediction at α={TARGET_ALPHA}°  (between 70° and 75°)")
print(f"  CorrNIF h_g(r=1): {h_corrnif[1]:.4f}")
print(f"  Linear blend    : {h_linear[1]:.4f}")
print(f"  NN (α=70°)      : {h_nn70[1]:.4f}")

# ── ground-truth comparison (available after simulation completes) ───────────
GT_VOXEL_PATH = VOXEL_DIR / "rho_LA_helical_alpha_300nm_alpha072_seed000_300nm_helical_pitch150.npy"
SIMULATION_DONE = GT_VOXEL_PATH.exists()

h_gt72   = None
r2_corrnif = None
r2_linear  = None
r2_nn      = None

def compute_corr_1d_local(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Pair correlation h_g(r;α) in corotating frame (mirrors run_statistical_morphology_pipeline)."""
    nx, ny, nz = arr.shape
    binary = (arr > BINARY_THR).astype(np.float64)
    f = float(binary.mean())
    if f < 1e-9 or f > 1 - 1e-9:
        r_vox = np.arange(int(0.48 * min(nx, ny)) + 1, dtype=np.float32)
        return r_vox, np.ones(len(r_vox), dtype=np.float32), f
    pitch_vox = HELIX_PITCH_NM / (FILM_THICKNESS / nz)
    phis = 2.0 * math.pi * np.arange(nz) / pitch_vox
    cx, cy = nx // 2, ny // 2
    C_full_sum = np.zeros((nx, ny))
    for k in range(nz):
        angle = phis[k]
        ca, sa = math.cos(-angle), math.sin(-angle)
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        ir_f = ca*(ii - cx) - sa*(jj - cy) + cx
        jr_f = sa*(ii - cx) + ca*(jj - cy) + cy
        ir_f = np.clip(ir_f, 0.0, nx - 1.0)
        jr_f = np.clip(jr_f, 0.0, ny - 1.0)
        ir0 = np.clip(np.floor(ir_f).astype(int), 0, nx - 2)
        jr0 = np.clip(np.floor(jr_f).astype(int), 0, ny - 2)
        ir1, jr1 = ir0 + 1, jr0 + 1
        fr, fc = ir_f - ir0, jr_f - jr0
        sl = (binary[:, :, k].astype(np.float32)[ir0, jr0] * (1-fr) * (1-fc)
            + binary[:, :, k].astype(np.float32)[ir1, jr0] * fr * (1-fc)
            + binary[:, :, k].astype(np.float32)[ir0, jr1] * (1-fr) * fc
            + binary[:, :, k].astype(np.float32)[ir1, jr1] * fr * fc)
        F = np.fft.fft2(sl.astype(np.float64))
        C_full_sum += np.fft.fftshift(np.fft.ifft2(np.abs(F)**2).real) / (nx * ny)
    C_full = C_full_sum / nz
    g = C_full / (f**2)
    g0 = float(g[cx, cy])
    denom = g0 - 1.0
    if abs(denom) < 1e-9:
        denom = (1.0 / f) - 1.0
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    r_map = np.sqrt((ii - cx)**2 + (jj - cy)**2)
    r_max = int(0.48 * min(nx, ny))
    r_vox = np.arange(r_max + 1, dtype=np.float32)
    h_r = np.zeros(len(r_vox))
    for ri, rv in enumerate(r_vox):
        mask = (r_map >= rv - 0.5) & (r_map < rv + 0.5)
        h_r[ri] = g[mask].mean() if mask.any() else 1.0
    h_g_r = (h_r - 1.0) / denom
    kernel = np.ones(3) / 3
    h_g_r[1:] = np.convolve(h_g_r[1:], kernel, mode="same")
    h_g_r[0] = 1.0
    return r_vox, h_g_r.astype(np.float32), f


def half_width_local(r_vox, h_r):
    for i in range(len(h_r) - 1):
        if h_r[i] >= 0.5 > h_r[i + 1]:
            t = (0.5 - h_r[i]) / (h_r[i + 1] - h_r[i])
            return float(r_vox[i] + t * (r_vox[i + 1] - r_vox[i]))
    return float(r_vox[-1])


if SIMULATION_DONE:
    print("\nα=72° voxel found — computing ground truth h_g(r; 72°) ...")
    arr      = np.load(GT_VOXEL_PATH).astype(np.float32)
    r_v72, h_gt72_arr, f72 = compute_corr_1d_local(arr)
    h_gt72 = h_gt72_arr[:len(r_ref)]   # clip to same length as r_ref

    def r2(pred, gt):
        ss_res = np.sum((gt - pred)**2)
        ss_tot = np.sum((gt - gt.mean())**2)
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    r2_corrnif = round(float(r2(h_corrnif[:len(h_gt72)], h_gt72)), 5)
    r2_linear  = round(float(r2(h_linear[:len(h_gt72)],  h_gt72)), 5)
    r2_nn      = round(float(r2(h_nn70[:len(h_gt72)],    h_gt72)), 5)
    print(f"  R² CorrNIF : {r2_corrnif:.4f}")
    print(f"  R² Linear  : {r2_linear:.4f}")
    print(f"  R² NN(70°) : {r2_nn:.4f}")
else:
    print("\nα=72° voxel NOT yet available — showing prospective prediction only.")
    print(f"  Expected path: {GT_VOXEL_PATH}")

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors7 = plt.cm.Blues(np.linspace(0.25, 0.7, len(ALPHAS)))

# All 7 training curves (thin, light)
for ai, a in enumerate(ALPHAS):
    lw  = 2.2 if a in (70, 75) else 0.8
    ls  = "-"
    c   = colors7[ai]
    lab = f"α={a}°" if a in (70, 75) else None
    ax.plot(all_r[a], all_h[a], color=c, lw=lw, ls=ls, label=lab, zorder=2)

# CorrNIF prospective prediction
ax.plot(r_ref, h_corrnif, color="crimson", lw=2.5, ls="--",
        label=f"CorrNIF predict α=72°" + (f" (R²={r2_corrnif:.3f})" if r2_corrnif else " (prospective)"),
        zorder=5)

# Linear interpolation
ax.plot(r_ref, h_linear, color="darkorange", lw=1.5, ls=":",
        label=f"Linear blend 70°/75°" + (f" (R²={r2_linear:.3f})" if r2_linear else ""),
        zorder=4)

# Ground truth (if available)
if h_gt72 is not None:
    ax.plot(r_ref[:len(h_gt72)], h_gt72, color="black", lw=2.5, ls="-",
            label=f"Simulated α=72° (ground truth)", zorder=6)

ax.axhline(0.5, ls="--", c="gray", lw=0.8, alpha=0.5)
ax.set_xlabel("Radial separation  r  (voxels)", fontsize=12)
ax.set_ylabel(r"$h_g(r)$", fontsize=12)
title = (r"Prospective CorrNIF prediction at $\alpha=72°$" +
         ("\n(ground truth available)" if SIMULATION_DONE else
          "\n(simulation pending — prospective only)"))
ax.set_title(title, fontsize=12)
ax.set_xlim(0, int(r_ref[-1]))
ax.set_ylim(-0.15, 1.10)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_fig = FIG_DIR / "fig8_alpha72_prospective.png"
plt.savefig(out_fig, dpi=150)
plt.close()
print(f"\nSaved: {out_fig}")

# ── JSON results ─────────────────────────────────────────────────────────────
results = {
    "target_alpha":   TARGET_ALPHA,
    "training_alphas": ALPHAS,
    "simulation_done": SIMULATION_DONE,
    "r_bins":          r_ref.tolist(),
    "h_corrnif":       h_corrnif.tolist(),
    "h_linear_blend":  h_linear.tolist(),
    "h_nn_alpha70":    h_nn70.tolist(),
    "r2_corrnif":      r2_corrnif,
    "r2_linear":       r2_linear,
    "r2_nn70":         r2_nn,
    "h_gt72":          h_gt72.tolist() if h_gt72 is not None else None,
    "linear_weights":  {"alpha70": round(w70, 4), "alpha75": round(w75, 4)},
}
out_json = OUT_DIR / "alpha72_prospective_results.json"
out_json.write_text(json.dumps(results, indent=2))
print(f"Saved: {out_json}")
print("\nDone.")
