"""
LENS1 WP1 -- local identifiability / conditioning of the CorrNIF inverse map.
===========================================================================
Object: F : alpha  ->  h_g(r; alpha), the two-point pair-correlation descriptor,
emulated by the frozen, differentiable CorrNIF (FiLM-conditioned MLP). P1/P2 show
alpha is recoverable from h_g by gradient descent through CorrNIF. This script
quantifies *where that inversion is well-conditioned and where it degrades*.

Method (mirrors run_corrnif_gradient_inverse_demo.py exactly -- same data, same
architecture, same normalisation, same seed, so the trained model is identical):
  1. Train CorrNIF once on the 7 campaign angles (65..89 deg), freeze.
  2. J(alpha)_i = d h_g(r_i; alpha) / d alpha  by autodiff, for every r bin,
     on a dense alpha grid AND at the 7 training angles.
  3. Report  ||J||_2(alpha)  (descriptor change per degree) and its RMS-per-bin
     form, plus a finite-difference cross-check across the 7 training angles.
  4. Achievable alpha precision  sigma_alpha(alpha) = sigma_h / ||J||_2  (MLE
     delta method, 1-D parameter, iid per-bin descriptor noise sigma_h);
     report sigma_alpha at the range endpoints for a few plausible sigma_h.
  5. Test the pre-registered prediction: ||J|| -> 0 near alpha = 87-89 deg.

Outputs (no GPU, ~15 s CPU):
  runs/lens1_wp1/lens1_wp1_corrnif_jacobian.json
  runs/lens1_wp1/lens1_wp1_corrnif_jacobian.png
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


def _find_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("Cannot find project root")


PROJECT_ROOT = _find_root()
PIPE_DIR = PROJECT_ROOT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline"
OUT_DIR = PROJECT_ROOT / "02_PINN_NIF/pinn_training/runs/lens1_wp1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [65, 70, 75, 80, 85, 87, 89]
ALPHA_NORM = lambda a: (a - 74.5) / 14.5
ALPHA_DENORM = lambda an: an * 14.5 + 74.5
DALPHA_DANORM = 14.5  # d(alpha_deg)/d(alpha_norm); chain rule for autodiff below

torch.manual_seed(0)
np.random.seed(0)

npz = np.load(PIPE_DIR / "corr_functions.npz")
stored_alphas = [int(a) for a in npz["alphas"]]
assert stored_alphas == ALPHAS, f"Alpha mismatch: {stored_alphas} vs {ALPHAS}"
all_r = {a: npz["r_bins"][i].astype(np.float32) for i, a in enumerate(ALPHAS)}
all_h = {a: npz["h_r"][i].astype(np.float32) for i, a in enumerate(ALPHAS)}
R_MAX_NORM = float(all_r[ALPHAS[0]][-1])


class CorrNIF(nn.Module):
    def __init__(self, n_freq: int = 12, hidden: int = 64, n_layers: int = 3,
                 sigma_r: float = 2.0):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.register_buffer("B_r", torch.randn(1, n_freq, generator=g) * sigma_r)
        in_dim = 2 * n_freq
        self.n_layers = n_layers
        self.hidden = hidden
        self.layers = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else hidden, hidden) for i in range(n_layers)
        ])
        self.head = nn.Linear(hidden, 1)
        self.film_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 2 * n_layers * hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x[:, 1:2] @ self.B_r
        h = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        film = self.film_mlp(x[:, 0:1])
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


def build_dataset(alphas):
    xs, ys = [], []
    for a in alphas:
        for ri, rv in enumerate(all_r[a]):
            xs.append([ALPHA_NORM(a), float(rv) / R_MAX_NORM])
            ys.append(float(all_h[a][ri]))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_nif(X, y, epochs: int = 5000, lr: float = 3e-3, wd: float = 1e-5) -> CorrNIF:
    model = CorrNIF()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


print("Training CorrNIF on 7 angles (identical recipe to the inversion demo) ...")
t0 = time.time()
Xall, yall = build_dataset(ALPHAS)
nif = train_nif(Xall, yall)
for p in nif.parameters():
    p.requires_grad_(False)
print(f"  done in {time.time() - t0:.1f}s")

# common r grid (model's own, as the inversion uses)
r_ref = all_r[65]
N_R = len(r_ref)
r_query_norm = torch.tensor((r_ref / R_MAX_NORM).astype(np.float32)).unsqueeze(-1)


def hg_of_alpha_norm(an_scalar: torch.Tensor) -> torch.Tensor:
    """h_g(r_ref; alpha) as a length-N_R tensor, differentiable in an_scalar."""
    a_col = an_scalar.expand(N_R).unsqueeze(-1)
    X = torch.cat([a_col, r_query_norm], dim=-1)
    return nif(X)


def jacobian_norm_at(alpha_deg: float):
    """||dh_g/d alpha||_2 (per-degree) and RMS-per-bin at one alpha."""
    an = torch.tensor(ALPHA_NORM(alpha_deg), dtype=torch.float32, requires_grad=True)
    h = hg_of_alpha_norm(an)                       # (N_R,)
    J_norm = torch.zeros(N_R)
    for i in range(N_R):
        g, = torch.autograd.grad(h[i], an, retain_graph=(i < N_R - 1))
        J_norm[i] = g
    J_deg = (J_norm / DALPHA_DANORM).detach().numpy()   # d h_g(r_i) / d alpha_deg
    l2 = float(np.sqrt(np.sum(J_deg ** 2)))
    rms = float(np.sqrt(np.mean(J_deg ** 2)))
    return l2, rms, J_deg


# dense sweep
grid = np.linspace(63.0, 89.0, 131)
l2s, rmss = [], []
for a in grid:
    l2, rms, _ = jacobian_norm_at(float(a))
    l2s.append(l2); rmss.append(rms)
l2s = np.array(l2s); rmss = np.array(rmss)

# at the 7 training angles + finite-difference cross-check
fd = {}
Xtr = {a: torch.tensor(np.stack([np.full(N_R, ALPHA_NORM(a), np.float32),
                                 (r_ref / R_MAX_NORM).astype(np.float32)], axis=1))
       for a in ALPHAS}
with torch.no_grad():
    hpred = {a: nif(Xtr[a]).numpy() for a in ALPHAS}
for k, a in enumerate(ALPHAS):
    ad, au = ALPHAS[max(0, k - 1)], ALPHAS[min(len(ALPHAS) - 1, k + 1)]
    if au == ad:
        continue
    dh = (hpred[au] - hpred[ad]) / (au - ad)
    fd[a] = dict(l2=float(np.sqrt(np.sum(dh ** 2))), rms=float(np.sqrt(np.mean(dh ** 2))))

auto_at_train = {a: jacobian_norm_at(a)[:2] for a in ALPHAS}

# conditioning: MLE delta-method precision  sigma_alpha(alpha) = sigma_h / ||J||_2  (deg)
# (1-D parameter, iid per-bin descriptor noise sigma_h; Var(alpha_hat) = sigma_h^2 / sum_i J_i^2)
peak = float(np.max(l2s))
a_peak = float(grid[int(np.argmax(l2s))])
mn = float(np.min(l2s))
a_min = float(grid[int(np.argmin(l2s))])
# alpha where ||J|| falls to 20% of peak (convention-light "inversion gets shaky" mark)
below = grid[l2s < 0.2 * peak]
a_20pct = float(below.min()) if below.size and below.min() > a_peak else None
prec = {}
for sig in (0.005, 0.01, 0.02):
    sa = sig / np.clip(l2s, 1e-12, None)
    hit = grid[sa > 1.0]
    prec[f"sigma_h={sig}"] = {
        "sigma_alpha_deg_at_grid": sa.round(4).tolist(),
        "sigma_alpha_deg_at_63": float(sa[0]), "sigma_alpha_deg_at_89": float(sa[-1]),
        "first_alpha_with_sigma_alpha_gt_1deg": (float(hit.min()) if hit.size else None),
    }

pred_holds = bool(np.mean(l2s[grid >= 87]) < 0.5 * np.mean(l2s[(grid >= 70) & (grid <= 80)]))

results = {
    "object": "F: alpha -> h_g(r; alpha), CorrNIF FiLM surrogate (frozen)",
    "training_alphas_deg": ALPHAS,
    "n_r_bins": N_R,
    "grid_alphas_deg": grid.round(3).tolist(),
    "J_l2_per_deg": l2s.round(6).tolist(),
    "J_rms_per_deg": rmss.round(6).tolist(),
    "J_l2_peak": peak,
    "alpha_at_peak_deg": a_peak,
    "J_l2_min": mn,
    "alpha_at_min_deg": a_min,
    "alpha_where_J_below_20pct_of_peak_deg": a_20pct,
    "autodiff_at_training_angles": {str(a): {"l2": v[0], "rms": v[1]} for a, v in auto_at_train.items()},
    "finite_difference_cross_check": {str(a): v for a, v in fd.items()},
    "precision_model": prec,
    "prediction_||J||->0_near_87-89deg_holds": pred_holds,
    "notes": (
        "1-D inverse problem, so cond(J^T J) = (||J||_2)^{-2}; the reported "
        "||J||_2(alpha) is the conditioning curve. sigma_alpha ~ sigma_h*sqrt(N_R)/||J|| "
        "is a first-order (delta-method) achievable-precision estimate assuming iid "
        "per-bin descriptor noise sigma_h. Model + data + seed identical to "
        "run_corrnif_gradient_inverse_demo.py."
    ),
}
with open(OUT_DIR / "lens1_wp1_corrnif_jacobian.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- figure ----
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(grid, l2s, "-", color="#c62828", lw=1.8, label=r"$\|\partial h_g/\partial\alpha\|_2$ (autodiff)")
ax.plot(list(auto_at_train.keys()), [v[0] for v in auto_at_train.values()], "o",
        color="#c62828", ms=5, label="at training angles")
ax.plot(list(fd.keys()), [v["l2"] for v in fd.values()], "s", mfc="none", mec="#1565c0",
        mew=1.4, ms=6, label="finite difference (cross-check)")
if a_20pct:
    ax.axvline(a_20pct, color="0.4", ls="--", lw=1.0)
    ax.text(a_20pct + 0.3, ax.get_ylim()[1] * 0.9,
            f"$\\|J\\|<20\\%$ peak\n$\\alpha\\approx{a_20pct:.0f}^\\circ$", fontsize=7, va="top")
ax.set_xlabel(r"deposition angle $\alpha$ (deg)")
ax.set_ylabel(r"$\|\partial h_g/\partial\alpha\|_2$  (descriptor change per degree)")
ax.set_title("CorrNIF inverse-map conditioning vs deposition angle", fontsize=9)
ax.grid(True, ls="--", alpha=0.3)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "lens1_wp1_corrnif_jacobian.png", dpi=200)
fig.savefig(OUT_DIR / "lens1_wp1_corrnif_jacobian.pdf")

print(f"||J||_2 peak {peak:.4g} at alpha={a_peak:.1f} deg; "
      f"drops below 20% of peak at alpha={a_20pct}; "
      f"prediction ||J||->0 near 87-89 holds: {pred_holds}")
print("wrote", OUT_DIR / "lens1_wp1_corrnif_jacobian.json")
