"""
CorrNIF gradient-based inverse-design demo.
============================================
Demonstrates the one claimed-but-previously-undemonstrated advantage of
CorrNIF over simple interpolation baselines: because CorrNIF is a smooth,
differentiable function of the deposition angle alpha (via the FiLM
conditioning MLP), alpha can be recovered from a *target* pair-correlation
curve h_g(r) by gradient descent on alpha directly, instead of a discrete
grid search or re-running the ballistic simulator at many candidate angles.

Task: given the dedicated-simulation ground-truth curve at alpha=72 deg
(genuinely held out of CorrNIF's training set, see
run_alpha72_prospective_validation.py), recover alpha=72 by gradient
descent through the FROZEN, already-trained CorrNIF model, starting from
several deliberately wrong initial guesses spanning the training range.

Correctness check: cross-validate the gradient-descent optimum against an
exhaustive fine-grid brute-force search over alpha in the training range,
confirming gradient descent converges to the same optimum (not just to
*some* stationary point).

Outputs:
  runs/corrnif_gradient_demo/corrnif_gradient_inverse_results.json
"""
from __future__ import annotations
import json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# This script lives at <repo_root>/code/nif_corrnif/. Data lives at
# <repo_root>/data/small_artifacts/ (see the repository README for what's included and why).
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_DIR  = REPO_ROOT / "data/small_artifacts"
OUT_DIR   = REPO_ROOT / "data/small_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS       = [65, 70, 75, 80, 85, 87, 89]   # CorrNIF training set (identical to companion scripts)
TARGET_ALPHA = 72                              # genuinely held-out, dedicated-simulation-confirmed
ALPHA_NORM   = lambda a: (a - 74.5) / 14.5     # identical normalisation to run_alpha72_prospective_validation.py
ALPHA_DENORM = lambda an: an * 14.5 + 74.5

torch.manual_seed(0)
np.random.seed(0)

# ── load data (same source as run_alpha72_prospective_validation.py) ────────
npz = np.load(PIPE_DIR / "corr_functions.npz")
stored_alphas = [int(a) for a in npz["alphas"]]
assert stored_alphas == ALPHAS, f"Alpha mismatch: {stored_alphas} vs {ALPHAS}"
all_r = {a: npz["r_bins"][i].astype(np.float32) for i, a in enumerate(ALPHAS)}
all_h = {a: npz["h_r"][i].astype(np.float32)    for i, a in enumerate(ALPHAS)}
R_MAX_NORM = float(all_r[ALPHAS[0]][-1])

with open(PIPE_DIR / "alpha72_prospective_results.json") as f:
    alpha72_ref = json.load(f)
assert alpha72_ref["simulation_done"], "Dedicated alpha=72 simulation not marked done"
r_bins_gt = np.array(alpha72_ref["r_bins"], dtype=np.float32)
h_gt72    = np.array(alpha72_ref["h_gt72"], dtype=np.float32)


# ── CorrNIF (identical architecture to run_alpha72_prospective_validation.py) ─
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
        self.head     = nn.Linear(hidden, 1)
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


def build_dataset(alphas):
    xs, ys = [], []
    for a in alphas:
        for ri, rv in enumerate(all_r[a]):
            xs.append([ALPHA_NORM(a), float(rv) / R_MAX_NORM])
            ys.append(float(all_h[a][ri]))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_nif(X, y, epochs: int = 5000, lr: float = 3e-3, wd: float = 1e-5) -> CorrNIF:
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
    return model.eval()


# ── train CorrNIF once on all 7 training alphas (frozen for the inverse step) ─
print("Training CorrNIF on all 7 alphas ...")
t0 = time.time()
Xall, yall = build_dataset(ALPHAS)
nif = train_nif(Xall, yall)
train_time_s = time.time() - t0
print(f"  training done in {train_time_s:.1f}s")
for p in nif.parameters():
    p.requires_grad_(False)

# Interpolate the target curve onto the model's own r-grid (r_ref, same as
# run_alpha72_prospective_validation.py uses for querying).
r_ref = all_r[70]
h_target = np.interp(r_ref, r_bins_gt, h_gt72).astype(np.float32)
r_query_norm = torch.tensor((r_ref / R_MAX_NORM).astype(np.float32))
h_target_t   = torch.tensor(h_target)


def loss_at_alpha_norm(alpha_norm_t: torch.Tensor) -> torch.Tensor:
    a_col = alpha_norm_t.expand(len(r_ref)).unsqueeze(-1)
    X = torch.cat([a_col, r_query_norm.unsqueeze(-1)], dim=-1)
    pred = nif(X)
    return torch.nn.functional.mse_loss(pred, h_target_t)


# ── (A) brute-force fine-grid ground truth for the inverse problem ──────────
grid_alphas = np.linspace(65.0, 89.0, 4801)  # 0.005 deg resolution
grid_losses = np.empty(len(grid_alphas))
with torch.no_grad():
    for i, a in enumerate(grid_alphas):
        an = torch.tensor(ALPHA_NORM(a), dtype=torch.float32)
        grid_losses[i] = loss_at_alpha_norm(an).item()
grid_best_idx   = int(np.argmin(grid_losses))
grid_best_alpha = float(grid_alphas[grid_best_idx])
grid_best_loss  = float(grid_losses[grid_best_idx])
print(f"Brute-force grid optimum: alpha={grid_best_alpha:.3f} deg "
      f"(loss={grid_best_loss:.3e})")

# ── (B) gradient descent from several deliberately-wrong starting points ────
init_alphas = [65.0, 70.0, 80.0, 85.0, 89.0]
n_steps     = 300
lr_grad     = 0.05

restarts = []
t0 = time.time()
for a0 in init_alphas:
    alpha_norm = torch.tensor(ALPHA_NORM(a0), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([alpha_norm], lr=lr_grad)
    traj_alpha, traj_loss = [], []
    for step in range(n_steps):
        opt.zero_grad()
        loss = loss_at_alpha_norm(alpha_norm)
        loss.backward()
        opt.step()
        traj_alpha.append(ALPHA_DENORM(alpha_norm.item()))
        traj_loss.append(loss.item())
    restarts.append({
        "init_alpha_deg":      a0,
        "recovered_alpha_deg": traj_alpha[-1],
        "final_loss":          traj_loss[-1],
        "trajectory_alpha":    traj_alpha,
        "trajectory_loss":     traj_loss,
    })
    print(f"  init={a0:5.1f} deg -> recovered={traj_alpha[-1]:.3f} deg "
          f"(final loss={traj_loss[-1]:.3e}, {n_steps} steps)")
grad_time_s = time.time() - t0

recovered = np.array([r["recovered_alpha_deg"] for r in restarts])
recovered_mean, recovered_std = float(recovered.mean()), float(recovered.std())
error_deg = recovered_mean - TARGET_ALPHA

results = {
    "target_alpha_deg":          TARGET_ALPHA,
    "training_alphas_deg":       ALPHAS,
    "n_gradient_steps_per_run":  n_steps,
    "n_restarts":                len(init_alphas),
    "init_alphas_deg":           init_alphas,
    "recovered_alpha_mean_deg":  recovered_mean,
    "recovered_alpha_std_deg":   recovered_std,
    "error_vs_true_deg":         error_deg,
    "grid_search_best_alpha_deg": grid_best_alpha,
    "grid_search_resolution_deg": float(grid_alphas[1] - grid_alphas[0]),
    "grid_vs_gradient_agreement_deg": abs(grid_best_alpha - recovered_mean),
    "corrnif_training_time_s":  train_time_s,
    "gradient_inverse_total_time_s": grad_time_s,
    "gradient_inverse_time_per_run_s": grad_time_s / len(init_alphas),
    "restarts": restarts,
    "grid_alphas_deg": grid_alphas.tolist(),
    "grid_losses": grid_losses.tolist(),
    "notes": (
        "Inverse problem: recover the deposition angle alpha that produced a "
        "given (held-out, dedicated-simulation-confirmed) pair-correlation "
        "curve h_g(r; alpha=72 deg), by gradient descent through the frozen, "
        "already-trained CorrNIF model, treating alpha as the only free "
        "parameter. This is the differentiability advantage claimed in the "
        "main text (Section on CorrNIF interpolation): CorrNIF's continuous, "
        "differentiable representation permits this gradient-based recovery, "
        "which nearest-neighbour, linear, and PCHIP baselines (non-analytic / "
        "non-differentiable in alpha) do not support in the same way."
    ),
}

out_path = OUT_DIR / "corrnif_gradient_inverse_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRecovered alpha = {recovered_mean:.3f} +/- {recovered_std:.3f} deg "
      f"(true = {TARGET_ALPHA} deg, error = {error_deg:+.3f} deg)")
print(f"Grid-search optimum = {grid_best_alpha:.3f} deg "
      f"(agreement with gradient descent: "
      f"{abs(grid_best_alpha - recovered_mean):.3f} deg)")
print(f"Gradient-descent wall-clock: {grad_time_s:.2f}s total for "
      f"{len(init_alphas)} restarts x {n_steps} steps "
      f"({grad_time_s/len(init_alphas):.3f}s/restart)")
print(f"Results written to {out_path}")
