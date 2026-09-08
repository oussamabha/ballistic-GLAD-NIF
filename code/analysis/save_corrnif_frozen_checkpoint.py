"""save_corrnif_frozen_checkpoint.py -- train the deterministic CorrNIF once (seed 0,
the exact recipe used by run_corrnif_gradient_inverse_demo.py / lens1_wp1) and save its
weights, so WP1 / WP2 / the inversion demo LOAD one frozen model instead of each
retraining from scratch (~60 s x N).

Output: runs/corrnif_frozen/corrnif_alpha_only_seed0.pt  (+ meta json)
Reproducibility: torch.manual_seed(0), np.seed(0), 5000 epochs Adam lr 3e-3 cosine,
weight_decay 1e-5, on corr_functions.npz alphas [65,70,75,80,85,87,89].
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def _find_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists():
            return p
    raise RuntimeError("root not found")


ROOT = _find_root()
PIPE = ROOT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline"
OUT = ROOT / "02_PINN_NIF/pinn_training/runs/corrnif_frozen"
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [65, 70, 75, 80, 85, 87, 89]
ALPHA_NORM = lambda a: (a - 74.5) / 14.5

torch.manual_seed(0)
np.random.seed(0)

npz = np.load(PIPE / "corr_functions.npz")
assert [int(a) for a in npz["alphas"]] == ALPHAS
all_r = {a: npz["r_bins"][i].astype(np.float32) for i, a in enumerate(ALPHAS)}
all_h = {a: npz["h_r"][i].astype(np.float32) for i, a in enumerate(ALPHAS)}
R_MAX_NORM = float(all_r[ALPHAS[0]][-1])


class CorrNIF(nn.Module):
    def __init__(self, n_freq=12, hidden=64, n_layers=3, sigma_r=2.0):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.register_buffer("B_r", torch.randn(1, n_freq, generator=g) * sigma_r)
        self.n_layers, self.hidden = n_layers, hidden
        self.layers = nn.ModuleList([nn.Linear(2 * n_freq if i == 0 else hidden, hidden)
                                     for i in range(n_layers)])
        self.head = nn.Linear(hidden, 1)
        self.film_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(),
                                      nn.Linear(32, 2 * n_layers * hidden))

    def forward(self, x):
        proj = 2.0 * math.pi * x[:, 1:2] @ self.B_r
        h = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        gamma, beta = self.film_mlp(x[:, 0:1]).chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.hidden)
        beta = beta.view(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.nn.functional.silu((1.0 + 0.1 * gamma[:, i]) * h + 0.1 * beta[:, i])
        return self.head(h).squeeze(-1)


xs, ys = [], []
for a in ALPHAS:
    for ri, rv in enumerate(all_r[a]):
        xs.append([ALPHA_NORM(a), float(rv) / R_MAX_NORM]); ys.append(float(all_h[a][ri]))
X = torch.tensor(np.array(xs, np.float32)); Y = torch.tensor(np.array(ys, np.float32))

model = CorrNIF()
opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
t0 = time.time()
for ep in range(5000):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(X), Y)
    loss.backward(); opt.step(); sch.step()
model.eval()
final_mse = float(torch.nn.functional.mse_loss(model(X), Y).item())

ckpt = OUT / "corrnif_alpha_only_seed0.pt"
torch.save({"state_dict": model.state_dict(),
            "arch": dict(n_freq=12, hidden=64, n_layers=3, sigma_r=2.0),
            "alphas": ALPHAS, "alpha_norm": "(a-74.5)/14.5", "r_max_norm": R_MAX_NORM,
            "train_mse": final_mse, "epochs": 5000, "seed": 0}, ckpt)
(OUT / "meta.json").write_text(json.dumps(
    {"checkpoint": ckpt.name, "train_mse": final_mse, "train_time_s": round(time.time() - t0, 1),
     "source_npz": str(PIPE / "corr_functions.npz"), "note":
     "Frozen alpha-only CorrNIF. Load in WP1/WP2/inversion instead of retraining."},
    indent=2))
print(f"saved {ckpt}  (train MSE {final_mse:.3e}, {time.time()-t0:.1f}s)")
