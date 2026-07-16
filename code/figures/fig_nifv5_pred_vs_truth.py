"""
Fig -- NIF v5 predicted vs ground-truth density field, one trained angle.
Standalone script. Loads the saved checkpoint (inference/forward pass only,
no training) and the archived ground-truth voxel array. Direct visual proof
of the r=0.984 / per-angle table (tab:nifv5_peralpha) headline result.
"""
import math
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIN = "/mnt/d/GLAD_PROJECT/deepseek_correction/pinn_production_env"
CKPT = os.path.join(PIN, "nif_v5_film_smooth_best.pt")
OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

BOX_W, BOX_D = 50.0, 50.0
Z_MAX = 700.0
T_MAX = 3500.0
NX, NY, NZ = 25, 25, 351
ALPHA_CENTER, ALPHA_SCALE = 81.0, 6.0
N_LAYERS = 4
ALPHA_SHOW = 80.0


class FiLMCondNIFv5(nn.Module):
    def __init__(self, B, hidden, out_scale):
        super().__init__()
        self.register_buffer("B", B)
        self.hidden = int(hidden)
        self.out_scale = float(out_scale)
        in_dim = 2 * B.shape[1]
        dims = [in_dim] + [hidden] * N_LAYERS
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(N_LAYERS)])
        self.head = nn.Linear(hidden, 1)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, 2 * N_LAYERS * hidden),
        )

    def forward(self, x_norm_xyzt, alpha_norm):
        proj = 2 * math.pi * x_norm_xyzt @ self.B
        h = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        film = self.alpha_mlp(alpha_norm.reshape(-1, 1))
        gamma_raw, beta_raw = film.chunk(2, dim=-1)
        gamma_raw = gamma_raw.view(-1, N_LAYERS, self.hidden)
        beta_raw = beta_raw.view(-1, N_LAYERS, self.hidden)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            gamma = 1.0 + 0.10 * gamma_raw[:, i, :]
            beta = 0.10 * beta_raw[:, i, :]
            h = torch.nn.functional.silu(gamma * h + beta)
        return self.out_scale * torch.sigmoid(self.head(h)).squeeze(-1)


def normalise_xyzt(xyzt):
    out = torch.empty_like(xyzt)
    out[:, 0] = xyzt[:, 0] / BOX_W - 0.5
    out[:, 1] = xyzt[:, 1] / BOX_D - 0.5
    out[:, 2] = xyzt[:, 2] / Z_MAX - 0.5
    out[:, 3] = xyzt[:, 3] / T_MAX - 0.5
    return out


def normalise_alpha(alpha_deg):
    return (alpha_deg - ALPHA_CENTER) / ALPHA_SCALE


device = torch.device("cpu")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = FiLMCondNIFv5(ckpt["B"], ckpt["hidden"], ckpt["out_scale"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

VOX_NM = 2.0
t_max_s = ckpt.get("t_max_s", T_MAX)
xs = (np.arange(NX, dtype=np.float32) + 0.5) * VOX_NM
ys = (np.arange(NY, dtype=np.float32) + 0.5) * VOX_NM
zs = (np.arange(NZ, dtype=np.float32) + 0.5) * VOX_NM
xx, yy, zzg = np.meshgrid(xs, ys, zs, indexing="ij")
coords_np = np.stack([xx.ravel(), yy.ravel(), zzg.ravel()], axis=-1).astype(np.float32)
t_np = np.full((coords_np.shape[0], 1), t_max_s, dtype=np.float32)
coords = np.concatenate([coords_np, t_np], axis=1)
coords_t = torch.from_numpy(coords)

pred = np.empty(coords.shape[0], dtype=np.float32)
bs = 65536
with torch.no_grad():
    for start in range(0, coords.shape[0], bs):
        end = min(start + bs, coords.shape[0])
        a_norm = normalise_alpha(torch.full((end - start,), ALPHA_SHOW))
        pred[start:end] = model(normalise_xyzt(coords_t[start:end]), a_norm).numpy()
pred = pred.reshape(NX, NY, NZ)

gt_path = os.path.join(PIN, "rho_voxel_alpha80_50x50.npy")
gt = np.load(gt_path).astype(np.float32)

r = np.corrcoef(pred.ravel(), gt.ravel())[0, 1]
print(f"alpha={ALPHA_SHOW}: Pearson r (this render) = {r:.4f}")

y_mid = NY // 2
pred_xz = pred[:, y_mid, :].T
gt_xz = gt[:, y_mid, :].T

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 8.5, "axes.labelsize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)
vmin, vmax = 0.0, max(gt.max(), pred.max())

im = axes[0].imshow(gt_xz, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                     extent=[0, BOX_W, 0, Z_MAX], aspect="auto")
axes[0].set_title(r"(a) Ground truth, $\alpha=80°$")
axes[0].set_xlabel("x (nm)")
axes[0].set_ylabel("z (nm)")

axes[1].imshow(pred_xz, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
               extent=[0, BOX_W, 0, Z_MAX], aspect="auto")
axes[1].set_title(r"(b) NIF v5 predicted")
axes[1].set_xlabel("x (nm)")

diff = pred_xz - gt_xz
dmax = np.abs(diff).max()
im2 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                      extent=[0, BOX_W, 0, Z_MAX], aspect="auto")
axes[2].set_title(rf"(c) Difference, $r={r:.3f}$")
axes[2].set_xlabel("x (nm)")

cbar = fig.colorbar(im, ax=axes[:2], fraction=0.05, pad=0.02)
cbar.set_label("Density")
cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.09, pad=0.03)
cbar2.set_label("Pred - truth")

for ext in ("pdf", "png"):
    p = OUT / f"nifv5_pred_vs_truth.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
