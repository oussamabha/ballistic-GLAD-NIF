"""
Publication-quality figure generator for P1 — LOADS PRE-COMPUTED DATA ONLY.
No training. No simulation. No descriptor extraction.
All data loaded from existing JSON / NPZ / NPY files.

Generates:
  Fig 1  — levelA_openness_trend.png
  Fig 2  — levelA_zprofile.png
  Fig 3  — morphology_pca_space_levelA.png
  Fig 4  — levelA_vs_literature_comparison.png
  Fig 5  — nif_pair_correlation.png   (with near-field inset)
  Fig 6  — nif_looa_predictions.png   (3 representative panels + NN comparison)
  Fig 7  — nif_optical_properties.png (with design window annotation)
  Fig 8  — nif_alpha72_prospective.png (reference simulation label)

Target journal: Thin Solid Films / JVST-A
  Single-column: 3.35" × 2.60"  (85 mm × 66 mm)
  Double-column: 6.69" × 4.00"  (170 mm × 102 mm)
  DPI: 300 for all raster outputs
  Font: 9 pt axis labels, 8 pt ticks and legends
"""

# ── matplotlib deepcopy patch (Python 3.14 + mpl 3.9 bug) ─────────────────
import copy, matplotlib.path as _mpath
def _fixed_path_deepcopy(self, memo):
    cls = type(self)
    r = cls.__new__(cls)
    memo[id(self)] = r
    r._vertices = copy.deepcopy(self._vertices, memo)
    r._codes = copy.deepcopy(self._codes, memo) if self._codes is not None else None
    r._interpolation_steps = self._interpolation_steps
    r._simplify_threshold = self._simplify_threshold
    r._should_simplify = self._should_simplify
    r._readonly = self._readonly
    return r
_mpath.Path.__deepcopy__ = _fixed_path_deepcopy
# ──────────────────────────────────────────────────────────────────────────────

import json, pathlib, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT = pathlib.Path("/mnt/d/GLAD_PROJECT")
LEVEL_A_JSON   = PROJECT / "03_VOXEL_DESCRIPTOR_ANALYSIS/analysis/levelA_results/levelA_accepted_results.json"
DESCR_JSON     = PROJECT / "03_VOXEL_DESCRIPTOR_ANALYSIS/analysis/levelA_results/levelA_descriptors.json"
VOXEL_DIR      = PROJECT / "02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline/voxels"
NPZ_FILE       = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/corr_functions.npz"
LOOA_JSON      = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/looa_results.json"
MG_JSON        = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/mg_properties.json"
PROSP_JSON     = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/alpha72_prospective_results.json"
OUT            = PROJECT / "05_THESIS_PAPER_ASSETS/figures/publication"
OUT.mkdir(parents=True, exist_ok=True)

# ── global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.75",
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.7,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "ytick.minor.size":   2.0,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "grid.linestyle":     "--",
    "grid.alpha":         0.35,
    "grid.linewidth":     0.5,
})

# ── single-column and double-column sizes ────────────────────────────────────
SC = (3.35, 2.60)   # single column
DC = (6.69, 3.80)   # double column
DC_TALL = (6.69, 4.50)

ALPHAS = [65, 70, 75, 80, 85, 87, 89]
VIRIDIS = [cm.viridis(v) for v in np.linspace(0.1, 0.9, len(ALPHAS))]
A2C = {a: c for a, c in zip(ALPHAS, VIRIDIS)}

def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  -> {p.name}  ({p.stat().st_size//1024} kB)")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Void fraction vs deposition angle
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Openness trend ...")

accepted = json.loads(LEVEL_A_JSON.read_text())
by_alpha = defaultdict(list)
for r in accepted:
    by_alpha[float(r["alpha"])].append(r)

a_vals, p_means, p_stds, n_list, cv_list = [], [], [], [], []
for alpha in sorted(by_alpha):
    rows = by_alpha[alpha]
    pors  = [float(r["porosity"]) for r in rows]
    atoms = [float(r["atoms_per_nm2"]) for r in rows]
    n     = len(pors)
    a_vals.append(alpha); p_means.append(np.mean(pors))
    p_stds.append(np.std(pors, ddof=1) if n > 1 else 0.0)
    n_list.append(n)
    cv_list.append(np.std(atoms, ddof=1)/np.mean(atoms)*100 if n > 1 else 0.0)

a_arr  = np.array(a_vals)
p_arr  = np.array(p_means)
ps_arr = np.array(p_stds)
multi  = np.array(n_list) > 1
single = ~multi

fig, ax = plt.subplots(figsize=SC)
ax.plot(a_arr, p_arr, color="steelblue", lw=1.5, zorder=2)
ax.plot(a_arr[single], p_arr[single], "o", color="steelblue",
        ms=5, mfc="white", mew=1.3, zorder=5, label="Single seed")
for i, (alpha, p, s) in enumerate(zip(a_arr[multi], p_arr[multi], ps_arr[multi])):
    kw = {"label": "Three-seed mean"} if i == 0 else {}
    ax.plot(alpha, p, "s", color="steelblue", ms=6, zorder=6, **kw)
    ax.errorbar(alpha, p, yerr=max(s, 0.01),
                fmt="none", ecolor="steelblue", elinewidth=1.3, capsize=3, zorder=6)
for i, alpha in enumerate(a_arr):
    if n_list[i] > 1:
        ax.annotate(f"CV$_{{\\rho_A}}$={cv_list[i]:.2f}%",
                    xy=(alpha, p_arr[i]),
                    xytext=(alpha - 3.5, p_arr[i] - 5),
                    fontsize=6.5, color="dimgray",
                    arrowprops=dict(arrowstyle="-", color="lightgray", lw=0.5))
ax.set_xlabel(r"Deposition angle $\alpha$ (°)")
ax.set_ylabel(r"Void fraction $P$ (%)")
ax.set_xlim(57, 93); ax.set_ylim(72, 101)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True)
ax.legend(loc="lower right", frameon=True)
fig.tight_layout()
savefig(fig, "levelA_openness_trend")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Depth-resolved void fraction (Z-profiles)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Z-profiles ...")

descr = json.loads(DESCR_JSON.read_text())
ANOMALOUS = {"LA_helical_alpha_300nm_alpha060_seed000_300nm_helical_pitch150"}
seen_a = set(); selected = []
for r in sorted(descr["results"], key=lambda x: (x["alpha_deg"], x["seed"])):
    a = r["alpha_deg"]
    if r["job_id"] in ANOMALOUS: continue
    if a not in seen_a: seen_a.add(a); selected.append(r)

NZ = 151; VOX_Z = 300.0 / (NZ - 1)
Z_NM = np.arange(NZ) * VOX_Z
alphas_plot = sorted(set(r["alpha_deg"] for r in selected))
n = len(alphas_plot)
cmap_plasma = matplotlib.colormaps.get_cmap("plasma").resampled(n)
a2col = {a: cmap_plasma(i) for i, a in enumerate(alphas_plot)}

fig, axes = plt.subplots(1, 2, figsize=DC, sharey=False)
ax_abs, ax_norm = axes

for r in selected:
    a = r["alpha_deg"]
    vpath = VOXEL_DIR / r["voxel_file"]
    if not vpath.exists(): continue
    rho = np.load(vpath).astype(np.float32)
    void_z = 1.0 - rho.mean(axis=(0, 1))
    occ = np.where(void_z < 0.97)[0]
    h_top = float(Z_NM[occ[-1]]) if occ.size > 0 else 300.0
    col = a2col[a]
    ax_abs.plot(Z_NM, void_z * 100, color=col, lw=1.3)
    ax_norm.plot(Z_NM / h_top, void_z * 100, color=col, lw=1.3)

for ax in axes:
    ax.set_ylabel(r"Layer void fraction $1 - \bar{\rho}_z$ (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True)

ax_abs.set_xlabel(r"Depth from substrate $z$ (nm)")
ax_abs.set_xlim(0, 305)
ax_abs.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax_abs.set_title("(a) Absolute depth", fontsize=9)

ax_norm.set_xlabel(r"Normalised height $z / h$")
ax_norm.set_xlim(0, 1.05)
ax_norm.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_norm.set_title("(b) Normalised height", fontsize=9)

# Proper colorbar outside axes
sm = plt.cm.ScalarMappable(cmap=cmap_plasma,
                            norm=plt.Normalize(vmin=min(alphas_plot), vmax=max(alphas_plot)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.03, fraction=0.035)
cbar.set_label(r"$\alpha$ (°)", fontsize=8)
cbar.set_ticks(alphas_plot)
cbar.set_ticklabels([f"{int(a)}°" for a in alphas_plot])

fig.tight_layout()
savefig(fig, "levelA_zprofile")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — PCA morphology space
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 3] PCA ...")

results = [r for r in descr["results"] if r["job_id"] not in ANOMALOUS]
by_a3 = defaultdict(list)
for r in results: by_a3[r["alpha_deg"]].append(r)

pca_rows = []
for a in sorted(by_a3):
    for r in by_a3[a]:
        fan = r.get("fanning_angle_deg")
        pca_rows.append({
            "alpha":   a,
            "seed":    r["seed"],
            "open":    r["openness_proxy"],
            "fill":    r["filling_fraction"],
            "aniso":   r["anisotropy_ratio"],
            "fan":     fan if fan is not None else 0.0,
        })

X = np.array([[r["open"], r["fill"], r["aniso"], r["fan"]] for r in pca_rows])
X_sc = StandardScaler().fit_transform(X)
pca_model = PCA(n_components=2)
X_pc = pca_model.fit_transform(X_sc)
var = pca_model.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=DC)
ax_pca, ax_load = axes

cmap_vir = matplotlib.colormaps.get_cmap("viridis")
a_uniq = sorted(set(r["alpha"] for r in pca_rows))
norm_a  = plt.Normalize(vmin=min(a_uniq), vmax=max(a_uniq))

for i, row in enumerate(pca_rows):
    col = cmap_vir(norm_a(row["alpha"]))
    ax_pca.scatter(X_pc[i, 0], X_pc[i, 1], color=col, s=40, zorder=5)

# Trajectory arrow (single-seed points in α order)
ss_idx = [i for i, r in enumerate(pca_rows) if r["seed"] == 0]
ss_sorted = sorted(ss_idx, key=lambda i: pca_rows[i]["alpha"])
xs = X_pc[ss_sorted, 0]; ys = X_pc[ss_sorted, 1]
ax_pca.plot(xs, ys, "k--", lw=0.8, zorder=3, alpha=0.5)
ax_pca.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

sm3 = plt.cm.ScalarMappable(cmap=cmap_vir, norm=norm_a)
sm3.set_array([]); fig.colorbar(sm3, ax=ax_pca, label=r"$\alpha$ (°)", pad=0.02)

ax_pca.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
ax_pca.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
ax_pca.set_title("(a) Morphology space", fontsize=9)
ax_pca.grid(True)

# PC1 loadings — rename fanning_angle to "width-gradient proxy"
feature_labels = [
    "Openness proxy", "Filling fraction",
    "XY anisotropy", "Width-gradient\nproxy"
]
load1 = pca_model.components_[0]
colours = ["steelblue" if v >= 0 else "tomato" for v in load1]
ax_load.barh(feature_labels, load1, color=colours, edgecolor="white", lw=0.5)
ax_load.axvline(0, color="black", lw=0.7)
ax_load.set_xlabel("PC1 loading")
ax_load.set_title("(b) PC1 feature loadings", fontsize=9)
ax_load.grid(True, axis="x")

fig.tight_layout()
savefig(fig, "morphology_pca_space_levelA")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Literature comparison
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Literature comparison ...")

sim_alphas = sorted(by_alpha.keys())
sim_p      = [np.mean([float(r["porosity"]) for r in by_alpha[a]]) for a in sim_alphas]

# Fischer & Schubert 2013 optical void fractions at α=85°
lit_points = [
    ("Co",       85, 75.38, 0.01),
    ("Ti",       85, 78.30, 0.04),
    ("NiFeMo",   85, 75.0,  0.1),
    ("Co/Al₂O₃", 85, 62.03, 0.07),
]
lit_vals = [p for _, _, p, _ in lit_points]
lit_band  = (min(lit_vals), max(lit_vals))

fig, ax = plt.subplots(figsize=SC)
ax.plot(sim_alphas, sim_p, "o-", color="steelblue", lw=1.5, ms=5,
        mfc="white", mew=1.3, label="Ballistic simulation (Cu)", zorder=5)
ax.axhspan(lit_band[0], lit_band[1], xmin=0, xmax=1,
           color="tomato", alpha=0.15, zorder=2, label="Literature range (85°)")

for mat, al, p, e in lit_points:
    ax.errorbar(al, p, yerr=e, fmt="^", color="tomato",
                ms=6, mfc="tomato", mew=1.0, capsize=3, zorder=6)
    ax.text(al + 0.7, p, mat, fontsize=6.5, color="tomato", va="center")

# Annotate offset
sim_p85 = np.interp(85, sim_alphas, sim_p)
lit_med = np.median(lit_vals)
ax.annotate("", xy=(85, lit_med), xytext=(85, sim_p85),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0))
ax.text(86, (lit_med + sim_p85) / 2,
        f"+{sim_p85 - lit_med:.1f} pp\n(indicative offset)",
        fontsize=6.5, color="dimgray", va="center")

ax.set_xlabel(r"Deposition angle $\alpha$ (°)")
ax.set_ylabel(r"Void fraction (%)")
ax.set_xlim(57, 93); ax.set_ylim(55, 101)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True)
ax.legend(loc="upper left", fontsize=7, frameon=True)
fig.tight_layout()
savefig(fig, "levelA_vs_literature_comparison")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Pair correlation h_g(r; α) with near-field inset
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Pair correlation ...")

npz = np.load(NPZ_FILE)
h_all = {a: npz["h_r"][i] for i, a in enumerate(npz["alphas"])}
r_all = {a: npz["r_bins"][i] for i, a in enumerate(npz["alphas"])}

fig, ax = plt.subplots(figsize=SC)
for ai, alpha in enumerate(ALPHAS):
    r_v = r_all[alpha]; h_v = h_all[alpha]
    ax.plot(r_v, h_v, color=A2C[alpha], lw=1.5,
            label=f"{alpha}°" if alpha in [65, 75, 85, 89] else None)

ax.set_xlabel(r"$r$ (voxels)")
ax.set_ylabel(r"$h_g(r;\alpha)$")
ax.set_xlim(0, 25); ax.set_ylim(-0.05, 1.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True)

# Near-field inset (r = 0 to 5 vox)
axins = ax.inset_axes([0.35, 0.38, 0.60, 0.58])
for ai, alpha in enumerate(ALPHAS):
    r_v = r_all[alpha]; h_v = h_all[alpha]
    mask = r_v <= 5.5
    axins.plot(r_v[mask], h_v[mask], color=A2C[alpha], lw=1.5)
axins.set_xlim(0, 5.5)
axins.set_ylim(-0.05, 1.05)
axins.set_xlabel(r"$r$ (vox)", fontsize=7)
axins.set_ylabel(r"$h_g$", fontsize=7)
axins.set_title("Near-field (r ≤ 5 vox)", fontsize=7)
axins.tick_params(labelsize=6)
axins.grid(True, alpha=0.3)
ax.indicate_inset_zoom(axins, edgecolor="gray", lw=0.7)

# Colorbar-style legend
sm5 = plt.cm.ScalarMappable(cmap=cm.viridis,
                             norm=plt.Normalize(vmin=65, vmax=89))
sm5.set_array([])
cbar5 = fig.colorbar(sm5, ax=ax, pad=0.02, fraction=0.04)
cbar5.set_label(r"$\alpha$ (°)", fontsize=8)
cbar5.set_ticks(ALPHAS)
cbar5.set_ticklabels([f"{a}°" for a in ALPHAS])

fig.tight_layout()
savefig(fig, "nif_pair_correlation")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — LOOA: 3 representative panels, ground truth + NN baseline
#          (CorrNIF prediction curves not stored; R² annotated from JSON)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 6] LOOA (3 representative panels) ...")

looa = json.load(open(LOOA_JSON))
corrnif_r2 = {int(k): v["r2"] for k, v in looa.items()}

# Compute NN baselines
nn_r2 = {}
for held in ALPHAS:
    train = [a for a in ALPHAS if a != held]
    h_gt  = h_all[held]
    nn_a  = min(train, key=lambda a: abs(a - held))
    h_nn  = h_all[nn_a]
    ss_tot = np.sum((h_gt - h_gt.mean())**2)
    nn_r2[held] = 1.0 - np.sum((h_gt - h_nn)**2) / ss_tot

# Representative panels: dense (65°), intermediate (80°), sparse/worst (89°)
rep_alphas = [65, 80, 89]
labels_rep = ["Dense (α=65°)", "Intermediate (α=80°)", "Sparse (α=89°)"]

fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.60), sharey=True)
for i, (held, label) in enumerate(zip(rep_alphas, labels_rep)):
    ax = axes[i]
    r_v   = r_all[held]; h_gt  = h_all[held]
    train = [a for a in ALPHAS if a != held]
    nn_a  = min(train, key=lambda a: abs(a - held))
    h_nn  = h_all[nn_a]

    ax.fill_between(r_v, h_gt, h_nn, alpha=0.18, color="tomato")
    ax.plot(r_v, h_gt,  color=A2C[held], lw=1.8,
            label=f"Reference ($h_g$, {held}°)")
    ax.plot(r_v, h_nn,  color="gray", lw=1.2, ls="--",
            label=f"NN baseline ({nn_a}°)")

    r2_nif = corrnif_r2[held]; r2_nn = nn_r2[held]
    ax.text(0.97, 0.97,
            f"CorrNIF $R^2$={r2_nif:.4f}\nNN $R^2$={r2_nn:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            bbox=dict(fc="white", ec="0.75", lw=0.5, pad=2))
    ax.set_title(label, fontsize=8)
    ax.set_xlabel(r"$r$ (voxels)", fontsize=8)
    ax.set_xlim(0, 25); ax.set_ylim(-0.1, 1.1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True)
    if i == 0:
        ax.set_ylabel(r"$h_g(r;\alpha_\mathrm{held})$")
    ax.legend(fontsize=6, loc="upper right", frameon=True)

fig.tight_layout()
savefig(fig, "nif_looa_predictions")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Maxwell-Garnett optical properties with design window
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 7] Optical properties ...")

mg = json.load(open(MG_JSON))
alpha_arr = np.array([int(k) for k in mg], dtype=float)
n_o = np.array([mg[k]["n_o_MG"]     for k in mg])
n_e = np.array([mg[k]["n_e_Wiener"] for k in mg])
k_o = np.array([mg[k]["k_o"]        for k in mg])
dn  = np.array([mg[k]["delta_n"]    for k in mg])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DC)

# Design window shading (65°–80°, Δn > 0.55)
for ax in [ax1, ax2]:
    ax.axvspan(65, 80, color="gold", alpha=0.15, zorder=1, label="Design window")

ax1.plot(alpha_arr, n_o, "o-", color="royalblue", ms=5, lw=1.5, label=r"$n_o$ (ord.)")
ax1.plot(alpha_arr, n_e, "s--", color="firebrick", ms=5, lw=1.5, label=r"$n_e$ (ext.)")
ax1.set_xlabel(r"Deposition angle $\alpha$ (°)")
ax1.set_ylabel("Real refractive index")
ax1.set_title("(a) Effective refractive indices", fontsize=9)
ax1.legend(fontsize=8, frameon=True)
ax1.grid(True); ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax2.plot(alpha_arr, dn, "^-", color="purple", ms=5, lw=1.5, label=r"$\Delta n = |n_e - n_o|$")
# Mark optimum
idx_opt = np.argmax(dn)
ax2.plot(alpha_arr[idx_opt], dn[idx_opt], "*", color="purple", ms=10, zorder=6)
ax2.annotate(f"α={int(alpha_arr[idx_opt])}°\nΔn={dn[idx_opt]:.3f}",
             xy=(alpha_arr[idx_opt], dn[idx_opt]),
             xytext=(alpha_arr[idx_opt] + 4, dn[idx_opt] - 0.12),
             fontsize=7, color="purple",
             arrowprops=dict(arrowstyle="->", color="purple", lw=0.8))
ax2.axhline(0.55, color="gray", ls=":", lw=1.0, label="Δn = 0.55 threshold")
ax2.set_xlabel(r"Deposition angle $\alpha$ (°)")
ax2.set_ylabel(r"Form birefringence $\Delta n$")
ax2.set_title("(b) Birefringence and design window", fontsize=9)
ax2.legend(fontsize=8, frameon=True)
ax2.grid(True); ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))

fig.tight_layout()
savefig(fig, "nif_optical_properties")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Prospective validation at α=72°
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 8] Prospective validation ...")

prosp = json.load(open(PROSP_JSON))
r_v  = np.array(prosp["r_bins"])
h_gt = np.array(prosp["h_gt72"])
h_nif = np.array(prosp["h_corrnif"])
h_lin = np.array(prosp["h_linear_blend"])
h_nn  = np.array(prosp["h_nn_alpha70"])
r2_nif = prosp["r2_corrnif"]
r2_lin = prosp["r2_linear"]
r2_nn  = prosp["r2_nn70"]

# Training neighbours
npz_r70 = r_all[70]; npz_h70 = h_all[70]
npz_r75 = r_all[75]; npz_h75 = h_all[75]

fig, ax = plt.subplots(figsize=SC)

# Training context curves (thin, low saturation)
for alpha in ALPHAS:
    ax.plot(r_all[alpha], h_all[alpha],
            color=A2C[alpha], lw=0.7, alpha=0.35)

# Two nearest training neighbours highlighted
ax.plot(npz_r70, npz_h70, color=A2C[70], lw=1.5, alpha=0.8, label="70° (train)")
ax.plot(npz_r75, npz_h75, color=A2C[75], lw=1.5, alpha=0.8, label="75° (train)")

# Predictions
ax.plot(r_v, h_gt,  "k-",  lw=2.0, zorder=7,
        label=f"Reference simulation 72° ($R^2$=—)")
ax.plot(r_v, h_nif, "r--", lw=1.8, zorder=8,
        label=f"CorrNIF pred. ($R^2$={r2_nif:.4f})")
ax.plot(r_v, h_lin, "b:",  lw=1.5, zorder=6,
        label=f"Linear blend ($R^2$={r2_lin:.4f})")
ax.plot(r_v, h_nn,  "m-.", lw=1.3, zorder=5,
        label=f"Nearest-neighbour ($R^2$={r2_nn:.4f})")

ax.set_xlabel(r"$r$ (voxels)")
ax.set_ylabel(r"$h_g(r;\,72°)$")
ax.set_xlim(0, 25); ax.set_ylim(-0.05, 1.1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True)
ax.legend(fontsize=7, loc="upper right", frameon=True)
fig.tight_layout()
savefig(fig, "nif_alpha72_prospective")

print("\nAll 8 publication figures generated.")
