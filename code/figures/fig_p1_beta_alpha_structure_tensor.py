#!/usr/bin/env python3
"""
Fig. (P1) -- Column-tilt angle beta(alpha): parameter-free ballistic model vs Cu / CuOx
SEM experiment and the two textbook geometric rules.

Standalone. All values are hard-coded from committed analysis artefacts (no GPU, no
checkpoint read):

  * beta_sim (structure-tensor estimator, beta_structure_tensor.py):
      median + 16-84 occupancy-weighted band MEASURED at all 9 angles
      (beta_two_estimator_20260908.json; beta_one on the GRIDFIX seed-0 checkpoints).
      No interpolation. Cross-checked vs beta_highalpha_ccl.py at alpha=89 (CCL 64 deg
      vs structure tensor 68 deg -- consistent within the band); the watershed
      estimator (beta_robust_column_axis.py) is superseded, biased low on dense films.
  * beta_sim with a conservative surface-diffusion budget (fixed kernel aa32952):
      P1_DIFFUSION_DECOMPOSITION_C2_20260907.md  (cap = 1: a75->48.9, a82->55.3,
      a89->64.9; cap = 2: a82->56.5).
  * Experimental anchors (SEM cross-section), from P1_BETA_VALIDATION_20260907.md:
      Ben Nacer et al. 2019 (Cu2O):  a60->44+/-0.5, a75->53+/-0.5, a85->60+/-0.5
      Benamor et al. 2026 (CuO):     a60->44+/-5,   a75->52+/-5,   a85->63
      Chargui et al. (Cu):           a80->56
      Gonzalez-Cobos et al. 2015 (Cu): a80->~60
  * cosine rule  beta = alpha - asin[(1 - cos alpha) / 2]
  * tangent rule beta = atan(tan alpha / 2)

Output: fig_p1_beta_alpha_structure_tensor.pdf (+ .png) in this directory.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = pathlib.Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 6.6, "legend.framealpha": 0.92, "legend.edgecolor": "0.75",
    "lines.linewidth": 1.6, "axes.linewidth": 0.8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "grid.linestyle": "--", "grid.alpha": 0.30, "grid.linewidth": 0.5,
})

# ---- geometric rules -------------------------------------------------------
a_dense = np.linspace(55, 90, 400)
ar = np.radians(a_dense)
beta_cos = np.degrees(ar - np.arcsin((1.0 - np.cos(ar)) / 2.0))
beta_tan = np.degrees(np.arctan(np.tan(ar) / 2.0))

# ---- ballistic structure-tensor beta_sim(alpha) --------------------------
# median + 16-84 band MEASURED at every angle (beta_two_estimator_20260908.json,
# beta_structure_tensor.beta_one on the GRIDFIX seed-0 checkpoints). No interpolation.
a_sim   = np.array([60,   65,   70,   72,   75,   80,   85,   87,   89],   float)
b_sim   = np.array([41.3, 45.4, 49.6, 51.4, 53.5, 58.5, 63.6, 65.9, 68.4], float)
b_lo    = np.array([33.2, 37.7, 41.8, 43.8, 45.6, 50.9, 56.5, 59.0, 61.7], float)
b_hi    = np.array([51.8, 54.9, 58.9, 60.2, 62.0, 66.3, 71.0, 73.1, 75.1], float)

# ---- ballistic + conservative surface diffusion --------------------------
a_dif1  = np.array([75, 82, 89], float)
b_dif1  = np.array([48.9, 55.3, 64.9], float)
a_dif2  = np.array([82], float)
b_dif2  = np.array([56.5], float)

# ---- experimental SEM anchors -------------------------------------------
bn_a, bn_b, bn_e = np.array([60, 75, 85.]), np.array([44, 53, 60.]), np.array([0.5, 0.5, 0.5])
bm_a, bm_b, bm_e = np.array([60, 75, 85.]), np.array([44, 52, 63.]), np.array([5, 5, 3.])
cu_a, cu_b       = np.array([80, 80.]), np.array([56, 60.])   # Chargui, Gonzalez-Cobos (Cu)

fig, ax = plt.subplots(figsize=(5.4, 3.25))

ax.plot(a_dense, beta_cos, color="0.35", ls="-",  lw=1.2, label=r"cosine rule")
ax.plot(a_dense, beta_tan, color="0.35", ls=":",  lw=1.2, label=r"tangent rule")

ax.fill_between(a_sim, b_lo, b_hi, color="#c62828", alpha=0.16, lw=0,
                label=r"$\beta_{\mathrm{sim}}$ 16-84 band")
ax.plot(a_sim, b_sim, "-o", color="#c62828", ms=4.2, mfc="#c62828", mec="#c62828",
        label=r"$\beta_{\mathrm{sim}}$ ballistic (structure tensor)")

ax.plot(a_dif1, b_dif1, "^", color="#ef6c00", ms=5.5, mfc="none", mew=1.3,
        label=r"$\beta_{\mathrm{sim}}$ + diffusion (hop cap 1)")
ax.plot(a_dif2, b_dif2, "v", color="#ef6c00", ms=5.5, mfc="none", mew=1.3,
        label=r"$\beta_{\mathrm{sim}}$ + diffusion (hop cap 2)")

ax.errorbar(bn_a, bn_b, yerr=bn_e, fmt="s", color="#1565c0", ms=4.5, capsize=2.5,
            lw=1.0, label=r"Ben Nacer 2019 (Cu$_2$O, SEM)")
ax.errorbar(bm_a, bm_b, yerr=bm_e, fmt="D", color="#00838f", ms=4.0, capsize=2.5,
            lw=1.0, label=r"Benamor 2026 (CuO, SEM)")
ax.plot(cu_a, cu_b, "*", color="#2e7d32", ms=9, mec="#1b5e20", mew=0.5,
        label=r"Chargui / Gonzalez-Cobos (Cu, SEM)")

ax.set_xlabel(r"deposition angle $\alpha$ (deg)")
ax.set_ylabel(r"column-tilt angle $\beta$ (deg)")
ax.set_xlim(57, 90.5)
ax.set_ylim(20, 90)
ax.grid(True)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
          handlelength=1.7, borderaxespad=0.0, labelspacing=0.5, frameon=False)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_p1_beta_alpha_structure_tensor.{ext}")
print("wrote", OUT / "fig_p1_beta_alpha_structure_tensor.pdf")
