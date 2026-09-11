#!/usr/bin/env python3
"""P5 manuscript figures. Numbers copied verbatim from project result files --
no fabrication, no re-derivation.

Fig 1: void-fraction collapse across 14 metals vs deposition angle (source:
  P5_RESULT_V2_FULL_MATERIAL_SET_20260910.md Table, same numbers as the
  manuscript's own Table tab:void).
Fig 2: tangent-offset (beta) vs homologous temperature at alpha=60 deg, the
  secondary pre-registered test (source: p5_analysis/p5_beta_tm_v2_perrun.csv).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "00_PHD_KNOWLEDGE_HUB").exists():
        ROOT = p
        break

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.linewidth": 0.8})

# ------------------------------------------------------- Fig 1: void fraction collapse
angles = [60, 75, 85]
lo = [87.6, 91.0, 95.4]
hi = [88.3, 91.6, 95.8]
mid = [(a + b) / 2 for a, b in zip(lo, hi)]
err = [(b - a) / 2 for a, b in zip(lo, hi)]

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.errorbar(angles, mid, yerr=err, fmt="o-", color="#AC581F", lw=1.6, ms=7,
            capsize=4, capthick=1.4, elinewidth=1.4, zorder=3,
            label="range across 14 metals")
for a, l, h in zip(angles, lo, hi):
    ax.annotate(f"{l:.1f}–{h:.1f}%", (a, h), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#231C13")
ax.set_xlabel("Deposition angle $\\alpha$ (deg)")
ax.set_ylabel("Void fraction (%)")
ax.set_title("Void fraction collapses onto a single\nangle-only curve across 14 metals", fontsize=10)
ax.set_xticks(angles)
ax.set_ylim(85, 98)
fig.tight_layout()
fig.savefig(OUT / "P5_void_fraction_collapse.pdf")
fig.savefig(OUT / "P5_void_fraction_collapse.png", dpi=200)
plt.close(fig)
print("Fig 1 written")

# ------------------------------------------------- Fig 2: tangent-offset vs Ts/Tm
rows = list(csv.DictReader(open(ROOT / "05_THESIS_PAPER_ASSETS/p5_analysis/p5_beta_tm_v2_perrun.csv")))
a60 = [r for r in rows if float(r["alpha"]) == 60.0 and r["status"] == "OK" and r["material"] != "Si"]
ts_tm = np.array([float(r["Ts_over_Tm"]) for r in a60])
offset = np.array([float(r["offset_tangent_deg"]) for r in a60])
names = [r["material"] for r in a60]

r_val = np.corrcoef(ts_tm, offset)[0, 1]

fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.scatter(ts_tm, offset, color="#2C7A70", s=55, zorder=3, edgecolor="#1a4a44", linewidth=0.6)
for x, y, n in zip(ts_tm, offset, names):
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7.5, color="#6C5F4E")
# best-fit line for visual reference only (not a claimed model)
m, b = np.polyfit(ts_tm, offset, 1)
xs = np.linspace(ts_tm.min() - 0.02, ts_tm.max() + 0.02, 10)
ax.plot(xs, m * xs + b, "--", color="#9B8D78", lw=1.2, zorder=2,
        label=f"linear trend, $r$={r_val:.2f}")
ax.set_xlabel("Homologous temperature $T_s/T_m$")
ax.set_ylabel("Tangent-rule offset $\\Delta\\beta$ (deg) at $\\alpha=60\\degree$")
ax.set_title(f"No significant melting-point dependence\n($n$={len(a60)}, $p_{{\\rm perm}}=0.16$)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "P5_beta_tm_scatter.pdf")
fig.savefig(OUT / "P5_beta_tm_scatter.png", dpi=200)
plt.close(fig)
print(f"Fig 2 written, r={r_val:.4f} (manuscript reports r=-0.41), n={len(a60)}")
