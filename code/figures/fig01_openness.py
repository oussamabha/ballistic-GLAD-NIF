"""
Fig 1 — Bead-sphere void fraction (openness proxy) vs deposition angle.
Standalone script. Reads the authoritative P1 27-run campaign manifest only.
No simulation, no training. Restyled to match the journal-clean convention
used by fig02-fig07 (no in-figure title/subtitle, no footer annotation box
-- that information belongs in the LaTeX caption).
"""
import copy
import json
import pathlib
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.path as _mpath


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

PROJECT = pathlib.Path("/mnt/d/GLAD_PROJECT")
CLOSURE = PROJECT / "00_PHD_KNOWLEDGE_HUB/99_PENDING_REVIEW/P1_PRIMARY_CAMPAIGN_CLOSURE_AND_TARGETED_VALIDATION_20260620"
OUT = PROJECT / "05_THESIS_PAPER_ASSETS/figures/publication"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.3,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "grid.linestyle":   "--",
    "grid.alpha":       0.30,
    "grid.linewidth":   0.5,
})

SC = (3.65, 3.05)


def savefig(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / "levelA_openness_trend.{}".format(ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
    plt.close(fig)


man = json.loads((CLOSURE / "P1_PRIMARY_27_AUTHORITATIVE_MANIFEST.json").read_text())
by_alpha = defaultdict(list)
for r in man["runs"]:
    by_alpha[r["alpha_deg"]].append(r)

alphas = sorted(by_alpha)
means = [statistics.mean(r["P_bead_pct"] for r in by_alpha[a]) for a in alphas]
stds = [statistics.stdev(r["P_bead_pct"] for r in by_alpha[a]) if len(by_alpha[a]) > 1 else 0.0
        for a in alphas]

fig, ax = plt.subplots(figsize=SC, constrained_layout=True)
ax.errorbar(alphas, means, yerr=stds, fmt="o-", color="steelblue", lw=1.3, ms=5.5,
            mfc="white", mew=1.2, capsize=3, elinewidth=1.0, zorder=4,
            label="P1 campaign (27 runs, 3 seeds/angle)")

ax.set_xlabel(r"Deposition angle $\alpha$ (°)")
ax.set_ylabel(r"Bead-sphere void fraction $P_{\mathrm{bead}}$ (%)")
ax.set_xticks(alphas)
ax.set_ylim(86, 100)
ax.grid(True)
ax.legend(loc="upper left", frameon=True)

savefig(fig, "levelA_openness_trend")
