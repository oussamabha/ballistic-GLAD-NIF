# P1 — Realization-Dependent Microstructure and Statistical Morphology Emulation in Ballistic Helical GLAD

## Abstract

Glancing-angle deposition (GLAD) links a single controlled parameter, the deposition angle `α`,
to film porosity and to the optical anisotropy relevant to gas-sensor and polarimetric-device
performance, but the range of structural information a ballistic simulator can predict from `α`
has not been systematically established. We characterise a bead-sphere ballistic GLAD simulator
across twelve helical Cu configurations (`α = 60°–89°`, `h = 300 nm`, pitch `= 150 nm`). Void
fraction increases monotonically from `88.1%` to `98.5%` with sub-`0.25%` seed-to-seed variability
at this box/height convention (degrading at larger lateral box size; see Appendix on finite-size
convergence). A corotating-frame oracle test shows absolute column positions carry no
realization-invariant signal recoverable by a coordinate-conditioned model, establishing this
boundary of predictive reach as a central, deliberately reported result. By contrast, the
two-point pair correlation `h_g(r; α)` is reproducible and varies smoothly with `α`: a compact
emulator (**CorrNIF**) reproduces held-out angles with interior leave-one-angle-out `R² = 0.992`,
though not more accurately than simple linear or PCHIP interpolation on the same folds, and
reproduces a genuinely unseen prospective angle (`α = 72°`, confirmed by a dedicated simulation)
with `R² = 0.9907`, again no better than linear interpolation (`R² = 0.9909`). CorrNIF's only
demonstrated advantage over these simpler baselines is exercised directly: gradient descent
through the frozen, trained model recovers the deposition angle behind a held-out target curve
from five different starting points, matching an independent grid search to within `0.002°`. An
illustrative aligned-cylinder Maxwell-Garnett calculation for tungsten columns gives form
birefringence `Δn ~ 0.3–0.8`, but strong extraordinary absorption and sensitivity to the
filling-fraction definition mean no polarimetric design window can be claimed. The result is a
simulation-only workflow for ranking helical GLAD geometries by ensemble morphology, with the
limits of predictive reach established as an explicit finding.

## What's here

- `P1_ballistic_GLAD_scope_limits.tex` / `.bib` / `.pdf` — the paper (single-column, portable
  template), plus all figure PDFs it embeds.
- `aip_template/` — the same content reformatted for the intended submission venue (*Journal of
  Vacuum Science & Technology A*, AIP `revtex4-1` class).

## The one runnable result in this paper

The gradient-based inverse-design demonstration (new subsection, "Demonstrating the
differentiability advantage") is fully reproducible from the small data artifacts in this
repository with no GPU — see the main repository [README](../../README.md#reproducibility) for the
one-command quick start, and [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) for how every other
figure/table maps to a script.
