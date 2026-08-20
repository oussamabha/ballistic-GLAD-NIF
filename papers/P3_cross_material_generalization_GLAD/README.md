# P3 — Cross-Material Generalization for GLAD Column-Tilt and Porosity: A Pre-Registered Negative Result and Its Statistical Boundary

## Abstract

Existing predictive relations for GLAD column-tilt angle `β` and porosity `φ` are either purely
geometric (the tangent rule for `β`) or material-specific empirical fits; no published model uses
material identity itself — via physically motivated descriptors such as atomic mass, melting
point, or bulk density — to predict how a *new* material, never seen during fitting, will behave
under GLAD. We assembled a literature-derived corpus spanning up to 40 distinct materials
(elemental metals, oxides, fluorides, chalcogenides), fitted four candidate models (linear
regression, Gaussian process, gradient-boosted trees, random forest) under leave-one-material-out
cross-validation, and pre-registered a strict two-part acceptance bar (≥15% RMSE improvement over
a physics baseline *and* a material-level permutation-test `p<0.05`) before inspecting any result.
Every learned model comfortably clears the RMSE bar for `β(α)` (up to +54.4%) and often for
`φ(α)` (up to +28.5%), but no configuration — the pooled fit, three pre-registered material-class
subgroups, two independent corpus-growth interventions targeting the most promising subgroup, or a
pairwise ranking reformulation of the task — ever clears the permutation-test bar (`p=0.20`–`0.36`
across every variant at the current corpus snapshot). We systematically tested and ruled out four
candidate explanations (wrong features, wrong material subgroup, wrong task formulation,
insufficient sample size), isolating material-level sample size as the specific barrier. A
complementary, deliberately different Bayesian hierarchical partial-pooling model finds that
atomic mass has a credible non-zero effect on a material's mean offset from the tangent rule (94%
HDI excluding zero), explaining — not contradicting — the frequentist null: a real-but-modest
effect of this size is exactly what a permutation test at `n=14`–`40` materials is underpowered to
detect. We present this as a fully pre-registered, diagnostically exhaustive account of where the
statistical-power boundary for this class of cross-material generalization problem currently sits.

## What's here

- `P3_cross_material_generalization_GLAD.tex` / `.bib` / `.pdf` — the paper (single-column,
  portable template). No figures — every result is reported as a table.
- `aip_template/` — the same content reformatted for AIP `revtex4-1` (target venue not yet
  finalized; the paper's own submission package README flags *Machine Learning: Science and
  Technology* or *npj Computational Materials* as possibly better fits than a standard AIP venue
  given its negative-result/Bayesian framing).

## Reading this alongside P1 and P2

P1 characterizes the underlying ballistic simulator; P2 builds a learned per-simulation surrogate
for it; this paper asks a different, corpus-level question — given real literature measurements
across many materials, can material identity itself predict β/φ response to GLAD. It draws on the
same physical quantities (β, porosity) as P1/P2 but a disjoint, literature-derived dataset (not
simulation output), and is independently readable and citable.
