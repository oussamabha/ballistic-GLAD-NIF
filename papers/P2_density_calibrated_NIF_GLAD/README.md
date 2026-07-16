# P2 — Density-Calibrated Neural Implicit Field for Ballistic Helical GLAD: Scalar Porosity Prediction, Label-Quality Diagnosis, and the Path to Binary Voxel Occupancy

## Abstract

Neural implicit fields (NIFs) are a compact, coordinate-based candidate surrogate for ballistic
helical glancing-angle deposition (GLAD) simulations, potentially replacing a full simulator run
with a fast lookup from deposition angle `α` and spatial coordinates `(x,y,z)` to film morphology.
We report three successive results. A FiLM-conditioned NIF variant reconstructs the full 3D
density field with mean spatial Pearson `r = 0.984` within a `75°–87°` training window (positive
within-window only). Trained instead on binary voxel occupancy with Gaussian-smeared labels, the
same architecture collapses to the trivial predictor at every angle (`ΔIoU < 0.003`); an
independently reproduced `(α, z)`-oracle diagnostic shows this is a labelling-scheme failure, not
an architecture or depth-signal limitation — 90–98% of soft labels fall in `(0.1, 0.9)`,
eliminating binary boundary gradients. Hard binary labels restore genuine spatial learning but
only at the sparsest film (`α = 89°`, `ΔIoU = +0.029`); denser angles remain trivial. Three further
remedies — a multi-resolution hash-grid encoder (a higher-capacity local spatial encoding, not a
structural inductive bias in the CNN/GNN sense), continuous-density regression, and
partial-realization conditioning — all fail to produce non-trivial spatial learning at the
multi-seed test angles. For partial-realization conditioning, we directly tested whether too few
realizations (three per angle) was the cause by enlarging one angle to eight realizations and
retraining: the permutation test still fails for both FiLM variants (`p = 0.35`, `p = 0.53`; 56
cross-realization pairs), ruling out the practically-achievable data-scarcity explanation. The
result is a verified, stepwise diagnostic framework for voxel-level occupancy prediction in GLAD
simulators, and a tested, strengthened negative conclusion: no remedy explored here, including
more data, resolves the underlying non-identifiability.

## What's here

- `P2_density_calibrated_NIF_GLAD.tex` / `.bib` / `.pdf` — the paper (single-column, portable
  template), plus all figure PDFs it embeds.
- `aip_template/` — the same content reformatted for the intended submission venue (*AIP
  Advances*, AIP `revtex4-1` class).

## Reading this alongside P1

P1 characterises and motivates the underlying simulator; this paper builds a learned surrogate for
it. If you read both closely: they compute "occupied voxel fraction" over different `z`-ranges of
the film in some tables, which is disclosed explicitly in this paper's Limitations section (and in
P1's oracle-test table) rather than left as a silent inconsistency — see
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) for exactly which script/table each number comes
from.
