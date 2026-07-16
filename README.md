# Ballistic Helical GLAD: Statistical Morphology, Neural Implicit Fields, and Their Predictive Limits

This repository accompanies two companion manuscripts studying **glancing-angle deposition
(GLAD)** — a thin-film growth technique that exploits extreme substrate tilt (deposition angle
`α > 70°`) to self-assemble tunable, porous, columnar nanostructures. GLAD films are used as
gas-sensing platforms and as polarimetric/optical elements (waveplates, chiral filters,
biosensors), because their column geometry controls both porosity and optical anisotropy through
a single experimentally accessible knob: `α`.

Both papers ask the same underlying question from two different angles: **what can a
physically-motivated ballistic growth model, and a machine-learning surrogate trained on its
output, actually predict about GLAD morphology — and where, precisely, does that predictive power
run out?** Answering that honestly (including the negative results) is the scientific content of
both papers, not an afterthought.

> **Status:** pre-submission draft repository, kept private while the work is finalized. Not yet
> peer-reviewed. See each paper's own PDF for the current abstract and scope.

---

## The two papers

### P1 — [`papers/P1_ballistic_GLAD_scope_limits/`](papers/P1_ballistic_GLAD_scope_limits/)
*Realization-Dependent Microstructure and Statistical Morphology Emulation in Ballistic Helical
GLAD*

Characterises a bead-sphere ballistic GLAD simulator across twelve helical Cu configurations
(`α = 60°–89°`). Establishes which morphological quantities are reproducible and predictable
(void fraction, the two-point pair-correlation function) and which are not (absolute per-voxel
column position, which is independently re-randomised by the simulator's stochastic nucleation on
every run). Trains a compact differentiable emulator, **CorrNIF**, on the pair-correlation
function and demonstrates its one genuine advantage over simple interpolation — gradient-based
inverse recovery of `α` from a target correlation shape — directly, with a runnable script (see
[Reproducibility](#reproducibility) below).

### P2 — [`papers/P2_density_calibrated_NIF_GLAD/`](papers/P2_density_calibrated_NIF_GLAD/)
*Density-Calibrated Neural Implicit Field for Ballistic Helical GLAD: Scalar Porosity Prediction,
Label-Quality Diagnosis, and the Path to Binary Voxel Occupancy*

A deeper, dedicated investigation of a neural implicit field (NIF) trained directly on simulator
voxel output. Shows a genuine positive result (continuous density-field reconstruction, mean
spatial Pearson `r = 0.984`, within a narrow `α ∈ [75°, 87°]` window) alongside a carefully
diagnosed negative one (binary voxel occupancy is *not* learnable from data pooled across
independently-nucleated realizations, tested against three structurally different remedies, none
of which resolve it). Establishes a general label-quality diagnosis framework for NIF training on
simulation output.

**How they relate:** P1 motivates and characterises the underlying simulator; P2 builds a
learned surrogate for it and studies exactly how far that surrogate can go. Both share a bead-sphere
Cu GLAD simulation campaign and cite each other as companion work. A cross-paper caveat worth
knowing if you read both closely: the two papers compute "occupied voxel fraction" over
different `z`-ranges of the film in different tables — each paper discloses this explicitly where
it matters (P1's Table on the oracle test; P2's Limitations section) rather than leaving it as a
silent inconsistency.

Each paper folder also contains an `aip_template/` subfolder with the same content reformatted for
its intended submission venue (P1 → *Journal of Vacuum Science & Technology A*; P2 → *AIP
Advances*); the main folder holds the more portable single-column version.

---

## Repository structure

```
papers/
  P1_ballistic_GLAD_scope_limits/     LaTeX source, bibliography, figures, compiled PDF
  P2_density_calibrated_NIF_GLAD/     same, for P2
code/
  simulation/                         the ballistic GLAD simulator (glad_v3_core.py) + a launcher
  nif_corrnif/                        every NIF / CorrNIF training, evaluation, and dataset-
                                       building script referenced by either paper
data/
  small_artifacts/                    small (<1 MB each) precomputed results/datasets needed to
                                       regenerate the reported numbers and figures without
                                       re-running the full simulator or retraining from scratch
REPRODUCIBILITY.md                    maps specific paper figures/tables to the script that
                                       produces them
requirements.txt
```

### What is *not* included, and why

The full simulation campaign (raw per-voxel `.npy` arrays) and NIF training checkpoints total
several tens of GB, which does not belong in a git repository. Only the small, already-aggregated
result files needed to reproduce the papers' reported numbers and figures are included under
`data/small_artifacts/`. The raw voxel datasets and training runs are available on request (see
each paper's Data Availability statement) and are candidates for a separate archival deposit
(e.g. Zenodo) alongside a public release of this repository.

---

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for a table mapping specific figures/results in
each paper to the exact script that generates them.

As a self-contained starting example, the CorrNIF gradient-based inverse-design demonstration in
P1 (the direct exercise of the "differentiable representation" claim) can be reproduced end-to-end
from the small artifacts already included, with no GPU and no access to raw simulation data:

```bash
pip install -r requirements.txt
python code/nif_corrnif/run_corrnif_gradient_inverse_demo.py
```

This retrains CorrNIF from `data/small_artifacts/corr_functions.npz` (~1 minute, CPU), then
recovers the deposition angle behind a held-out target curve by gradient descent from five
different starting points, cross-validated against an independent brute-force grid search — the
exact experiment reported in P1's new subsection "Demonstrating the differentiability advantage."

---

## Environment

The simulator (`code/simulation/glad_v3_core.py`) needs an NVIDIA GPU with CUDA and `cupy` for its
ray-marching kernel. Everything under `code/nif_corrnif/` (dataset building, NIF/CorrNIF training
and evaluation) runs on CPU or GPU via plain PyTorch. See `requirements.txt`.

## License / citation

No license has been assigned yet — this repository is private and pre-publication. Please contact
the author before reusing any code or data here. A citation entry will be added once the papers
are submitted/published.
