# Ballistic Helical GLAD: Statistical Morphology, Neural Implicit Fields, and Their Predictive Limits

This repository accompanies three companion manuscripts studying **glancing-angle deposition
(GLAD)** — a thin-film growth technique that exploits extreme substrate tilt (deposition angle
`α > 70°`) to self-assemble tunable, porous, columnar nanostructures. GLAD films are used as
gas-sensing platforms and as polarimetric/optical elements (waveplates, chiral filters,
biosensors), because their column geometry controls both porosity and optical anisotropy through
a single experimentally accessible knob: `α`.

P1 and P2 ask the same underlying question from two different angles: **what can a
physically-motivated ballistic growth model, and a machine-learning surrogate trained on its
output, actually predict about GLAD morphology — and where, precisely, does that predictive power
run out?** P3 asks a related but genuinely different question, moving from a single simulator to a
literature-derived corpus spanning many materials: **can material identity itself — not just
deposition angle — predict how GLAD column-tilt and porosity respond, for a material never seen
during fitting, and if so, how confidently?** Answering both questions honestly (including the
negative results) is the scientific content of all three papers, not an afterthought.

> **Status:** pre-submission draft repository, kept private while the work is finalized. Not yet
> peer-reviewed. See each paper's own PDF for the current abstract and scope.

---

## The three papers

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

### P3 — [`papers/P3_cross_material_generalization_GLAD/`](papers/P3_cross_material_generalization_GLAD/)
*Cross-Material Generalization for GLAD Column-Tilt and Porosity: A Pre-Registered Negative Result
and Its Statistical Boundary*

Assembles a literature-derived corpus spanning up to 40 distinct materials (elemental metals,
oxides, fluorides, chalcogenides) and asks whether material identity itself — via descriptors such
as atomic mass, melting point, or bulk density — can predict how a *new* material, never seen
during fitting, will behave under GLAD, for both column-tilt angle `β` and porosity `φ`. Under a
strict, pre-registered two-part acceptance bar (≥15% RMSE improvement over a physics baseline
*and* a material-level permutation-test `p<0.05`) evaluated by leave-one-material-out
cross-validation, every learned model comfortably clears the RMSE bar for `β(α)` (up to +54.4%)
and often for `φ(α)` (up to +28.5%), but no configuration — the pooled fit, three pre-registered
material-class subgroups, two independent corpus-growth interventions, or a pairwise ranking
reformulation — ever clears the permutation-test bar (`p=0.20`–`0.36` across every variant at the
current corpus snapshot). Four candidate explanations are systematically tested and ruled out,
isolating material-level sample size as the specific barrier; a complementary Bayesian
hierarchical partial-pooling model finds a credible non-zero effect of atomic mass (94% HDI
excluding zero), explaining rather than contradicting the frequentist null. The paper reports this
as a fully pre-registered, diagnostically exhaustive account of where the statistical-power
boundary for this class of problem currently sits.

**How they relate:** P1 motivates and characterises the underlying ballistic simulator; P2 builds
a learned surrogate for it and studies exactly how far that surrogate can go. P3 is a different
kind of companion: it draws on the same physical quantities (`β`, porosity) but works from a
disjoint, literature-derived corpus across many real materials rather than simulator output, asking
a corpus-level generalization question instead of characterising a single simulation campaign. It
is independently readable and citable, and does not depend on P1's or P2's code or data. A
cross-paper caveat worth knowing if you read P1 and P2 closely together: the two papers compute
"occupied voxel fraction" over different `z`-ranges of the film in different tables — each paper
discloses this explicitly where it matters (P1's Table on the oracle test; P2's Limitations
section) rather than leaving it as a silent inconsistency.

Each of P1's and P2's paper folders also contains an `aip_template/` subfolder with the same
content reformatted for its intended submission venue (P1 → *Journal of Vacuum Science & Technology
A*; P2 → *AIP Advances*); the main folder holds the more portable single-column version. P3 also
has an `aip_template/` subfolder, but its target venue is not yet finalized — P3's own README notes
that *Machine Learning: Science and Technology* or *npj Computational Materials* may be a better
fit than a standard AIP venue, given its negative-result/Bayesian framing.

### P4 — [`papers/P4_atomistic_anchor_kernel_limits/`](papers/P4_atomistic_anchor_kernel_limits/)
*An Atomistic Calibration Anchor for a Ballistic GLAD Simulator: Cu(100) Adatom Energetics and the
Limits of a Grid-Local Surface-Diffusion Kernel*

A short methods paper, independent of P1-P3's simulator/NIF pipeline. Computes the Cu(100) adatom
hop barrier by nudged elastic band on the Mishin et al. (2001) EAM potential (`E_a = 0.5106 eV`,
agreeing with three literature values to 0.02-0.03 eV), converts it to a room-temperature diffusion
length before burial (`L_D ~ 20-30 nm`), and shows — via a geometry-kinetics decomposition at two
height-matched deposition angles — that the ballistic simulator's built-in surface-diffusion
kernel (grid-local, ~1.5 nm reach) cannot represent transport at the `L_D` scale by any parameter
choice. Serves both as a calibration anchor and as a specification for a future range-extended
kernel; deliberately does not claim a calibrated ballistic run (that is out of scope, named
explicitly as future work). Contains the NEB result JSON and driver script, the manuscript, bib,
and the two result figures with their generator script.

### P5 — [`papers/P5_material_generalization_ballistic_GLAD/`](papers/P5_material_generalization_ballistic_GLAD/)
*Deposition Angle, Not Material Identity, Sets Void Fraction in Purely Ballistic
Glancing-Angle Deposition: A 14-Metal Simulation Study*

A controlled simulation study, independent of the P1-P3 pipeline and P4's atomistic
scope. Runs the same purely ballistic deposition model for 14 elemental metals
(melting points 923-3695 K) at three deposition angles, changing nothing between
materials except each metal's tabulated covalent radius. Finds void fraction is
material-invariant to <1 pp at every angle despite atom count varying by more than
2x, and that a pre-registered melting-point-dependence test for the column-tilt
offset fails (|r|=0.41, permutation p=0.16, n=13) -- material identity enters
ballistic GLAD morphology only through the atomic length scale, not the packing.
Complementary to the literature-corpus negative result reported separately (a
companion, non-staged manuscript). Contains the manuscript, bib, the two result
figures with generator script, and the underlying per-run regression CSV/script.

### P6 — [`papers/P6_fullwave_optics_real_GLAD_morphology/`](papers/P6_fullwave_optics_real_GLAD_morphology/)
*Full-Wave Optical Response of Simulated Cu Glancing-Angle Films: FDTD on
Voxelised Ballistic Morphology versus Measured Mueller-Matrix Ellipsometry*

Runs finite-difference time-domain (FDTD, MEEP) electromagnetic simulations
directly on the voxelised morphology of the ballistic GLAD model, with a
validated Nicolson-Ross-Weir index retrieval, literature material dispersion,
and an oxide core-shell geometry, then compares to a real Cu GLAD film measured
by Mueller-matrix ellipsometry. Three of four adjustable Mueller elements move
toward the measured value under realisation averaging; the dominant
depolarisation channel is shown to be structurally inaccessible to that
averaging method, identifying depolarisation modelling as the principal open
problem. A nine-angle void-fraction/index sweep is also reported. Contains the
manuscript, bib, and the two result figures with generator script.

---

## Repository structure

```
papers/
  P1_ballistic_GLAD_scope_limits/          LaTeX source, bibliography, figures, compiled PDF
  P2_density_calibrated_NIF_GLAD/          same, for P2
  P3_cross_material_generalization_GLAD/   same, for P3 (no figures — all results are tables)
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
