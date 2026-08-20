# Reproducibility map

Which script produces which figure/result in each paper. Paths are relative to the repository
root. Figure-generation scripts live in `code/figures/`; they read from either the small
artifacts in `data/small_artifacts/` or from full simulation output that is not included in this
repository (see the main [README](README.md) for why).

## P1 — `papers/P1_ballistic_GLAD_scope_limits/`

| Figure / result | Script | Needs full sim. data? |
|---|---|---|
| Void-fraction trend vs. α (Fig. "openness trend") | `code/figures/fig01_openness.py` | yes |
| Depth-resolved (z-profile) void fraction | `code/figures/fig02_zprofile.py` | yes |
| Morphology PCA latent space | `code/figures/fig03_pca.py` | yes |
| Void fraction vs. literature comparison | `code/figures/fig04_literature.py`, `gen_publication_figures_v1.py` | yes |
| Pair-correlation curves `h_g(r; α)` | `code/figures/fig05_pair_correlation.py` | no — uses `data/small_artifacts/corr_functions.npz` |
| CorrNIF LOOA prediction curves | `gen_publication_figures_v1.py` | partially (LOOA fold results) |
| Prospective α=72° validation figure | `code/figures/fig07_alpha72.py`, `code/nif_corrnif/run_alpha72_prospective_validation.py` | no — uses `corr_functions.npz` + `alpha72_prospective_results.json` |
| Calibration grid (train vs. held-out angle) | `code/figures/fig_calibration_grid_72.py` | no |
| Corotating-frame oracle maps / Table `tab:oracle` | `code/figures/fig_corotating_oracle_maps.py`, `code/nif_corrnif/oracle_corotating_ceiling.py`, `code/nif_corrnif/reproduce_alpha_z_oracle.py` | yes |
| **CorrNIF gradient-based inverse-design demo (Fig. "gradient descent recovery of α")** | `code/nif_corrnif/run_corrnif_gradient_inverse_demo.py` (results) + `code/figures/fig_corrnif_gradient_inverse_demo.py` (figure) | **no — fully reproducible from `data/small_artifacts/` alone, see main README quick-start** |
| CorrNIF/deposition schematic diagrams (conceptual) | `code/figures/fig_corrnif_architecture.py`, `fig_deposition_schematic.py` | no (conceptual, no data) |
| Maxwell-Garnett optical screening figure | `code/figures/fig_mg_concept.py`, `gen_publication_figures_v1.py` | yes |
| Morphology cross-section (real voxel slices) | `code/figures/fig_morphology_crosssection.py` | yes |
| Finite-size (box convergence) appendix figure | `code/figures/fig_boxconv_appendix.py` | yes |

## P2 — `papers/P2_density_calibrated_NIF_GLAD/`

| Figure / result | Script | Needs full sim. data? |
|---|---|---|
| NIF architecture diagram (conceptual) | `code/figures/fig_nif_architecture_p2.py` | no |
| Voxel-occupancy concept diagram (conceptual) | `code/figures/fig_voxel_occupancy_concept.py` | no |
| NIF v5 prediction vs. ground truth (full-field density) | `code/figures/fig_nifv5_pred_vs_truth.py` | yes — checkpoint forward pass |
| Soft-label / `f_soft` histogram | `code/figures/fig_soft_label_histogram.py` | yes |
| BCE-plateau training curve | `code/nif_corrnif/train_nif_levelA_helical.py` (training run output) | yes |
| Oracle Δ-IoU diagnostic | `code/nif_corrnif/oracle_corotating_ceiling.py`, `reproduce_alpha_z_oracle.py` | yes |
| Hard-label IoU vs. epoch | `code/nif_corrnif/train_nif_hardlabel_v2.py` (training run output) | yes |
| Three-fixes summary (Fix A/B/C) | `code/figures/fig_three_fixes.py` | yes |
| Fix A — hash-grid spatial encoding | `code/nif_corrnif/models/nif_hashgrid.py`, `code/nif_corrnif/train_nif_hardlabel_v2.py` | yes |
| Fix B — continuous-density regression | `code/nif_corrnif/train_nif_continuous_regression.py` | yes |
| Fix C — partial-realization conditioning | `code/nif_corrnif/models/nif_partial_conditioning.py`, `code/nif_corrnif/train_nif_partial_conditioning.py`, `code/nif_corrnif/build_levelA_helical_nif_partial_conditioning_dataset.py` | yes |
| Fix C permutation test (n=3→n=8 seeds) | `code/nif_corrnif/test_permutation_fixC.py` | yes |
| Base NIF model (FiLM-conditioned) | `code/nif_corrnif/models/nif_levelA_helical.py` | — |

## P3 — `papers/P3_cross_material_generalization_GLAD/`

P3's reproducibility story is different in kind from P1's and P2's. It reports no figures (every
result is a table) and does not use the ballistic simulator or any NIF/CorrNIF training code in
`code/simulation/` or `code/nif_corrnif/` — its inputs are a literature-derived, multi-material
corpus of `β`/`φ` measurements, not simulator output. Its analysis (leave-one-material-out
cross-validation of the four candidate models, the pre-registered permutation tests, the
corpus-growth interventions, and the Bayesian hierarchical partial-pooling model) was produced by
a separate set of literature-corpus analysis scripts (`P3_ML/` in the working project tree) that
are **not currently included in this repository's `code/` directory** — `code/` contains only
`figures/`, `nif_corrnif/`, and `simulation/`, none of which cover P3. This is a known gap, not an
oversight: bringing P3's analysis pipeline into this repository is tracked as separate follow-up
work. Until then, P3's reported numbers cannot be regenerated from what is currently in this repo.

## Simulator

| Component | Path |
|---|---|
| Core ballistic GLAD engine | `code/simulation/glad_v3_core.py` |
| Example campaign launcher | `code/simulation/run_levelA_staged_safe.py` |
| Simulation protocol / parameter conventions | `code/simulation/SIMULATION_MASTER_PROTOCOL.md` |

**"yes" in the "needs full sim. data?" column** means the script expects voxel `.npy` arrays or
training checkpoints that are not included in this repository (see main README). Everything
marked "no" runs end-to-end from what's already here.

**Note on paths:** scripts in the "yes" rows still contain the original project's absolute paths
(they were written to run inside the full, non-public project tree, and need data this repository
doesn't include regardless of path). The two scripts in the "no" rows
(`run_corrnif_gradient_inverse_demo.py` and `fig_corrnif_gradient_inverse_demo.py`) have been
adapted to resolve paths relative to this repository, so they run as-is after cloning.
