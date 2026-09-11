# Data dictionary

Schema and vocabulary for the released data artefacts. FAIR-steward requirement for the
paper data releases (JVST-A / Thin Solid Films / J. Appl. Phys. tier all ask for it).

## `parameter_database` (literature parameter corpus -- `data/` export)

One row = one measured or model-derived parameter value from one source.

| column | meaning | notes / vocabulary |
|---|---|---|
| `id` | integer primary key | export-local; not stable across releases |
| `parameter_id` | short human key, e.g. `BENAMOR2026_BETA_A85` | stable within a release |
| `parameter_name_standard` | canonical parameter name | e.g. `column_tilt_angle_beta`, `void_fraction`, `refractive_index`, `band_gap_Eg`, `diffusion_barrier_Ea`, `melting_point` |
| `parameter_category` | grouping | `structural` / `optical` / `electronic` / `kinetic` / `thermophysical` / `process` |
| `value` | the value as text | ranges stored as `"a--b"`; see `unit` |
| `unit` | SI-ish unit string | `deg`, `%`, `nm`, `eV`, `K`, `nm/s`, `dimensionless` |
| `material` | material system | e.g. `Cu`, `Cu2O`, `CuO`, `WO3`, `TiO2`, `SiO2`, `HfO2`, `Mo`, `W`, `Au` |
| `article_id` | source key | matches `article_registry`; web-sourced rows carry a `..._WEBESCALATION_<date>` provenance in `row_origin` |
| `article_title` | source title | free text |
| `status` | curation state | `confirmed_primary_source` / `literature_needs_verification` / `derived` / `superseded` |
| `evidence_strength` | how load-bearing | `PRIMARY` / `STRONG_GLAD_ANALOGUE` / `WEAK_CONTEXT` / `NOT_TRANSFERABLE` |
| `extraction_confidence` | how the value was read | `HIGH` (direct table/text read) / `MEDIUM` (WebFetch summary of an open page) / `LOW` (snippet only -- not inserted) |
| `exact_source_location` | where in the source | e.g. `Table 4`, `Fig. 5b`, `p. 15468` |
| `original_file_path` | local provenance path | private-repo path; blank in the public export |
| `notes` | free text | caveats, unit-ambiguity resolutions, cross-checks |
| `row_json` | the full extracted record | JSON blob, machine-readable superset of the columns |
| `source_file` / `source_sha256` | the exact file the value came from + its hash | for the byte-level audit trail; hash retained in the public export, path stripped |
| `imported_at` | ISO timestamp of insertion | |
| `row_origin` | provenance tag | `CANONICAL`, `CANDIDATE_WEBESCALATION_<YYYYMMDD>`, `CANDIDATE_<batch>` |
| `canonical_status` | is this the canonical row for its (parameter, material, angle)? | `ACTIVE` / `SUPERSEDED` |

### Known caveats carried in the data (state these in any paper using the corpus)
- `void_fraction` from the simulator is a **bead-sphere** measure; literature values are
  **optical Bruggeman** or **SEM** -- a documented method-family gap of ~1-3 pp (see
  `P1_POROSITY_DEFENSIBILITY_ASSESSMENT`), plus a lateral box-size dependence of the
  absolute value (P1 Appendix box-convergence).
- `column_tilt_angle_beta` for the simulator is the **structure-tensor** estimator
  (`code/.../beta_structure_tensor.py`); an earlier watershed estimator is superseded
  (biased low 15-20 deg on dense films). See `P1_ESTIMATOR_PROVENANCE`.
- The SiO2 corpus source (`article_id` containing `ma18102225`) was mis-attributed to
  "Romero-Perez" in one 2026-08-16 web-escalation record; correct first author is
  **B. Jia** (Shanghai Inst. of Technical Physics, CAS). DOI/title/year are correct.
  Article-id renamed to `jia2025_ma18102225_sio2_oad` on 2026-09-08 (SQLite candidate rows).

## `article_registry`
`article_id`, `article_title`, `source_path` (stripped in export), `analyzed_status`,
`absorption_status` (`ABSORBED` / `PENDING`), `duplicate_status`, `extraction_status`
(`FULL_TEXT_READ` / `ABSTRACT_ONLY` / `TEXT_OK`), `row_origin`, `canonical_status`.

## Analysis-code artefacts (`code/`)
| script | input | output | notes |
|---|---|---|---|
| `beta_structure_tensor.py` | checkpoint `.h5` | median beta + 16-84 band | density-field structure tensor; the P1 beta estimator |
| `beta_highalpha_ccl.py` | checkpoint `.h5` | beta from 3-D connected components | cross-check, alpha >= 87 only |
| `overlap_graded_pbc.py` | checkpoint `.h5` | graded interpenetration fraction | periodic min-image, `nn < cd(1-t)` |
| `lens1_wp1_corrnif_jacobian.py` | `corr_functions.npz` | \|dh_g/dalpha\|(alpha), sigma_alpha | inverse-map conditioning |
| `lens1_wp3_position_alpha_mi.py` | 9 replicate `.h5` | I(position;alpha) vs null | Ross-2014 mixed k-NN MI |
| `save_corrnif_frozen_checkpoint.py` | `corr_functions.npz` | `corrnif_alpha_only_seed0.pt` | the frozen inversion model |

## Frozen simulator core
`code/simulation/glad_v3_core.py` -- version-frozen; SHA-256 recorded in
`REPRODUCIBILITY.md`. The grid-rebuild correction is commit `aa32952`; the
diffusion-kernel flaw + fix is `01_.../DIFFUSION_KERNEL_MODEL_FLAW_AND_FIX_20260907.md`.
