# SIMULATION_MASTER_PROTOCOL.md

**Status:** MANDATORY_REFERENCE_FOR_ALL_FUTURE_SIMULATIONS  
**Adopted:** 2026-06-23  
**Task:** PROJECT_ORGANIZATION_AND_MASTER_PROMPT_UPDATE_V1  
**Cross-references:** `CURRENT_PROJECT_DECISION_STATE.md` (project state); `PROTOCOLS/05_SIMULATION_PROTOCOL.md` (authorization rules); `PROTOCOLS/07_CHECKPOINT_SAFETY_PROTOCOL.md` (checkpoint integrity); `MASTER_PROJECT_AUDIT_LEDGER.md` (run history)

---

> **Read this file before launching ANY simulation.** It supersedes informal notes scattered across
> ledger entries and pending-review folders. For authorization rules (do not launch without explicit
> authorization) see `PROTOCOLS/05_SIMULATION_PROTOCOL.md`. This file focuses on calibrated parameters,
> safety checks, and campaign governance.

---

## Section 1 — CALIBRATED Cu PHYSICAL PARAMETERS

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| `r` (Cu hard-sphere radius) | **0.128 nm** | LITERATURE_CALIBRATED | Cordero 2008, Cu FCC covalent radius. **CANDIDATE FOR NEW CAMPAIGN — see Section 5.** The legacy baseline used r inferred from density; 0.128 nm is the physically correct value. |
| `pitch` | **150 nm** | PRESCRIBED_NUMERICAL_REFERENCE | Groupe Chaffar prescribed value. NOT derived from a formula or validated from Cu SEM. Unit ambiguity in the source formula unresolved (see `99_PENDING_REVIEW/P1_PARAMETER_SELECTION_REAUDIT/PITCH_150NM_ORIGIN_AND_UNIT_AUDIT.md`). |
| `batch_size` | **512** | VALIDATED_NUMERICAL_PARAMETER | Validated against Level-A baseline; topology-invariant at α=89°/300nm (CONNECTED_FILM LF=0.974 vs LF=0.980 for bs=2048, 84% atom-count difference). **No physical analogue — do not change without prior topology-invariance test.** |
| `substrate_spacing` | **1.0 nm** | CURRENT_BASELINE | Control parameter; isolatability as a causal variable is UNCONFIRMED pending read-only audit. |
| `Ea_diffusion` | **0.041 eV** | UNCALIBRATED_EFFECTIVE | Cu/Cu(111) homoepitaxy reference — too low for Cu/CuOx. Diffusion is disabled in the P1 baseline kernel. **For any future physical test using diffusion:** use Ea ∈ {0.10, 0.20, 0.30} eV (Cu/oxide analogues). Do NOT use 0.041 eV as a calibrated Cu value. |
| `sticking` (probabilistic) | **1.0 (default)** | SENSITIVITY_ABLATION_ONLY | No Cu/Cu₂O/CuO first-contact sticking coefficient in the 35-article complementary corpus. Ablation at α=89°/300nm/helical showed no topology change for s∈{0.2, 0.5, 1.0} (all CONNECTED_FILM). Any sticking s<1.0 is hypothetical, not Cu-calibrated. |
| `random_seed` | **must be explicit** | BUG_FIXED_2026-06-22 | **Sentinel bug:** `random_seed=0` was silently unseeded (falsy-zero gate `if chosen_seed:` + absent key in defaults dict). Fixed in core hash `fd9b9634`. Seed=0 is now valid and deterministic. Always specify seed explicitly; never rely on default. |

---

## Section 2 — GPU ANTI-THROTTLING PROTOCOL

**Rule: always check GPU temperature before a long run.**

```bash
# Check temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# If T > 80°C: wait 5 minutes before launching
# If T > 90°C: stop and investigate cooling
```

Additional GPU rules:
- `batch_size=512` is the only validated value — do not change without invariance test (see Section 1)
- **Single-shot execution preferred** — staged-resume is allowed only for α=60° (hardware-infeasible single-shot at box150; box100 single-shot is fine)
- **One run at a time** — never launch parallel simulations; GPU memory is shared and thermal load stacks
- α=60° box150 is hardware-infeasible (~2.3 GB positions > 4 GB GPU VRAM) — do not attempt

---

## Section 3 — MEMORY OVERFLOW PREVENTION

**Always verify GPU memory headroom before launching:**

```bash
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader
```

| Configuration | Atom count (approx.) | GPU RAM req. | Status |
|---------------|----------------------|-------------|--------|
| α=89° box100 (300 nm) | ~11 M | ~0.3 GB | Fast (~3 min) |
| α=85° box100 (300 nm) | ~20 M | ~0.5 GB | OK |
| α=80° box100 (300 nm) | ~24 M | ~0.6 GB | OK |
| α=75° box100 (300 nm) | ~35 M | ~0.9 GB | OK |
| α=60° box100 (300 nm) | ~86 M | ~2.1 GB | OK (tight) |
| α=60° box150 (300 nm) | ~280 M est. | ~6–7 GB est. | **HARDWARE_INFEASIBLE** |

Minimum recommended free VRAM before launch: **headroom ≥ 2× expected atom-count RAM** (accounts for gradient buffers and intermediate arrays).

### Section 3b — GPU VRAM overflow via CUDA managed memory (added 2026-07-03)

The `HARDWARE_INFEASIBLE` label above (α=60° box150, and by extension any box≳200nm at r=0.128)
is **no longer an absolute hardware ceiling** as of 2026-07-03. Root cause was re-investigated:
large box sizes hit `[GRID-PREFLIGHT] ABORT ... std::bad_alloc` from **progressive GPU memory
fragmentation across successive grid rebuilds** during growth — not fixable by tuning
`active_window_height` or `target_height` (both tried and ruled out; confirmed via direct
`cp.cuda.runtime.memGetInfo()` tracing that free VRAM is ample right after grid init and depletes
only after several rebuild cycles).

**Fix applied to `glad_v3_core.py` (~line 175):**
```python
# before:
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
# after:
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
```
This switches CuPy's allocator to CUDA Unified (Managed) Memory, which transparently pages
allocations to host RAM once the 4GB VRAM budget is exhausted, instead of raising
`std::bad_alloc`. **Verified standalone** (cupy 14.1.0): successfully allocated 5.6GB (>4GB VRAM)
via `cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc`.

**Tradeoffs — read before using:**
- **~100x slower** once paging is active: observed ~1,300 rays/s (managed, box=200nm, mid-run)
  vs ~114,000 rays/s (pure VRAM, box=100-150nm, comparable stage). A run that took ~10 min on
  pure VRAM can take several hours once paging kicks in.
- Host RAM (not GPU VRAM) becomes the new ceiling — check `free -h` before launching; WSL2's
  own RAM allocation is capped by `.wslconfig` (`memory=`), which may itself be well below the
  Windows host's total physical RAM (see AGENTS.md note on the separate host-RAM bottleneck for
  CPU-side analyses like Item 1's contact-graph runs).
- Does **not** change simulation physics — same ray-sphere kernel, same physical model, only
  where the atom-position data physically resides. Safe to combine with any `author_radius_nm`,
  `pitch`, `alpha` value.
- **Use only when a run is otherwise VRAM-blocked** (box≳175-200nm at r=0.128, or comparable).
  For box≤150nm, keep the plain `MemoryPool().malloc` allocator (faster, no reason to pay the
  paging cost) — this would require reverting the allocator line for those campaigns, or running
  a build/branch that keeps both options; not implemented as a runtime toggle in this pass.

**Before touching this line again**: check `MASTER_PROJECT_AUDIT_LEDGER.md` for the
"GPU VRAM overflow via CUDA managed memory" entry (2026-07-03) — it has the full diagnostic
trail (what was ruled out: active-window shrinkage, height reduction, curve-fit extrapolation)
so you don't have to re-derive why this was needed.

---

## Section 4 — MANDATORY VERIFICATION PROTOCOL

### Before every run

1. **Confirm core hash** — `glad_v3_core.py` SHA-256 prefix must be `fd9b9634` (post-sentinel-fix frozen version)
   ```bash
   sha256sum "01_GLAD_SIMULATION/simulation MD/src/glad_v3_core.py" | cut -c1-8
   # Expected: fd9b9634
   ```

2. **Dry-run config check** — print resolved parameter values before committing to a run
   > Note: the universal GLAD runner's `--dry-run` writes per-stage summary files even though it launches no physics. Classify as `DRY_RUN_WITH_SIDE_EFFECTS`, not `READ_ONLY`. Run only in an isolated test directory.

3. **Smoke-test** — run 1 short foreground run at h=30 nm before any new campaign:
   - Confirms the config resolves correctly
   - Confirms GPU launches without error
   - Topology at 30 nm is ISOLATED_COLUMNS (normal), not CONNECTED_FILM — this does NOT validate the mature-film result

4. **Sticking check** — verify `enable_sticking=False` (default baseline). Any deviation from default is a SENSITIVITY_ABLATION run and must be labelled as such.

5. **Seed check** — verify `random_seed` is explicitly defined in config (never omit). Seed=0 is now valid after the sentinel-fix but must be declared explicitly.

### After every run

1. **SHA-256 checkpoint** — hash the output `.h5` file and record in run metadata
2. **HDF5 attrs** — verify `height_nm` attribute matches target height (tolerance ±0.5 nm)
3. **Atom count plausibility** — compare to baseline for same α (expect CV < 0.2%)
4. **Status marker** — write `STATUS_COMPLETED` or `STATUS_FAILED` marker file immediately after run
5. **Ledger entry** — append to `MASTER_PROJECT_AUDIT_LEDGER.md` before starting any analysis

---

## Section 5 — NEW CAMPAIGN: r=0.128 nm (AUTHORISED, NOT YET EXECUTED)

| Field | Value |
|-------|-------|
| Status | `AUTHORISED_NOT_EXECUTED` |
| Priority | **HIGH** — required before submission to journals above Thin Solid Films tier |
| Campaign root | `01_GLAD_SIMULATION/simulation_batch/runs/P1_CALIBRATED_r128_CAMPAIGN_V1/` |
| Physical change | `r = 0.128 nm` (Cordero 2008 Cu hard-sphere) vs legacy baseline r |
| All other params | Identical to P1 baseline (α sweep, pitch=150nm, h=300nm, helical, seeds 0/1/2) |
| Expected output | Full topology + openness sweep confirming or revising the P1 morphological claims |
| Gate before execution | Read-only audit of how `r` enters the simulator (density formula, contact distance, substrate_spacing interaction) — confirm no implicit coupling to other parameters |
| Forbidden | Do not launch before completing the read-only r-parameter audit |

---

## Section 6 — GITHUB / ZENODO READINESS

### Files to include in the publication package

| File | Location | Notes |
|------|----------|-------|
| `glad_v3_core.py` | `01_GLAD_SIMULATION/simulation MD/src/` | Frozen at hash `fd9b9634`. Include verbatim; do not clean up. |
| `topology_analysis_v1.py` | `01_GLAD_SIMULATION/simulation_batch/runs/P1_STATIC_VS_HELICAL_TOPOLOGY_V1/` | Frozen; used for all topology classifications. |
| `topology_sweep_all27_RESULTS.json` | (locate in topology sweep run dir) | 27-run topology outcome table. |
| `P1_v2_clean_homogeneous_20260620.tex` | `05_THESIS_PAPER_ASSETS/article_draft/` | Current manuscript source (v22 as of 2026-06-23). |
| `P1_references.bib` | `05_THESIS_PAPER_ASSETS/article_draft/` | BibTeX bibliography. |
| `requirements.txt` | `gladwsl` conda environment | List Python/CUDA dependencies. |

### Files to EXCLUDE

| File type | Reason |
|-----------|--------|
| HDF5 checkpoints (`*.h5`) | Too large (>2 GB each); host on Zenodo as separate data deposit |
| Voxel arrays (`*.npy`) | Too large; Zenodo data deposit |
| Personal protocols and internal governance docs | Not for public release (`PROTOCOLS/`, `MASTER_PROJECT_AUDIT_LEDGER.md`, etc.) |
| Backup files (`*.backup_*`, `*.bak_*`) | Not for release |
| Pending-review and scratchpad files | Not for release |

### Current readiness status (2026-06-23)

`05_THESIS_PAPER_ASSETS/GITHUB_ZENODO_REPOSITORY_PACKAGE/` is **OUTDATED** — see Part 5 audit in the task report for files requiring update before Zenodo submission.

---

## See Also

- `PROTOCOLS/05_SIMULATION_PROTOCOL.md` — authorization rules, run classification, runner governance
- `PROTOCOLS/07_CHECKPOINT_SAFETY_PROTOCOL.md` — active-slab incident, core-hash compatibility
- `PROTOCOLS/12_SCIENTIFIC_REASONING_PROTOCOL.md` — simulation output ≠ experimental validation
- `CURRENT_PROJECT_DECISION_STATE.md` — current model posture (Option B, no further baseline runs needed)
- `MASTER_PROJECT_AUDIT_LEDGER.md` — full run history with SHA-256 provenance

---

## Verification Tools (adopted 2026-06-25)

Two scripts are now mandatory before/after simulation sessions:

### `simulation_batch/preflight_param_check.py` — Pre-launch check
Compares pipeline JSON defaults vs `glad_v3_core.py` constants vs protocol locked values.
Run before every simulation launch. If exit code ≠ 0 → DO NOT LAUNCH.
```bash
python3 simulation_batch/preflight_param_check.py
```

### `simulation_batch/db_diff.py` — DB session audit
Three modes:
- `--snap LABEL`  → save DB state before a PDF/absorption session
- `--diff LABEL`  → verify DB actually changed after session (catches false "done" claims)
- `--audit RUN`   → cross-check run manifest vs pipeline JSON vs DB for key params

```bash
# Before absorption session:
python3 simulation_batch/db_diff.py --snap PRE_<article_name>
# After Claude says "done":
python3 simulation_batch/db_diff.py --diff PRE_<article_name>
# Before simulation launch:
python3 simulation_batch/db_diff.py --audit simulation_batch/runs/<run_dir>
```

### `execution_manifest.json` — Per-run parameter record
Automatically written by `glad_v3_core.py` at simulation start (after all cfg overrides).
Located in each run's `checkpoints/` directory.
Contains: alpha, pitch, author_radius_nm, batch_size, enable_sticking, enable_surface_diffusion, cfg_hash.
