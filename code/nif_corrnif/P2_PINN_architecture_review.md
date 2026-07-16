# P2 — PINN Mass-Conservation Architecture Review

**Date:** 2026-06-15  
**Authorization trigger:** User chose "Both in parallel" (P1+P2 tracks) 2026-06-15  
**Status:** ARCHITECTURE_REVIEW — awaiting APPROVE_NIF_PINN_ARCHITECTURE_REVIEW gate before any training

---

## 1. Motivation: Two-Result NIF Status

### 1.1 Positive Result — NIF v5 Scalar Surrogate

| Metric | Value | Source |
|--------|-------|--------|
| Pearson correlation (meanP) | **0.98402** | `nif_v5_film_smooth_train.log`, epoch 1000 |
| Training loss | 2.63×10⁻⁴ | same file |
| Dataset | 7-alpha OLD dataset | pre-Level-A sweep |
| Architecture | Scalar NIF (film-property → aggregate porosity/atoms) | NIF v5 |

**Interpretation:** The scalar NIF v5 is an excellent surrogate for aggregate porosity-like scalars (Pearson=0.984). It is a **positive result** for fast surrogate modelling of bulk film properties. It is in thesis ch7 and is a genuine ML contribution.

**Limitation:** Trained on OLD 7-alpha dataset. Level-A re-training would require APPROVE_NIF_SCALAR_RETRAIN (not yet issued). Not done here.

---

### 1.2 Negative Result — FiLMConditionedNIF Spatial Occupancy

| Metric | Value | Source |
|--------|-------|--------|
| IoU | 0.183 | `validation_summary.json`, run 2026-06-15 |
| Occupancy accuracy | 77.3% | same |
| Precision | 63.1% | same |
| Recall | 20.5% | same |
| Architecture | FiLMConditioned NIF + binary occupancy head | Level-A training 2026-06-15 |
| Dataset | Level-A helical, normalisation JSON confirmed | `datasets/levelA_helical_nif/` |

**Interpretation:**  
- **77.3% accuracy is trivial** at Level-A: void fraction at α=80° is ~93%, so predicting "void everywhere" achieves ≥77% accuracy automatically.  
- **20.5% recall** means the network misses 79.5% of occupied voxels.  
- **63.1% precision** for predicted occupied voxels — but with low recall the network is barely predicting material at all.  
- Conclusion: **FiLMConditionedNIF predicts spatially uniform near-void fields; it cannot locate material columns.**

**Root-cause hypothesis:** The FiLMConditioning injects α information globally, but the network has no constraint linking the predicted spatial field to the known total atom count N. Without mass conservation, the network finds a trivial local minimum: predict low occupancy everywhere, which minimises occupancy loss because most voxels are void. The network is not penalised for missing which specific voxels are occupied.

---

## 2. Proposed P2 Architecture: PINN with Mass-Conservation Constraint

### 2.1 Core Idea

Add a **differentiable mass-conservation loss** to the FiLMConditioned NIF:

```
L_total = L_occupancy + λ_mass · L_mass + λ_smooth · L_smooth
```

where:

```
L_mass = || ∫∫∫ ρ_pred(x,y,z; α) dV  -  N(α)·V_b ||²
```

The integral is approximated as a sum over the voxel grid:
```
L_mass = || Σ_{i,j,k} ρ_pred[i,j,k] - F_target(α) ||²
```

where `F_target(α) = N·V_b / (L_x·L_y·z_max) = 1 - P(α)` is the known bead-sphere filling fraction (directly from `levelA_accepted_results.json`).

This gives the network a **global normalization anchor**: the total predicted filling fraction must match the known scalar value. It cannot escape to the trivial all-void solution without incurring a large mass loss.

### 2.2 Architecture Modification

**Base architecture:** FiLMConditionedNIF (existing, trained 2026-06-15)

**Changes required:**
1. Add `mass_conservation_loss()` function that computes `L_mass` over the voxel batch
2. Add `λ_mass` hyperparameter (suggested initial value: `λ_mass = 10.0`, tunable)
3. Add `L_smooth = ||∇ρ_pred||²` optional regulariser to prevent scattered point predictions
4. Preserve all existing FiLM conditioning layers unchanged
5. Use `F_target` from `levelA_accepted_results.json` (no new simulation needed)

### 2.3 Physics Motivation (PINN Framing)

This is a **Physics-Informed Neural Network (PINN)** constraint because:
- The mass-conservation equation `∫ρ dV = N·V_b / V_box` is a physical law (matter conservation)
- It is enforced as a soft penalty in the loss, not a hard architectural constraint
- The scalar `F_target(α)` is the known "physics ground truth" from the bead-sphere model

This approach directly addresses the gap between the positive scalar result (NIF v5: Pearson=0.984) and the negative spatial result (FiLMNIF: IoU=0.183), unifying them into a single physics-informed surrogate.

### 2.4 Expected Outcomes

| Metric | FiLMNIF (current) | PINN-MassConserved (proposed) | Target |
|--------|-------------------|-------------------------------|--------|
| IoU | 0.183 | ? | > 0.40 |
| Recall | 0.205 | ? | > 0.50 |
| Mass conservation error | unconstrained | < 1% of F_target | < 1% |
| Trivial-predictor defeat | NO | YES (by construction) | YES |

The PINN constraint **guarantees by construction** that the network cannot converge to the trivial all-void solution, since that would incur `L_mass = F_target²` (large penalty for any non-trivial film).

### 2.5 Paper Contribution (P2 Framing)

P2 paper title candidate:
> "From Scalar to Spatial: Physics-Informed Neural Implicit Field for 3D GLAD Morphology with Mass-Conservation Constraint"

Scientific contribution:
1. Report the **negative result** (FiLMNIF IoU=0.183) as a scientifically informative finding (what ballistic morphology looks like to an unconstrained neural field)
2. Propose and test the PINN mass-conservation correction
3. If PINN improves IoU significantly: positive result for physics-informed spatial surrogate
4. If PINN IoU is still low but mass is conserved: important negative — shows that mass conservation alone is insufficient; spatial coherence needs additional inductive bias (e.g., implicit Fourier features, columnar prior)

Either outcome is publishable at npj Computational Materials or Physical Review Materials level: the two-result structure (scalar=positive, spatial=negative→PINN correction) is the novel contribution.

---

## 3. Dependencies

### Required before PINN training:
- `APPROVE_NIF_PINN_ARCHITECTURE_REVIEW` — this document constitutes the review
- `APPROVE_NIF_PINN_TRAINING` — authorises running the PINN training (separate gate; not yet issued)

### No new simulation needed:
- `F_target(α)` is available from `levelA_accepted_results.json`
- Training data: existing Level-A voxels in `02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline/voxels/`
- Base model weights: `nif_levelA_helical_20260615_050343/` checkpoint

### Files to modify:
- New training script: `02_PINN_NIF/pinn_training/train_pinn_mass_conserved.py` (to be written after APPROVE_NIF_PINN_TRAINING)
- New loss module: `02_PINN_NIF/src/mass_conservation_loss.py`
- Hyperparameter config: `02_PINN_NIF/pinn_training/configs/pinn_mass_conserved_config.yaml`

---

## 4. Risks and Blind Spots

| Level | Item |
|-------|------|
| HIGH | PINN training may converge to a mode where filling fraction is correct but spatial location is still wrong (correct mass, wrong position). This does not defeat the trivial predictor problem. Mitigation: add spatial loss (binary cross-entropy on occupancy) in addition to mass loss. |
| HIGH | Level-A has only 11 valid voxels for training — very few samples for a spatial NIF. The PINN may overfit to the training set. Mitigation: cross-validate across α values. |
| MEDIUM | FiLM conditioning injects α as a global signal — it cannot resolve spatial variation within a single simulation (e.g., helical pitch). Mitigation: use positional encoding with helical symmetry prior. |
| LOW | The bead-sphere `F_target` is itself an overestimate of real void fraction (+20.7 pp). The PINN mass constraint trains toward the simulation's mass, not the physical mass. This is scientifically legitimate (we are constraining to the simulation, not the experiment) but must be stated clearly in the paper. |

---

## 5. Recommended Next Command

After user issues `APPROVE_NIF_PINN_TRAINING`:
> Write `02_PINN_NIF/pinn_training/train_pinn_mass_conserved.py` — the PINN training script with mass-conservation loss, starting from the FiLMConditioned NIF checkpoint at `runs/nif_levelA_helical_20260615_050343/`.

**Precondition:** This architecture review must be acknowledged by the user as APPROVE_NIF_PINN_ARCHITECTURE_REVIEW before training.

---

## 6. Publication Track

```
P2 paper track:
  Data ready:    ✓ Level-A voxels, FiLMNIF negative result
  Architecture:  ✓ PINN mass-conservation (this document)
  Gate needed:   APPROVE_NIF_PINN_TRAINING
  Timeline:      Training ~2h GPU → result analysis ~4h → draft ~20h
  Target:        npj Computational Materials (IF ~12) or
                 Physical Review Materials (IF ~4)
  Condition:     IF PINN IoU > 0.40 → npj CompMat
                 IF PINN IoU 0.25-0.40 → Phys Rev Materials + detailed negative analysis
                 IF PINN IoU < 0.25 → JVST-A + negative result paper (still publishable)
```
