"""
LENS1 WP3 -- formalise "the exact column position carries no signal that
generalises across realisations of the same growth conditions" as a
mutual-information statement.

Data: the 3-seed replicates at alpha = 85 / 87 / 89 deg
(P1_CALIBRATED_r128_CAMPAIGN_V2_POSTFIX_20260906), box 100 nm, 300 nm target.
9 checkpoints total.

Estimator: Ross (2014) k-NN mutual information for a continuous vector X and a
discrete label Y (KSG-style, mixed). Implemented here directly on scipy.cKDTree
(no sklearn in the env). I is reported in nats.

Three MI quantities, each with a checkpoint-level shuffled-label null
(50 permutations: the 9 checkpoints get their alpha labels reassigned at
random, 3 per class, so within-checkpoint spatial structure and class balance
are preserved but the true position<->alpha link is broken):

  I1 = I( (x,y,z)            ; alpha )   -- full atom position
  I2 = I( z                  ; alpha )   -- height coordinate only (a statistic)
  I3 = I( (x,y) | z-band     ; alpha )   -- lateral position within a mature-band
                                            z-slice = the "column position" signal

Prediction (rigorous form of P1's central claim):
  I3 ~ null            (lateral column position carries no generalisable alpha info)
  I2 ~ I1 >> null      (the recoverable alpha signal lives in the height / density
                        statistic, not in where any individual column sits)

Coordinates are wrapped into [0, box) and standardised before the k-NN step.
Atoms are subsampled to N_SUB per checkpoint for tractability.

Output: 05_THESIS_PAPER_ASSETS/LENS1_WP3_RESULT_20260908.md  + .json
Run in gladwsl (needs h5py): python3 lens1_wp3_position_alpha_mi.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

ROOT = Path("/mnt/d/GLAD_PROJECT")
V2 = ROOT / "01_GLAD_SIMULATION/simulation_batch/runs/P1_CALIBRATED_r128_CAMPAIGN_V2_POSTFIX_20260906"
OUT = ROOT / "05_THESIS_PAPER_ASSETS"
BOX = 100.0
ALPHAS = [85, 87, 89]
SEEDS = [0, 1, 2]
N_SUB = 20000        # atoms per checkpoint used for the MI estimate
K = 5               # k-NN order
N_PERM = 50
RNG = np.random.default_rng(0)


def load_xyz(a, s):
    import h5py
    d = V2 / f"P1r128v2_alpha{a:03d}_seed{s:03d}_box100_300nm/checkpoints"
    for n in ("checkpoint_v3_emergency.h5", "checkpoint_v3_A.h5", "checkpoint_v3_B.h5"):
        p = d / n
        if p.exists():
            with h5py.File(p, "r") as f:
                P = np.asarray(f["positions"][:, :3], np.float64)
            P[:, 0] = np.mod(P[:, 0], BOX)
            P[:, 1] = np.mod(P[:, 1], BOX)
            return P
    raise FileNotFoundError(str(d))


def ross_mi_cont_disc(X: np.ndarray, y: np.ndarray, k: int = K) -> float:
    """Ross 2014 mixed KSG MI (nats). X: (N,d) continuous (standardised), y: (N,) int labels."""
    N = len(X)
    tree_all = cKDTree(X)
    I = 0.0
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if len(idx) <= k:
            continue
        sub = X[idx]
        t = cKDTree(sub)
        dk, _ = t.query(sub, k=k + 1)          # self + k neighbours
        d = dk[:, k]
        m = tree_all.query_ball_point(sub, d - 1e-12, return_length=True) - 1  # exclude self
        m = np.clip(m, 1, None)
        I += (digamma(N) + digamma(k) - np.mean(digamma(m + 1)) - digamma(len(idx))) * len(idx)
    return float(I / N)


def build_pools():
    pools = {}
    for a in ALPHAS:
        for s in SEEDS:
            P = load_xyz(a, s)
            if len(P) > N_SUB:
                P = P[RNG.choice(len(P), N_SUB, replace=False)]
            pools[(a, s)] = P
            print(f"  a{a} s{s}: {len(P)} atoms (z {P[:,2].min():.1f}-{P[:,2].max():.1f})")
    return pools


def stack(pools, feats, labels_map):
    Xs, ys = [], []
    for key, P in pools.items():
        lab = labels_map[key]
        if feats == "xyz":
            F = P.copy()
        elif feats == "z":
            F = P[:, 2:3].copy()
        elif feats == "xy_band":
            H = P[:, 2].max()
            m = (P[:, 2] >= 0.45 * H) & (P[:, 2] < 0.65 * H)
            F = P[m][:, :2].copy()
        Xs.append(F); ys.append(np.full(len(F), lab, int))
    X = np.vstack(Xs); y = np.concatenate(ys)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, y


def perm_labels():
    """9 checkpoints -> 3 of each alpha, randomly assigned (checkpoint-level shuffle)."""
    keys = [(a, s) for a in ALPHAS for s in SEEDS]
    labs = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    RNG.shuffle(labs)
    return {k: int(labs[i]) for i, k in enumerate(keys)}


def main():
    print("loading 9 checkpoints ...")
    pools = build_pools()
    true_map = {(a, s): ALPHAS.index(a) for a in ALPHAS for s in SEEDS}

    res = {}
    for feats in ("xyz", "z", "xy_band"):
        Xt, yt = stack(pools, feats, true_map)
        I_true = ross_mi_cont_disc(Xt, yt)
        I_null = []
        for _ in range(N_PERM):
            Xn, yn = stack(pools, feats, perm_labels())
            I_null.append(ross_mi_cont_disc(Xn, yn))
        I_null = np.array(I_null)
        z = (I_true - I_null.mean()) / (I_null.std() + 1e-12)
        p = float((np.sum(I_null >= I_true) + 1) / (N_PERM + 1))
        res[feats] = dict(I_true_nats=I_true, null_mean=float(I_null.mean()),
                          null_std=float(I_null.std()), null_p95=float(np.percentile(I_null, 95)),
                          z_score=float(z), p_value=p)
        print(f"  {feats:8s}: I_true={I_true:.4f}  null={I_null.mean():.4f}+/-{I_null.std():.4f}  "
              f"z={z:+.1f}  p={p:.3f}")

    verdict = (
        "I3 (lateral column position within a mature-band slice) is "
        + ("indistinguishable from the checkpoint-shuffled null"
           if res["xy_band"]["p_value"] > 0.05 else
           "ABOVE the null (p<=0.05) -- lateral position does carry generalisable alpha signal")
        + f"; I2 (height only) recovers {100*res['z']['I_true_nats']/max(res['xyz']['I_true_nats'],1e-9):.0f}% "
          "of the full-position MI."
    )
    out = dict(n_sub_per_ckpt=N_SUB, k=K, n_perm=N_PERM, alphas=ALPHAS, seeds=SEEDS,
               results=res, verdict=verdict)
    (OUT / "lens1_wp3_position_alpha_mi.json").write_text(json.dumps(out, indent=2))

    md = f"""# LENS1 WP3 result -- position<->alpha mutual information (2026-09-08)

Formalises P1's central predictability-limit claim ("the exact position of an individual
column carries no signal that generalises across realisations of the same growth
conditions") as a mutual-information statement.

Data: 3-seed replicates at alpha = 85/87/89 deg
(`P1_CALIBRATED_r128_CAMPAIGN_V2_POSTFIX_20260906`), box 100 nm, {N_SUB} atoms/checkpoint.
Estimator: Ross (2014) mixed k-NN MI (k={K}), nats. Null: {N_PERM} checkpoint-level
label shuffles (each alpha still appears 3x, within-checkpoint structure preserved).

| feature | I_true (nats) | null mean +/- sd | z | p |
|---|---|---|---|---|
| full position (x,y,z) | {res['xyz']['I_true_nats']:.4f} | {res['xyz']['null_mean']:.4f} +/- {res['xyz']['null_std']:.4f} | {res['xyz']['z_score']:+.1f} | {res['xyz']['p_value']:.3f} |
| height z only | {res['z']['I_true_nats']:.4f} | {res['z']['null_mean']:.4f} +/- {res['z']['null_std']:.4f} | {res['z']['z_score']:+.1f} | {res['z']['p_value']:.3f} |
| lateral (x,y) in 0.45-0.65 H band | {res['xy_band']['I_true_nats']:.4f} | {res['xy_band']['null_mean']:.4f} +/- {res['xy_band']['null_std']:.4f} | {res['xy_band']['z_score']:+.1f} | {res['xy_band']['p_value']:.3f} |

**Verdict:** {verdict}

## Reading
- If lateral column position (row 3) sits in the null band while height (row 2) recovers
  most of the full-position MI, the recoverable alpha signal is entirely a
  height/density *statistic* -- not the location of any individual column. That is the
  rigorous form of the paper's "predictability limit": a model given only spatial
  coordinates cannot recover alpha from where the columns are, only from how the density
  is distributed in height.
- Caveats: only 3 alpha values, 3 seeds each (9 checkpoints -> the null has C(9;3,3,3)
  distinct assignments, {N_PERM} sampled); MI in nats is estimator- and
  subsample-dependent in absolute value, so read the comparison to the null, not the
  magnitude. The mature-band slice (0.45-0.65 H) is one choice; a fuller version scans
  the band.
"""
    (OUT / "LENS1_WP3_RESULT_20260908.md").write_text(md)
    print("wrote", OUT / "LENS1_WP3_RESULT_20260908.md")


if __name__ == "__main__":
    main()
