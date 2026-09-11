#!/usr/bin/env python3
"""P5 beta-offset vs melting-point regression -- v2, full material set (2026-09-10).

The 2026-09-09 result (P5_BETA_TM_REGRESSION_RESULT_20260909.md) used only the
7 materials whose checkpoints existed at analysis time (Au, Cr, Fe, Mo, Pt, Ti, W).
More materials (Ag, Al, Co, Mg, Ni, Zr; Si kept separate as a non-metal) finished
afterward and were never folded in. This re-runs the identical pre-registered
test at the larger n.

Method (identical to v1 / P1's validated beta pipeline):
  beta_sim     : analyze_beta_tilt_laws.beta_from_h5  (PBC-aware windowed x/y-vs-z regression)
  beta_tangent : atan(tan(alpha)/2)
  offset       : beta_sim - beta_tangent          (strongly alpha-dependent -> test per alpha)
  predictor    : T_s / T_m   (T_s = 298.15 K, T_m from each checkpoint's own attrs)
  test         : Pearson r(offset, T_s/T_m) + EXACT permutation p (all n! perms for
                 n! <= 400k, else 200k Monte-Carlo). Bar: |r| >= 0.5 AND p < 0.05.

Reads the final checkpoint per run (max current_height among emergency/B/A).
Read-only. Writes CSV + JSON next to this script; prints a report.
"""
from __future__ import annotations
import csv, glob, itertools, json, math, os, sys

import numpy as np
import h5py

ROOT = "/mnt/d/GLAD_PROJECT"
CAMP = f"{ROOT}/01_GLAD_SIMULATION/simulation_batch/runs/P5_MATERIAL_GENERALIZATION_20260909"
sys.path.insert(0, f"{ROOT}/03_VOXEL_DESCRIPTOR_ANALYSIS/tools")
from analyze_beta_tilt_laws import beta_from_h5, tangent_beta  # noqa: E402

BOX_W = BOX_D = 100.0
T_S = 298.15
HEIGHT_MIN_NM = 114.0
OUTDIR = os.path.dirname(os.path.abspath(__file__))
NONMETAL = {"Si"}


def final_checkpoint(run_dir):
    best = None
    for name in ("checkpoint_v3_emergency.h5", "checkpoint_v3_B.h5", "checkpoint_v3_A.h5"):
        p = os.path.join(run_dir, "checkpoints", name)
        if not os.path.isfile(p):
            continue
        try:
            with h5py.File(p, "r") as h:
                ch = float(h.attrs.get("current_height", -1))
                at = {k: h.attrs[k] for k in h.attrs}
        except Exception:
            continue
        if best is None or ch > best[1]:
            best = (p, ch, at)
    return best


def collect():
    rows = []
    for run_dir in sorted(glob.glob(f"{CAMP}/P5_*_alpha*_seed*")):
        base = os.path.basename(run_dir)
        parts = base.split("_")
        mat = parts[1]
        alpha = float(parts[2].replace("alpha", ""))
        fc = final_checkpoint(run_dir)
        if fc is None:
            rows.append(dict(material=mat, alpha=alpha, status="NO_CHECKPOINT"))
            continue
        path, ch, at = fc
        tm = float(at.get("melting_point_K", "nan"))
        status = "OK" if ch >= HEIGHT_MIN_NM else f"SHORT_{ch:.0f}nm"
        b = beta_from_h5(path, z_max_nm=max(ch, 60.0), box_w=BOX_W, box_d=BOX_D)
        bmed = b.get("beta_median_deg")
        rows.append(dict(
            material=mat, alpha=alpha, status=status,
            height_nm=round(ch, 1), T_m_K=tm,
            Ts_over_Tm=(T_S / tm if tm == tm else float("nan")),
            radius_nm=float(at.get("radius", "nan")),
            beta_sim_deg=bmed, beta_mean_deg=b.get("beta_mean_deg"),
            n_windows=b.get("n_windows"),
            beta_tangent_deg=round(tangent_beta(alpha), 3),
            offset_tangent_deg=(None if bmed is None else round(bmed - tangent_beta(alpha), 3)),
            checkpoint=os.path.relpath(path, ROOT),
        ))
    return rows


def perm_p(x, y, r_obs, n_mc=200_000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = math.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d else 0.0
    count = 0; total = 0
    if math.factorial(n) <= 400_000:
        for pz in itertools.permutations(range(n)):
            total += 1
            if abs(pearson(x, y[list(pz)])) >= abs(r_obs) - 1e-12:
                count += 1
        return count / total, total, "exact"
    rng = np.random.default_rng(seed)
    yy = y.copy()
    for _ in range(n_mc):
        rng.shuffle(yy)
        total += 1
        if abs(pearson(x, yy)) >= abs(r_obs) - 1e-12:
            count += 1
    return count / total, total, f"mc_{n_mc}"


def regress(rows, alpha, exclude_nonmetal=True):
    sub = [r for r in rows
           if r.get("status") == "OK" and r["alpha"] == alpha
           and r.get("offset_tangent_deg") is not None
           and (not exclude_nonmetal or r["material"] not in NONMETAL)]
    sub.sort(key=lambda r: r["material"])
    if len(sub) < 4:
        return dict(alpha=alpha, n=len(sub), note="too few points")
    x = np.array([r["Ts_over_Tm"] for r in sub])
    y = np.array([r["offset_tangent_deg"] for r in sub])
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt((xm * xm).sum() * (ym * ym).sum())
    r = float((xm * ym).sum() / denom) if denom else 0.0
    p, ntot, kind = perm_p(x, y, r)
    passes = (abs(r) >= 0.5) and (p < 0.05)
    return dict(alpha=alpha, n=len(sub),
                materials=[q["material"] for q in sub],
                pearson_r=round(r, 4), r2=round(r * r, 4),
                perm_p=round(p, 5), perm_kind=kind, perm_total=ntot,
                spread_offset_deg=round(float(y.max() - y.min()), 3),
                verdict=("PASS pre-registered bar (|r|>=0.5 AND p<0.05)" if passes
                         else "does NOT pass (bar = |r|>=0.5 AND p<0.05)"),
                table=[dict(material=q["material"], Ts_over_Tm=round(q["Ts_over_Tm"], 4),
                            T_m_K=q["T_m_K"], offset_deg=q["offset_tangent_deg"],
                            beta_sim_deg=q["beta_sim_deg"]) for q in sub])


def main():
    rows = collect()
    with open(f"{OUTDIR}/p5_beta_tm_v2_perrun.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    results = {f"alpha_{int(a)}": regress(rows, a) for a in (60.0, 75.0, 85.0)}
    results["alpha_60_incl_Si"] = regress(rows, 60.0, exclude_nonmetal=False)
    with open(f"{OUTDIR}/p5_beta_tm_v2_regression.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("=" * 78)
    print("P5 beta-offset vs T_s/T_m -- v2 (full material set), 2026-09-10")
    print("=" * 78)
    ok = [r for r in rows if r.get("status") == "OK"]
    short = [r for r in rows if str(r.get("status", "")).startswith("SHORT")]
    missing = [r for r in rows if r.get("status") == "NO_CHECKPOINT"]
    print(f"runs: {len(rows)} total | {len(ok)} complete (>= {HEIGHT_MIN_NM:.0f} nm) "
          f"| {len(short)} short | {len(missing)} no-checkpoint")
    if short:
        print("  short:", ", ".join(f"{r['material']}a{int(r['alpha'])}({r['status']})" for r in short))
    if missing:
        print("  missing:", ", ".join(f"{r['material']}a{int(r['alpha'])}" for r in missing))
    for key in ("alpha_60", "alpha_75", "alpha_85", "alpha_60_incl_Si"):
        R = results[key]
        print(f"\n--- {key}  (n={R.get('n')}) ---")
        if R.get("note"):
            print("   ", R["note"]); continue
        print(f"    materials: {', '.join(R['materials'])}")
        print(f"    Pearson r = {R['pearson_r']}   R^2 = {R['r2']}   perm p = {R['perm_p']} ({R['perm_kind']})")
        print(f"    offset spread across materials: {R['spread_offset_deg']} deg")
        print(f"    -> {R['verdict']}")
    print(f"\nwrote {OUTDIR}/p5_beta_tm_v2_perrun.csv")
    print(f"wrote {OUTDIR}/p5_beta_tm_v2_regression.json")


if __name__ == "__main__":
    sys.exit(main())
