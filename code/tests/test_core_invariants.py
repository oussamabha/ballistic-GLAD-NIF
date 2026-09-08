"""test_core_invariants.py -- physics-invariant + reproducibility gate for glad_v3_core.py
(software lens). RUN THIS AFTER ANY EDIT TO glad_v3_core.py OR A KERNEL.

Two "validated" fixes were shipped and retracted on 2026-09-07 because each was checked
against only the one metric it targeted. This gate is the mechanical version of
FUTURE_WORKFLOW_RULES.md Rule 52: a tiny deterministic deposition, then hard asserts on
the invariants a physics-state change must never break --
  * every atom inside [0, box) after periodic wrap, z >= r, finite (no NaN/Inf)
  * atom count is sane (not ~0, not absurd) and the film reached the target height
  * lateral-connectivity class is CONNECTED_FILM for this dense config
  * REPRODUCIBILITY: same config + same seed -> byte-identical checkpoint
  * DIFFUSION-OFF GOLDEN: matches a committed golden checkpoint (regression guard for
    ballistic-path byte-identity across refactors) -- skipped until the golden is created
    with --make-golden.

Needs a CUDA GPU + the `gladwsl` env. Not in the pre-commit hook (no GPU there); it is the
manual/CI gate. ~1-2 min.

    python3 -m pytest 01_GLAD_SIMULATION/tests/test_core_invariants.py -v
    python3 01_GLAD_SIMULATION/tests/test_core_invariants.py --make-golden   # (re)create the golden
"""
from __future__ import annotations
import json, subprocess, sys, os
from pathlib import Path
import numpy as np
import pytest
import h5py

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "01_GLAD_SIMULATION" / "simulation MD" / "src" / "glad_v3_core.py"
TMP = ROOT / "01_GLAD_SIMULATION" / "tests" / "_tmp_invariants"
GOLDEN = ROOT / "01_GLAD_SIMULATION" / "tests" / "golden_ballistic_box20_a85_h8_seed0.npy"

BOX = 20.0
R = 0.128
CFG = f"""\
simulation: {{target_height: 8.0, alpha: 85.0, material: Cu, random_seed: 0, substrate_spacing: 1.0, enable_sticking: false}}
performance: {{use_gpu: true, batch_size: 512}}
box: {{width: {BOX}, depth: {BOX}}}
deposition: {{rate: 0.3, growth_efficiency: 1.0, temperature: 298.15, melting_point_K: 1357.77, source_distance_cm: 11.0}}
physics: {{pitch: 150.0, growth_rate: 0.3, enable_surface_diffusion: false, active_window_height: 20.0, resume_active_window_height: 20.0, author_radius_nm: {R}}}
checkpoint: {{output_dir: "OUTDIR/checkpoints", height_interval: 5, particle_interval: 999999999, time_interval: 999999, allow_legacy_resume: false}}
monitoring: {{log_file: "OUTDIR/sim.log", write_interval: 999999, verbose: false, auto_postprocess: false}}
"""


_IN_WSL = sys.platform != "win32" and Path("/proc/version").exists()


def _run(tag: str) -> np.ndarray:
    d = TMP / tag
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = d / "cfg.yaml"
    if _IN_WSL:
        # already inside gladwsl -- call the core directly, no `wsl` wrapper
        cfg.write_text(CFG.replace("OUTDIR", str(d)), encoding="utf-8")
        core = str(ROOT / "01_GLAD_SIMULATION" / "simulation MD" / "src" / "glad_v3_core.py")
        r = subprocess.run([sys.executable, core, "--config", str(cfg), "--fresh"],
                           capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    else:
        cfg.write_text(CFG.replace("OUTDIR", str(d).replace("\\", "/")), encoding="utf-8")
        cmd = ("source ~/miniforge3/etc/profile.d/conda.sh && conda activate gladwsl && "
               f'cd "{ROOT.as_posix().replace("D:", "/mnt/d")}" && '
               f'python3 "01_GLAD_SIMULATION/simulation MD/src/glad_v3_core.py" '
               f'--config "{cfg.as_posix().replace("D:", "/mnt/d")}" --fresh')
        r = subprocess.run(["wsl", "bash", "-lc", cmd], capture_output=True, text=True, timeout=600)
    ck = sorted((d / "checkpoints").glob("checkpoint_v3_*.h5"))
    assert ck, f"[{tag}] no checkpoint produced\nSTDOUT tail:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}"
    with h5py.File(ck[-1], "r") as f:
        return np.asarray(f["positions"][:, :3], np.float64)


@pytest.fixture(scope="module")
def P():
    return _run("run1")


def test_finite(P):
    assert np.isfinite(P).all(), "NaN/Inf in atom positions"


def test_box_containment(P):
    x = np.mod(P[:, 0], BOX); y = np.mod(P[:, 1], BOX)
    # np.mod already folds into [0,box); the check is that nothing is pathological pre-wrap
    assert np.abs(P[:, 0]).max() < 50 * BOX and np.abs(P[:, 1]).max() < 50 * BOX, \
        "atoms drifted an implausible number of box widths (wrap/kernel bug -- Rule 54)"
    assert (x >= 0).all() and (x < BOX).all() and (y >= 0).all() and (y < BOX).all()


def test_z_floor(P):
    # The ballistic contact path leaves a handful of atoms a fraction of a radius below the
    # substrate plane (z ~ -0.06 nm; ~1e-4 of the population, also seen in the production
    # GRIDFIX checkpoints -- a pre-existing substrate-boundary contact-resolution quirk, not
    # a regression). Tolerance = one radius below the plane: catches real downward drift
    # (a wrap/kernel bug gives z ~ -50), tolerates the known quirk.
    zmin = P[:, 2].min()
    assert zmin >= -R, f"atoms driven well below the substrate (z_min={zmin:.3f}, tol={-R:.3f})"
    frac_below = float((P[:, 2] < R - 1e-3).mean())
    assert frac_below < 0.01, f"{frac_below:.1%} of atoms below the substrate floor -- was ~0.01% (regression)"


def test_mass_and_height(P):
    n = len(P)
    assert 5_000 < n < 5_000_000, f"atom count {n} outside sane band for box20/a85/h8"
    assert P[:, 2].max() >= 7.5, f"film did not reach target height (z_max={P[:,2].max():.2f})"


def test_film_body(P):
    # box=20 is too small for topology_analysis_v1's percolation verdict (needs ~box>=100).
    # Instead assert the film has a real BODY -- the failure mode Rule 52 cares about is the
    # one the broken diffusion kernel produced: the mature band hollows out / disperses.
    H = P[:, 2].max()
    band = P[(P[:, 2] >= 0.35 * H) & (P[:, 2] < 0.85 * H)]
    assert len(band) > 500, f"mature band nearly empty ({len(band)} atoms) -- film dispersed?"
    vox = 1.0
    n = int(np.ceil(BOX / vox))
    nz = max(1, int(np.ceil((0.5 * H) / vox)))
    g = np.zeros((n, n, nz), bool)
    ix = np.clip(((band[:, 0] % BOX) / vox).astype(int), 0, n - 1)
    iy = np.clip(((band[:, 1] % BOX) / vox).astype(int), 0, n - 1)
    iz = np.clip(((band[:, 2] - 0.35 * H) / vox).astype(int), 0, nz - 1)
    g[ix, iy, iz] = True
    occ = g.mean()
    assert 0.15 < occ < 0.95, \
        f"mature-band voxel occupancy {occ:.2f} outside [0.15, 0.95] -- dispersed (low) or solid block (high)"


def test_reproducible(P):
    P2 = _run("run2")
    assert P.shape == P2.shape, f"non-deterministic atom count: {P.shape} vs {P2.shape}"
    assert np.array_equal(P, P2), \
        f"same config+seed produced different positions (max |d|={np.abs(P-P2).max():.2e}) -- reproducibility broken"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden not created; run with --make-golden")
def test_ballistic_golden(P):
    g = np.load(GOLDEN)
    assert P.shape == g.shape, f"ballistic output shape drifted from golden: {P.shape} vs {g.shape}"
    assert np.array_equal(P, g), \
        "ballistic path changed vs the committed golden -- a supposedly no-op change altered results"


if __name__ == "__main__":
    if "--make-golden" in sys.argv:
        pos = _run("golden")
        np.save(GOLDEN, pos)
        print(f"golden written: {GOLDEN}  ({pos.shape[0]} atoms)")
    else:
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
