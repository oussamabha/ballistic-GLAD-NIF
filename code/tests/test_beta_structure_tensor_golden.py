"""test_beta_structure_tensor_golden.py -- regression guard for the structure-tensor
column-tilt estimator (05_THESIS_PAPER_ASSETS/beta_structure_tensor.py).

WHY: P1's beta(alpha) headline result flipped this session -- the previous watershed
estimator under-read beta by 15-20 deg on dense films, the structure-tensor estimator
matches Cu/CuOx SEM to ~3 deg, and three limitations items turned out to be watershed
artefacts. That result now rests on ONE estimator with no test. This builds synthetic
density fields of straight columns tilted at KNOWN angles and asserts beta_one() recovers
them within tolerance, so the flipped result cannot silently drift back.

Needs h5py + scipy (the gladwsl env). ~5 s.
    python3 -m pytest 01_GLAD_SIMULATION/tests/test_beta_structure_tensor_golden.py -v
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EST = ROOT / "05_THESIS_PAPER_ASSETS" / "beta_structure_tensor.py"
h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")

_spec = importlib.util.spec_from_file_location("bst", EST)
bst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bst)

BOX = 100.0
H = 240.0           # film height (nm)
R = 0.128


def _synth_columns(beta_deg: float, n_col: int = 25, atoms_per_col: int = 40000,
                   seed: int = 0) -> np.ndarray:
    """Straight cylindrical columns, all tilted by beta_deg from the substrate normal
    toward +x, on a jittered lattice of footprints. Returns an (N,3) position array
    with UNWRAPPED x,y (the estimator wraps internally)."""
    rng = np.random.default_rng(seed)
    b = np.radians(beta_deg)
    tanb = np.tan(b)
    side = int(round(np.sqrt(n_col)))
    step = BOX / side
    cols = []
    for ix in range(side):
        for iy in range(side):
            x0 = (ix + 0.5) * step + rng.normal(0, 0.15 * step)
            y0 = (iy + 0.5) * step + rng.normal(0, 0.15 * step)
            z = rng.uniform(0.0, H, atoms_per_col)
            # column axis: dx/dz = tan(beta); radial scatter ~ 3 nm
            rad = rng.normal(0, 3.0, (atoms_per_col, 2))
            x = x0 + z * tanb + rad[:, 0]
            y = y0 + rad[:, 1]
            cols.append(np.column_stack([x, y, z]))
    P = np.vstack(cols)
    return P


def _write_h5(P: np.ndarray, path: Path):
    with h5py.File(path, "w") as f:
        f.create_dataset("positions", data=P.astype(np.float64))
        f.attrs["box_width"] = BOX
        f.attrs["current_height"] = float(P[:, 2].max())


@pytest.mark.parametrize("beta_true", [25.0, 45.0, 60.0, 70.0])
def test_recovers_known_tilt(tmp_path, beta_true):
    P = _synth_columns(beta_true)
    p = tmp_path / f"synth_b{int(beta_true)}.h5"
    _write_h5(P, p)
    r = bst.beta_one(str(p), BOX, bst.VOX)
    assert r is not None and r.get("beta") is not None, f"estimator returned no beta: {r}"
    err = abs(r["beta"] - beta_true)
    assert err <= 8.0, (
        f"structure-tensor beta={r['beta']:.1f} deg vs true {beta_true} deg "
        f"(err {err:.1f} deg > 8 deg tol); 16-84 band {r['lo']:.1f}-{r['hi']:.1f}, n={r['n']}"
    )


def test_monotonic_in_true_tilt(tmp_path):
    got = []
    for bt in (20.0, 40.0, 60.0):
        _write_h5(_synth_columns(bt), tmp_path / f"m{int(bt)}.h5")
        r = bst.beta_one(str(tmp_path / f"m{int(bt)}.h5"), BOX, bst.VOX)
        got.append(r["beta"])
    assert got[0] < got[1] < got[2], f"estimator not monotonic in true tilt: {got}"


if __name__ == "__main__":
    import sys, subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
