"""beta_fft_streak.py -- [SHELVED 2026-09-08: the box-periodic film envelope dominates
the 2-D spectrum and swamps the column-tilt ridge -> returns a degenerate ~0.5 deg at
every angle. Kept for provenance; needs an envelope-removal / windowing fix before use.
P1 uses beta_structure_tensor.py, cross-checked vs beta_highalpha_ccl.py. See
P1_ESTIMATOR_PROVENANCE_20260908.md.]

Original intent: independent second column-tilt estimator for P1 (physicist /
adversarial-referee lens). The structure-tensor estimator (beta_structure_tensor.py) is
one method whose predecessor (watershed) was wrong by 15-20 deg; this is a mechanically
different route so the two can be cross-checked inside the 16-84 band.

Method: a film of columns tilted by beta from the substrate normal shows tilted streaks
in a vertical (x-z) density slice. The 2-D power spectrum of that slice has a ridge
ORTHOGONAL to the streak direction; the streak inclination from vertical is beta.
  1. wrap positions into [0, box); mature band z in [0.35 H, 0.85 H].
  2. for a set of y-slabs, project atoms to an (x, z) occupancy image (VOX nm pixels).
  3. 2-D FFT, magnitude spectrum, remove DC / low-freq disk.
  4. orientation histogram of spectral power vs angle theta (0 = kx axis); the peak
     theta_p gives the streak normal; streak inclination from the z-axis is
     beta = |90 - theta_p_deg_measured_from_x|  (folded into [0, 90)).
  5. occupancy-weighted median over slabs + 16-84 band.

CLI: python3 beta_fft_streak.py <checkpoint.h5> [--box 100] [--vox 1.0]
API: beta_fft_one(path, box, vox) -> dict(beta, lo, hi, n_slabs, frac)
"""
from __future__ import annotations
import sys
import numpy as np

try:
    import h5py
except Exception:
    h5py = None

VOX = 1.0
BAND = (0.35, 0.85)
N_SLABS = 24
NBINS_THETA = 180
LOWFREQ_CUT = 3          # pixels: radius of the DC disk removed


def _slab_beta(img: np.ndarray):
    """beta (deg from vertical) of the dominant streak in a 2-D (x rows, z cols) image."""
    img = img - img.mean()
    if img.std() < 1e-9:
        return None
    F = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    ny, nx = F.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    F[r < LOWFREQ_CUT] = 0.0
    # angle of each spectral pixel measured from the +x (kx) axis
    ang = np.degrees(np.arctan2(yy - cy, xx - cx))     # (-180, 180]
    ang = np.mod(ang, 180.0)                            # ridge is symmetric -> [0,180)
    w = F.ravel()
    a = ang.ravel()
    m = w > 0
    if m.sum() < 50:
        return None
    hist, edges = np.histogram(a[m], bins=NBINS_THETA, range=(0, 180), weights=w[m])
    theta_ridge = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    # spectral ridge is orthogonal to the real-space streak. streak angle from x-axis:
    streak_from_x = np.mod(theta_ridge + 90.0, 180.0)
    # beta is measured from the z (vertical) axis; image cols are z, rows are x, so a
    # streak at 90 deg from x is vertical (beta=0); fold to [0,90).
    beta = abs(90.0 - streak_from_x)
    if beta > 90.0:
        beta = 180.0 - beta
    return float(beta)


def beta_fft_one(path: str, box: float | None, vox: float = VOX) -> dict | None:
    if h5py is None:
        raise RuntimeError("h5py required")
    with h5py.File(path, "r") as f:
        P = np.asarray(f["positions"][:, :3], np.float64)
        bw = box or float(f.attrs.get("box_width", 0)) or 0.0
    if bw <= 0:
        bw = 5 * round(max(P[:, 0].ptp(), P[:, 1].ptp()) / 5)
    P[:, 0] = np.mod(P[:, 0], bw)
    P[:, 1] = np.mod(P[:, 1], bw)
    H = P[:, 2].max()
    zlo, zhi = BAND[0] * H, BAND[1] * H
    Q = P[(P[:, 2] >= zlo) & (P[:, 2] < zhi)]
    if len(Q) < 2000:
        return None
    nx = max(8, int(np.ceil(bw / vox)))
    nz = max(8, int(np.ceil((zhi - zlo) / vox)))
    betas, wts = [], []
    y_edges = np.linspace(0, bw, N_SLABS + 1)
    for s in range(N_SLABS):
        sl = Q[(Q[:, 1] >= y_edges[s]) & (Q[:, 1] < y_edges[s + 1])]
        if len(sl) < 300:
            continue
        img = np.zeros((nx, nz), np.float32)
        ix = np.clip(((sl[:, 0]) / vox).astype(int), 0, nx - 1)
        iz = np.clip(((sl[:, 2] - zlo) / vox).astype(int), 0, nz - 1)
        np.add.at(img, (ix, iz), 1.0)
        b = _slab_beta(img)
        if b is not None:
            betas.append(b); wts.append(len(sl))
    if len(betas) < 4:
        return dict(beta=None, n_slabs=len(betas), frac=0.0)
    betas = np.array(betas); wts = np.array(wts, float)
    o = np.argsort(betas); cw = np.cumsum(wts[o]); cw /= cw[-1]
    med = float(np.interp(0.5, cw, betas[o]))
    lo = float(np.interp(0.16, cw, betas[o]))
    hi = float(np.interp(0.84, cw, betas[o]))
    return dict(beta=med, lo=lo, hi=hi, n_slabs=len(betas),
                frac=len(betas) / N_SLABS)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    box = None; vox = VOX
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--box":
            box = float(next(it))
        elif a == "--vox":
            vox = float(next(it))
    r = beta_fft_one(args[0], box, vox)
    print(r)
