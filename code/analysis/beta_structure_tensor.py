"""beta_structure_tensor.py -- density-adaptive column tilt beta (lens 2 / physicist).

The two existing sim-beta estimators each work only in one density regime:
  - beta_robust_column_axis.py (per-slab watershed + centroid tracking): needs SEPARATED
    columns -> fails at alpha >= 87 (merged) AND, subtly, over-reports at low alpha where
    the near-continuous film resists segmentation.
  - beta_highalpha_ccl.py (3-D connected components): needs SPARSE columns -> fails at
    alpha <= 85 (whole mature band = one component).

This estimator needs NEITHER -- it reads the tilt straight off the density field via a
3-D structure tensor, so it works whether the columns are discrete or a merged slab:

  1. voxelize the mature band z in [0.35 H, 0.85 H] at VOX nm, Gaussian-smoothed occupancy.
  2. per voxel, 3-D gradient g; structure tensor S = <g g^T> over a local window.
  3. keep voxels that are "columnar" -- planar/linear anisotropy above ANISO_MIN and
     occupancy in a mid band (a solid interior voxel has ~0 gradient; a noise voxel is
     isotropic). The column axis is the structure-tensor eigenvector of SMALLEST eigenvalue
     (the direction of least density variation = along the column).
  4. beta = angle of that axis from +z; report the occupancy-weighted median over the band,
     plus a bootstrap 16-84 percentile band and the fraction of the band that qualified.

Cross-check: on separated films it should agree with the watershed / CCL estimators; its
value-add is that it also returns a number at alpha=85-89 where they do not.

Usage: beta_structure_tensor.py <checkpoint.h5> [<checkpoint.h5> ...] [--vox 1.0] [--box N]
"""
from __future__ import annotations
import sys, re, numpy as np, h5py
from scipy import ndimage

VOX = 1.0
BAND = (0.35, 0.85)
SMOOTH_NM = 1.5
WIN_NM = 4.0
ANISO_MIN = 0.15          # (l_mid - l_min) / (l_max + eps): reject near-isotropic voxels
OCC_LO, OCC_HI = 0.05, 0.9   # smoothed-occupancy window for "surface/columnar" voxels


def _grid(P, box, vox):
    H = P[:, 2].max()
    zlo, zhi = BAND[0] * H, BAND[1] * H
    m = (P[:, 2] >= zlo) & (P[:, 2] < zhi)
    Q = P[m]
    if len(Q) < 500:
        return None, H
    nx = max(4, int(np.ceil(box / vox)))
    nz = max(4, int(np.ceil((zhi - zlo) / vox)))
    g = np.zeros((nx, nx, nz), np.float32)
    ix = np.clip(((Q[:, 0] % box) / vox).astype(int), 0, nx - 1)
    iy = np.clip(((Q[:, 1] % box) / vox).astype(int), 0, nx - 1)
    iz = np.clip(((Q[:, 2] - zlo) / vox).astype(int), 0, nz - 1)
    np.add.at(g, (ix, iy, iz), 1.0)
    occ = ndimage.gaussian_filter((g > 0).astype(np.float32), SMOOTH_NM / vox, mode=("wrap", "wrap", "nearest"))
    dens = ndimage.gaussian_filter(g, SMOOTH_NM / vox, mode=("wrap", "wrap", "nearest"))
    return (occ, dens, H), H


def beta_one(path, box, vox):
    with h5py.File(path, "r") as f:
        P = np.asarray(f["positions"][:, :3], np.float64)
        bw = box or float(f.attrs.get("box_width", 0)) or 0.0
    if bw <= 0:
        bw = 5 * round(max(P[:, 0].ptp(), P[:, 1].ptp()) / 5)
    gg, H = _grid(P, bw, vox)
    if gg is None:
        return None
    occ, dens, _ = gg
    gx, gy, gz = np.gradient(dens)
    win = max(1, int(round(WIN_NM / vox)))
    sm = lambda a: ndimage.uniform_filter(a, win, mode=("wrap", "wrap", "nearest"))
    Sxx, Syy, Szz = sm(gx * gx), sm(gy * gy), sm(gz * gz)
    Sxy, Sxz, Syz = sm(gx * gy), sm(gx * gz), sm(gy * gz)

    mask = (occ > OCC_LO) & (occ < OCC_HI)
    idx = np.argwhere(mask)
    if len(idx) < 50:
        return dict(beta=None, n=0, frac=0.0, H=H)
    if len(idx) > 60000:
        idx = idx[np.random.default_rng(0).choice(len(idx), 60000, replace=False)]

    betas, wts = [], []
    for (i, j, k) in idx:
        S = np.array([[Sxx[i, j, k], Sxy[i, j, k], Sxz[i, j, k]],
                      [Sxy[i, j, k], Syy[i, j, k], Syz[i, j, k]],
                      [Sxz[i, j, k], Syz[i, j, k], Szz[i, j, k]]])
        ev, evec = np.linalg.eigh(S)          # ascending
        lo, mid, hi = ev
        if hi < 1e-12:
            continue
        aniso = (mid - lo) / (hi + 1e-12)
        if aniso < ANISO_MIN:
            continue
        axis = evec[:, 0]                      # least density variation -> along column
        if axis[2] < 0:
            axis = -axis
        b = np.degrees(np.arctan2(np.hypot(axis[0], axis[1]), abs(axis[2])))
        betas.append(b)
        wts.append(dens[i, j, k])
    if len(betas) < 30:
        return dict(beta=None, n=len(betas), frac=len(betas) / max(1, len(idx)), H=H)
    betas = np.array(betas); wts = np.array(wts)
    order = np.argsort(betas)
    cw = np.cumsum(wts[order]); cw /= cw[-1]
    med = float(np.interp(0.5, cw, betas[order]))
    p16 = float(np.interp(0.16, cw, betas[order]))
    p84 = float(np.interp(0.84, cw, betas[order]))
    return dict(beta=med, lo=p16, hi=p84, n=len(betas),
                frac=len(betas) / max(1, len(idx)), H=H)


if __name__ == "__main__":
    vox = VOX; box = None; args = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--vox":
            vox = float(next(it))
        elif a == "--box":
            box = float(next(it))
        elif a.startswith("--"):
            continue
        else:
            args.append(a)
    print("%-58s  %6s  %6s  %8s  %6s" % ("checkpoint", "H", "beta", "16-84", "frac"))
    for p in args:
        r = beta_one(p, box, vox)
        tag = "/".join(p.split("/")[-4:-2]) or p
        if r is None or r["beta"] is None:
            print("%-58s  %6s  %6s  %8s  %6s" % (tag, "-", "None", "-",
                                                 f"{(r or {}).get('frac',0):.2f}"))
        else:
            print("%-58s  %6.1f  %5.1f  %4.1f-%4.1f  %5.2f"
                  % (tag, r["H"], r["beta"], r["lo"], r["hi"], r["frac"]))
