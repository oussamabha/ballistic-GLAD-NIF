"""High-alpha column tilt beta via 3-D connected-component labelling.

beta_robust_column_axis.py (watershed) returns n=0 at alpha 87/89 -- the sparse,
partly-merged high-alpha film defeats peak_local_max/watershed. In that regime the
columns are instead WELL SEPARATED, so direct 3-D CCL isolates them cleanly.

Per checkpoint:
  1. voxelize solid fraction at VOX nm, mature band z in [0.35 H, 0.85 H]
  2. 3-D 26-connectivity label -> components
  3. keep components spanning >= MIN_DZ nm in z and >= MIN_VOX voxels
  4. per component: robust line fit centroid(z) -> (dx/dz, dy/dz), PBC-unwrapped
     beta_col = atan(hypot(sx, sy))
  5. report median beta over columns; aggregate across seeds

Usage: beta_highalpha_ccl.py <checkpoint.h5> [<checkpoint.h5> ...]
"""
import sys, numpy as np, h5py
from scipy import ndimage

VOX = 1.0          # nm
MIN_DZ = 30.0      # nm of vertical column extent required
MIN_VOX = 40       # occupied voxels required
BAND = (0.35, 0.85)

def beta_one(path):
    with h5py.File(path, "r") as f:
        P = np.asarray(f["positions"][:, :3], float)
        bw = float(f.attrs.get("box_width", 100.0)) or 100.0
    H = P[:, 2].max()
    zlo, zhi = BAND[0] * H, BAND[1] * H
    m = (P[:, 2] >= zlo) & (P[:, 2] < zhi)
    P = P[m]
    if len(P) < 500:
        return None, 0, H
    nx = int(np.ceil(bw / VOX))
    nz = int(np.ceil((zhi - zlo) / VOX))
    ix = np.clip(((P[:, 0] % bw) / VOX).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] % bw) / VOX).astype(int), 0, nx - 1)
    iz = np.clip(((P[:, 2] - zlo) / VOX).astype(int), 0, nz - 1)
    grid = np.zeros((nx, nx, nz), bool)
    grid[ix, iy, iz] = True
    # wrap x,y for PBC-aware labelling: pad one plane each side
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    betas = []
    for c in range(1, n + 1):
        cx, cy, cz = np.where(lab == c)
        if len(cx) < MIN_VOX:
            continue
        zspan = (cz.max() - cz.min()) * VOX
        if zspan < MIN_DZ:
            continue
        # centroid per z-layer
        zs = np.unique(cz)
        if len(zs) < MIN_DZ / VOX * 0.6:
            continue
        xm = np.array([cx[cz == z].mean() for z in zs]) * VOX
        ym = np.array([cy[cz == z].mean() for z in zs]) * VOX
        zz = zs * VOX
        # PBC unwrap
        xm = np.unwrap(xm / bw * 2 * np.pi) / (2 * np.pi) * bw
        ym = np.unwrap(ym / bw * 2 * np.pi) / (2 * np.pi) * bw
        sx = np.polyfit(zz, xm, 1)[0]
        sy = np.polyfit(zz, ym, 1)[0]
        betas.append(np.degrees(np.arctan(np.hypot(sx, sy))))
    if not betas:
        return None, 0, H
    return float(np.median(betas)), len(betas), H

if __name__ == "__main__":
    from collections import defaultdict
    agg = defaultdict(list)
    for p in sys.argv[1:]:
        b, n, H = beta_one(p)
        tag = "/".join(p.split("/")[-4:-2])
        print(f"{tag:55s}  H={H:6.1f}  beta={b if b is None else round(b,1)}  (n_col={n})")
        if b is not None:
            # crude alpha parse from path
            import re
            mm = re.search(r"alpha0?(\d\d)", p)
            a = mm.group(1) if mm else "?"
            agg[a].append(b)
    print("\n=== median beta_sim by alpha (CCL, high-alpha) ===")
    for a in sorted(agg):
        v = agg[a]
        print(f"  alpha {a}:  {np.median(v):.1f} deg   (over {len(v)} seed-checkpoints, spread {min(v):.1f}-{max(v):.1f})")
