"""Graded interpenetration: what fraction of atoms are meaningfully closer than contact?
median NN == cd exactly (hard-sphere contact model), so `nn < cd - 1e-3` just catches
float-rounding. Real interpenetration needs a real threshold."""
import numpy as np, h5py
from scipy.spatial import cKDTree

def graded(path, box):
    with h5py.File(path, "r") as f:
        P = np.asarray(f["positions"][:, :3], np.float64)
        cd = float(f.attrs.get("collision_diameter", 0.256))
    x = np.mod(P[:, 0], box); y = np.mod(P[:, 1], box)
    z = P[:, 2] - P[:, 2].min() + 1.0
    Q = np.column_stack([x, y, z])
    tree = cKDTree(Q, boxsize=[box, box, z.max() + 2.0])
    dd, _ = tree.query(Q, k=2, workers=-1)
    nn = dd[:, 1]
    fr = lambda t: 100.0 * np.mean(nn < cd * (1 - t))
    return cd, len(P), fr(0.004), fr(0.02), fr(0.05), fr(0.10), fr(0.20), float(np.median(nn)/cd)

R = "/mnt/d/GLAD_PROJECT/01_GLAD_SIMULATION/simulation_batch/runs/"
cases = [
    ("PREFIX hopcap1  b40 diffON c1",   R+"P1_DIFFUSION_HOPCAP_TEST_V1/hopcap1_test/checkpoints/checkpoint_v3_A.h5", 40.0),
    ("PREFIX hopcap200 b40 diffON c200", R+"P1_DIFFUSION_HOPCAP_TEST_V1/hopcap200_baseline/checkpoints/checkpoint_v3_A.h5", 40.0),
    ("GRIDFIX a85 b100 diff OFF",        R+"P1_GRIDFIX_CAMPAIGN_V1/P1_alpha085_seed000_300nm_pitch150/checkpoints/checkpoint_v3_emergency.h5", 100.0),
    ("GRIDFIX a75 b100 diff OFF",        R+"P1_GRIDFIX_CAMPAIGN_V1/P1_alpha075_seed000_300nm_pitch150/checkpoints/checkpoint_v3_emergency.h5", 100.0),
    ("V2 a89 s0 b100 diff OFF",          R+"P1_CALIBRATED_r128_CAMPAIGN_V2_POSTFIX_20260906/P1r128v2_alpha089_seed000_box100_300nm/checkpoints/checkpoint_v3_emergency.h5", 100.0),
    ("D1 step3 b40 diffON c200 FIXON",   R+"P1_DIFFUSION_D1_VALIDATION_20260907/step3_sanity_a85_300nm/checkpoints/checkpoint_v3_emergency.h5", 40.0),
]
print("%-32s %8s  %7s %7s %7s %7s %7s  medNN/cd" % ("case","N",">0.4%",">2%",">5%",">10%",">20%"))
for name, p, b in cases:
    try:
        cd, n, a, c, e, g, h, m = graded(p, b)
        print("%-32s %8d  %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%   %.3f" % (name, n, a, c, e, g, h, m))
    except Exception as ex:
        print("%-32s  ERR %s" % (name, ex))
