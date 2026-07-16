"""
Reproduce the (alpha,z)-oracle diagnostic for P2 (2026-07-10, Phase-2 audit S2).

The original run's script was not preserved (see P2 ssec:res_oracle provenance note).
This script recomputes the same two numbers directly from the archived dataset
(datasets/levelA_helical_nif/{train,test}_samples.npz, generated 2026-06-15, columns
x_n, y_n, z_n, alpha_norm, rho -- confirmed via normalization.json and metadata.json):

  1. Table tab:label_stats: per-alpha mean(rho), Var(rho), f_soft = P(rho in (0.1,0.9))
     on the TRAIN split.
  2. Fig oracle_delta_iou: an (alpha,z_bin)-oracle (mean train rho per (alpha,z_bin)
     cell, z at 2nm resolution over 151 bins) evaluated on the TEST split with a
     threshold sweep tau in [0.05,0.95]; per-alpha best-tau IoU minus the trivial
     all-occupied baseline (IoU_trivial = test occupied fraction f = P(rho>0.5)),
     exactly the "reference baseline is the all-positive segmentation" definition
     used throughout both papers.
"""
import numpy as np

ALPHA_NORM_TO_DEG = {
    -1.0: 60.0, -0.655: 65.0, -0.310: 70.0, 0.034: 75.0,
    0.379: 80.0, 0.724: 85.0, 0.862: 87.0, 1.0: 89.0,
}
N_Z_BINS = 151
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def alpha_key(alpha_norm_col):
    keys = np.array(sorted(ALPHA_NORM_TO_DEG.keys()))
    idx = np.abs(alpha_norm_col[:, None] - keys[None, :]).argmin(axis=1)
    return keys[idx]


def iou(pred_bin, true_bin):
    inter = (pred_bin & true_bin).sum()
    union = (pred_bin | true_bin).sum()
    return inter / union if union > 0 else np.nan


def main():
    train = np.load("datasets/levelA_helical_nif/train_samples.npz")["data"]
    test = np.load("datasets/levelA_helical_nif/test_samples.npz")["data"]

    train_alpha = alpha_key(train[:, 3])
    test_alpha = alpha_key(test[:, 3])

    print("=== Table tab:label_stats reproduction (train split) ===")
    print(f"{'alpha':>6} {'mean_rho':>10} {'var_rho':>10} {'f_soft(%)':>10}")
    for k in sorted(ALPHA_NORM_TO_DEG):
        deg = ALPHA_NORM_TO_DEG[k]
        mask = train_alpha == k
        rho = train[mask, 4]
        mean_rho = rho.mean()
        var_rho = rho.var()
        f_soft = ((rho > 0.1) & (rho < 0.9)).mean() * 100
        print(f"{deg:6.0f} {mean_rho:10.3f} {var_rho:10.3f} {f_soft:9.1f}%")

    print()
    print("=== Fig oracle_delta_iou reproduction ===")
    z_bin_train = np.clip(np.round(train[:, 2] * (N_Z_BINS - 1)).astype(int), 0, N_Z_BINS - 1)
    z_bin_test = np.clip(np.round(test[:, 2] * (N_Z_BINS - 1)).astype(int), 0, N_Z_BINS - 1)

    print(f"{'alpha':>6} {'f_test':>8} {'iou_trivial':>12} {'best_iou':>10} {'delta_iou':>10} {'best_tau':>9}")
    results = {}
    for k in sorted(ALPHA_NORM_TO_DEG):
        deg = ALPHA_NORM_TO_DEG[k]
        train_mask = train_alpha == k
        test_mask = test_alpha == k

        # Build (alpha,z_bin) oracle from TRAIN: mean rho per z_bin cell for this alpha
        oracle_by_zbin = np.full(N_Z_BINS, np.nan)
        zb_train = z_bin_train[train_mask]
        rho_train = train[train_mask, 4]
        for zb in range(N_Z_BINS):
            cell = rho_train[zb_train == zb]
            if cell.size > 0:
                oracle_by_zbin[zb] = cell.mean()
        # fill any empty bins with the alpha's overall train mean (edge case)
        global_mean = rho_train.mean()
        oracle_by_zbin = np.where(np.isnan(oracle_by_zbin), global_mean, oracle_by_zbin)

        zb_test = z_bin_test[test_mask]
        rho_test = test[test_mask, 4]
        true_bin = rho_test > 0.5
        f_test = true_bin.mean()
        iou_trivial = f_test  # all-positive segmentation baseline, IoU = f exactly

        oracle_pred_continuous = oracle_by_zbin[zb_test]
        best_iou = -1.0
        best_tau = np.nan
        for tau in THRESHOLDS:
            pred_bin = oracle_pred_continuous > tau
            val = iou(pred_bin, true_bin)
            if not np.isnan(val) and val > best_iou:
                best_iou = val
                best_tau = tau

        delta_iou = best_iou - iou_trivial
        results[deg] = (f_test, iou_trivial, best_iou, delta_iou, best_tau)
        print(f"{deg:6.0f} {f_test:8.3f} {iou_trivial:12.4f} {best_iou:10.4f} {delta_iou:10.4f} {best_tau:9.2f}")

    print()
    print("Paper's claim: delta_iou <= 0.007 for all alpha in {65,...,87} deg.")
    max_delta_main = max(abs(results[d][3]) for d in (65, 70, 75, 80, 85, 87))
    print(f"Reproduced max |delta_iou| over 65-87 deg: {max_delta_main:.4f}")


if __name__ == "__main__":
    main()
