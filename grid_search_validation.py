"""
Grid-search validation of the empirical midpoint threshold (Section 3.4/5.3).

Reproduces the confirmatory analysis reported in the article: an explicit
grid search over the 50-pair calibration sample, jointly maximizing the
F1-score and Youden's J index (sensitivity + specificity - 1), to check
that the empirical midpoint threshold (0.7858) falls within the
grid-search-optimal decision region.

Expected output (as reported in the article):
    Decision region (F1 and Youden's J both optimal): [0.7736, 0.7886]
    Plateau width: 0.015
    Empirical midpoint threshold (0.7858) falls within this region: True

Input: calibration_mpnet_finale.csv
    - similarite_mpnet   : cosine similarity score (Sentence-BERT mpnet)
    - type_pair_final    : consensus label, "MATCH" or "NON_MATCH"
"""

import numpy as np
import pandas as pd

CSV_PATH = "calibration_mpnet_finale.csv"
SCORE_COL = "similarite_mpnet"
LABEL_COL = "type_pair_final"
STEP = 0.0001  # grid resolution


def youden_j(tp, fn, tn, fp):
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity + specificity - 1


def f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    df = pd.read_csv(CSV_PATH)
    y_true = (df[LABEL_COL] == "MATCH").to_numpy()
    scores = df[SCORE_COL].to_numpy()

    thresholds = np.arange(scores.min(), scores.max() + STEP, STEP)

    f1_values = []
    j_values = []
    for t in thresholds:
        y_pred = scores >= t
        tp = int(np.sum(y_pred & y_true))
        fp = int(np.sum(y_pred & ~y_true))
        fn = int(np.sum(~y_pred & y_true))
        tn = int(np.sum(~y_pred & ~y_true))
        f1_values.append(f1_score(tp, fp, fn))
        j_values.append(youden_j(tp, fn, tn, fp))

    f1_values = np.array(f1_values)
    j_values = np.array(j_values)

    f1_max = f1_values.max()
    j_max = j_values.max()

    f1_optimal_mask = np.isclose(f1_values, f1_max)
    j_optimal_mask = np.isclose(j_values, j_max)
    both_optimal_mask = f1_optimal_mask & j_optimal_mask

    optimal_thresholds = thresholds[both_optimal_mask]
    region_low, region_high = optimal_thresholds.min(), optimal_thresholds.max()
    plateau_width = region_high - region_low

    midpoint_threshold = 0.7858
    midpoint_in_region = region_low <= midpoint_threshold <= region_high

    print(f"F1 max            : {f1_max:.4f}")
    print(f"Youden's J max     : {j_max:.4f}")
    print(f"Decision region    : [{region_low:.4f}, {region_high:.4f}]")
    print(f"Plateau width      : {plateau_width:.4f}")
    print(f"Midpoint threshold : {midpoint_threshold}")
    print(f"Midpoint in region : {midpoint_in_region}")


if __name__ == "__main__":
    main()
