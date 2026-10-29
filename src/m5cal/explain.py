"""Diễn giải mô hình (mục 7.7).

Thước đo CHUNG cho cả hai mô hình: permutation importance theo nhóm đặc trưng
(xáo trộn các cột/kênh của một nhóm giữa các chuỗi, đo mức tăng WRMSSE).
TreeSHAP (pred_contrib của LightGBM) chỉ là phân tích bổ sung cho LightGBM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import aggregate_scores, attach_truth, series_window_scores

GROUPS = ["cal_regular", "cal_irregular", "snap", "price", "id", "lag", "roll"]


def grouped_permutation_importance(model, panel, calf, windows, groups=None, n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Mức tăng WRMSSE trung bình khi xáo trộn từng nhóm. model cần có predict(..., permute=, rng=)."""
    groups = groups or GROUPS

    def _score(permute=None, rng=None):
        preds = pd.concat([attach_truth(panel, model.predict(panel, calf, w, permute=permute, rng=rng)).assign(window=w.name)
                           for w in windows])
        return aggregate_scores(series_window_scores(panel, preds, windows))["wrmsse"].mean()

    base = _score()
    rows = []
    for g in groups:
        vals = [_score(g, np.random.default_rng(seed + r)) for r in range(n_repeats)]
        rows.append({"group": g, "wrmsse_increase": float(np.mean(vals) - base), "sd": float(np.std(vals))})
    return pd.DataFrame(rows).sort_values("wrmsse_increase", ascending=False).reset_index(drop=True)


def treeshap_by_group(lgbm_model, panel, calf, window) -> pd.DataFrame:
    """Trung bình |SHAP| theo nhóm đặc trưng cho LightGBM (giá trị Shapley chính xác của TreeSHAP)."""
    from .features import feature_group_of

    res = lgbm_model.predict(panel, calf, window, return_contrib=True)
    feat_cols = [c for c in res.columns if c not in ("sid", "d", "y_pred")]
    mean_abs = res[feat_cols].abs().mean()
    df = mean_abs.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    df["group"] = df["feature"].map(feature_group_of)
    return df.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()
