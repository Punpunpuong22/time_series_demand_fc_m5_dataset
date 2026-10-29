"""Kiểm định thống kê (mục 7.6).

- Diebold–Mariano với hiệu chỉnh mẫu nhỏ Harvey–Leybourne–Newbold (HLN), theo từng chuỗi.
- Hiệu chỉnh kiểm định bội Benjamini–Hochberg.
- Friedman + Nemenyi (khoảng cách tới hạn) để xếp hạng nhiều cấu hình.
- Bootstrap theo chuỗi cho khoảng tin cậy của chênh lệch WRMSSE.
- Mô hình hồi quy hỗn hợp cho tương tác mô hình × bộ đặc trưng × loại ngày (H2b).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> tuple[float, float]:
    """H0: hai dự báo chính xác như nhau. Trả về (thống kê DM-HLN, p hai phía)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    if T < 3 or np.allclose(d, 0):
        return 0.0, 1.0
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[: T - k]) / T for k in range(h)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        var = gamma[0]
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """p-value đã hiệu chỉnh (q-value) theo Benjamini–Hochberg."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def dm_by_series(preds_a: pd.DataFrame, preds_b: pd.DataFrame, h: int = 28, alpha: float = 0.05) -> pd.DataFrame:
    """So sánh hai cấu hình trên từng chuỗi (ghép mọi cửa sổ đánh giá theo thứ tự thời gian).

    Thống kê dương nghĩa là cấu hình A có sai số bình phương lớn hơn B.
    """
    m = preds_a.merge(preds_b, on=["sid", "d"], suffixes=("_a", "_b")).sort_values(["sid", "d"])
    rows = []
    for sid, g in m.groupby("sid"):
        ea = (g["y_pred_a"] - g["y_true_a"]).to_numpy()
        eb = (g["y_pred_b"] - g["y_true_b"]).to_numpy()
        s, p = dm_test(ea, eb, h=h)
        rows.append((sid, s, p))
    res = pd.DataFrame(rows, columns=["sid", "dm_stat", "p_value"])
    res["q_value"] = benjamini_hochberg(res["p_value"].to_numpy())
    res["significant"] = res["q_value"] < alpha
    res["better"] = np.where(~res["significant"], "tie", np.where(res["dm_stat"] > 0, "B", "A"))
    return res


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    """scores: bảng rộng (dòng = chuỗi, cột = cấu hình), giá trị càng nhỏ càng tốt."""
    s = scores.dropna()
    k, n = s.shape[1], s.shape[0]
    chi2, p = stats.friedmanchisquare(*[s[c].to_numpy() for c in s.columns])
    ranks = s.rank(axis=1).mean().sort_values()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {"chi2": float(chi2), "p_value": float(p), "mean_ranks": ranks.to_dict(), "critical_difference": float(cd), "n": n}


def bootstrap_wrmsse_diff(rmsse_a: np.ndarray, rmsse_b: np.ndarray, weights: np.ndarray,
                          n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Khoảng tin cậy bootstrap (lấy mẫu lại chuỗi) cho WRMSSE(A) − WRMSSE(B)."""
    a, b, w = map(lambda x: np.asarray(x, float), (rmsse_a, rmsse_b, weights))
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b, w = a[ok], b[ok], w[ok]
    rng = np.random.default_rng(seed)
    n = len(a)

    def wr(x, ww):
        return float((ww * x).sum() / ww.sum())

    point = wr(a, w) - wr(b, w)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = wr(a[idx], w[idx]) - wr(b[idx], w[idx])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi), "n_series": n}


def interaction_model(strat: pd.DataFrame):
    """log(RMSSE phân tầng) ~ mô hình × bộ đặc trưng × loại ngày, hiệu ứng ngẫu nhiên theo chuỗi.

    strat cần các cột: sid, family (lgbm/lstm), feature_set (H/HC/HCP), daytype, rmsse.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần statsmodels: pip install statsmodels") from e
    df = strat.dropna(subset=["rmsse"]).copy()
    df = df[df["rmsse"] > 0]
    df["log_rmsse"] = np.log(df["rmsse"])
    model = smf.mixedlm("log_rmsse ~ C(family) * C(feature_set) * C(daytype)", df, groups=df["sid"])
    return model.fit(method="lbfgs")
