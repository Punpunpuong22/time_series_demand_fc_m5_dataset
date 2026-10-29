"""Độ đo đánh giá (mục 7.5): MAE, RMSE, độ chệch, RMSSE, WRMSSE và RMSSE phân tầng theo loại ngày.

Mẫu số của RMSSE và trọng số WRMSSE được tính tại mốc của từng cửa sổ, chỉ từ dữ liệu ≤ mốc.
RMSSE phân tầng giữ nguyên mẫu số của RMSSE để các tầng so sánh được với nhau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmsse_denominator(panel, origin: int) -> np.ndarray:
    """(1/(n−1)) Σ (y_t − y_{t−1})², từ ngày bán khác 0 đầu tiên đến mốc. NaN nếu chuỗi chưa bán."""
    Y = panel.Y[:, :origin].astype(np.float64)
    has = (Y > 0).any(axis=1)
    first = np.where(has, (Y > 0).argmax(axis=1), origin)
    dif2 = np.diff(Y, axis=1) ** 2
    mask = np.arange(1, origin)[None, :] > first[:, None]
    num = (dif2 * mask).sum(axis=1)
    cnt = mask.sum(axis=1)
    with np.errstate(all="ignore"):
        den = np.where(cnt > 0, num / cnt, np.nan)
    den[den == 0] = np.nan  # chuỗi hằng: loại khỏi đánh giá
    return den


def revenue_weights(panel, origin: int, days: int = 28) -> np.ndarray:
    """Trọng số WRMSSE: doanh thu 28 ngày cuối trước mốc, chuẩn hóa tổng = 1 trong tập chuỗi đang xét."""
    cols = slice(origin - days, origin)
    rev = np.nansum(panel.Y[:, cols] * np.nan_to_num(panel.P[:, cols]), axis=1)
    total = rev.sum()
    return rev / total if total > 0 else np.full(panel.n, 1.0 / panel.n)


def attach_truth(panel, preds: pd.DataFrame) -> pd.DataFrame:
    """Thêm y_true cho bảng dự báo (sid, d, y_pred)."""
    preds = preds.copy()
    preds["y_true"] = panel.Y[preds["sid"].to_numpy(), preds["d"].to_numpy() - 1]
    return preds


def series_window_scores(panel, preds: pd.DataFrame, windows) -> pd.DataFrame:
    """Điểm theo (chuỗi, cửa sổ): mae, rmse, mse, bias, rmsse, weight."""
    out = []
    by_name = {w.name: w for w in windows}
    for wname, g in preds.groupby("window", sort=False):
        w = by_name[wname]
        den = rmsse_denominator(panel, w.origin)
        wt = revenue_weights(panel, w.origin)
        e = (g["y_pred"] - g["y_true"]).to_numpy()
        df = pd.DataFrame({"sid": g["sid"].to_numpy(), "e": e, "ae": np.abs(e), "se": e * e})
        agg = df.groupby("sid").agg(mae=("ae", "mean"), mse=("se", "mean"), bias=("e", "mean"))
        agg["rmse"] = np.sqrt(agg["mse"])
        agg["rmsse"] = np.sqrt(agg["mse"] / den[agg.index.to_numpy()])
        agg["weight"] = wt[agg.index.to_numpy()]
        agg["window"] = wname
        out.append(agg.reset_index())
    return pd.concat(out, ignore_index=True)


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp theo cửa sổ: WRMSSE (trọng số doanh thu, chuẩn hóa lại trên chuỗi hợp lệ) và trung bình."""
    def _agg(g):
        ok = g["rmsse"].notna()
        w = g.loc[ok, "weight"]
        w = w / w.sum() if w.sum() > 0 else w
        return pd.Series({
            "wrmsse": float((w * g.loc[ok, "rmsse"]).sum()),
            "rmsse_mean": float(g.loc[ok, "rmsse"].mean()),
            "mae": float(g["mae"].mean()),
            "rmse": float(g["rmse"].mean()),
            "bias": float(g["bias"].mean()),
            "n_series": int(ok.sum()),
        })
    return scores.groupby("window", sort=False).apply(_agg).reset_index()


def stratified_scores(panel, preds: pd.DataFrame, windows, labels: pd.DataFrame) -> pd.DataFrame:
    """RMSSE phân tầng theo loại ngày.

    labels: DataFrame (sid, d, daytype). Trả về (sid, window, daytype, n_days, mse, rmsse, bias, weight).
    """
    df = preds.merge(labels, on=["sid", "d"], how="left")
    by_name = {w.name: w for w in windows}
    out = []
    for wname, g in df.groupby("window", sort=False):
        w = by_name[wname]
        den = rmsse_denominator(panel, w.origin)
        wt = revenue_weights(panel, w.origin)
        e = (g["y_pred"] - g["y_true"]).to_numpy()
        tmp = pd.DataFrame({"sid": g["sid"].to_numpy(), "daytype": g["daytype"].to_numpy(), "e": e, "se": e * e})
        agg = tmp.groupby(["sid", "daytype"]).agg(n_days=("e", "size"), mse=("se", "mean"), bias=("e", "mean")).reset_index()
        agg["rmsse"] = np.sqrt(agg["mse"] / den[agg["sid"].to_numpy()])
        agg["weight"] = wt[agg["sid"].to_numpy()]
        agg["window"] = wname
        out.append(agg)
    return pd.concat(out, ignore_index=True)


def aggregate_stratified(strat: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp RMSSE phân tầng theo loại ngày (trung bình qua cửa sổ): có trọng số và không trọng số."""
    def _agg(g):
        ok = g["rmsse"].notna()
        w = g.loc[ok, "weight"]
        return pd.Series({
            "wrmsse": float((w * g.loc[ok, "rmsse"]).sum() / w.sum()) if w.sum() > 0 else np.nan,
            "rmsse_mean": float(g.loc[ok, "rmsse"].mean()),
            "bias": float(g["bias"].mean()),
            "n_obs": int(g["n_days"].sum()),
        })
    per_window = strat.groupby(["window", "daytype"]).apply(_agg).reset_index()
    return per_window.groupby("daytype")[["wrmsse", "rmsse_mean", "bias", "n_obs"]].agg(
        {"wrmsse": "mean", "rmsse_mean": "mean", "bias": "mean", "n_obs": "sum"}
    ).reset_index()
