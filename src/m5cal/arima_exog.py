"""ARIMA có biến ngoại sinh (ARIMAX, dùng SARIMAX của statsmodels) cho dự báo 1 ngày tới, từng mặt hàng.

Cách làm:
1. Chọn bậc (p, q) cho từng mặt hàng bằng AIC trên tập huấn luyện, với mô hình chỉ có lịch sử (S0).
   Bậc này được giữ nguyên cho mọi bộ biến lịch, để chênh lệch giữa các bộ chỉ đến từ biến lịch.
2. Ước lượng tham số trên tập huấn luyện.
3. Giữ nguyên tham số, đưa toàn bộ chuỗi (huấn luyện + kiểm định + kiểm tra) vào bộ lọc Kalman
   (`results.apply`) và lấy dự báo một bước (`get_prediction(start=n_train)`): dự báo cho ngày t chỉ
   dùng doanh số đến ngày t−1 và biến lịch của chính ngày t (biết trước), không huấn luyện lại.

Mô hình được truyền vào qua `model_cls` để kiểm thử được mà không cần statsmodels.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _sarimax():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần statsmodels (Colab đã cài sẵn): pip install statsmodels") from e
    return SARIMAX


def _fit(model_cls, y, X, order):
    mod = model_cls(y, exog=X, order=order, trend="c",
                    enforce_stationarity=False, enforce_invertibility=False)
    return mod.fit(disp=False, maxiter=200)


def select_order(y_train: np.ndarray, candidates: Sequence[tuple[int, int, int]] = ((1, 0, 0), (1, 0, 1), (2, 0, 1)),
                 model_cls=None) -> tuple[int, int, int]:
    """Bậc có AIC nhỏ nhất trên tập huấn luyện (mô hình không có biến ngoại sinh)."""
    model_cls = model_cls or _sarimax()
    best, best_aic = candidates[0], np.inf
    for order in candidates:
        try:
            aic = _fit(model_cls, np.asarray(y_train, float), None, order).aic
        except Exception:  # bậc không hội tụ thì bỏ qua
            continue
        if np.isfinite(aic) and aic < best_aic:
            best, best_aic = order, aic
    return best


def one_step_predictions(y: np.ndarray, X: np.ndarray | None, n_train: int, order: tuple[int, int, int],
                         model_cls=None) -> np.ndarray:
    """Dự báo một bước cho các vị trí n_train, ..., len(y)−1 với tham số ước lượng trên y[:n_train]."""
    model_cls = model_cls or _sarimax()
    y = np.asarray(y, float)
    if X is not None:
        X = np.asarray(X, float)
        keep = X[:n_train].std(axis=0) > 0                     # bỏ cột hằng trong tập huấn luyện
        X = X[:, keep] if keep.any() else None
    res = _fit(model_cls, y[:n_train], None if X is None else X[:n_train], order)
    res_all = res.apply(y, exog=X)                             # cùng tham số, toàn bộ chuỗi
    pred = np.asarray(res_all.get_prediction(start=n_train).predicted_mean, float)
    if len(pred) != len(y) - n_train:
        raise RuntimeError("Số dự báo không khớp số ngày sau tập huấn luyện")
    return pred


def _arima_item(item, g, exog_cols, train_end, fallback_end, min_train, order, candidates, model_cls,
                item_col, date_col, y_col):
    """Một mặt hàng: trả về (item, bậc, DataFrame dự báo) hoặc (item, None, None) nếu quá ít dữ liệu."""
    g = g.sort_values(date_col)
    y = g[y_col].to_numpy(float)
    n_train = int((g[date_col] <= train_end).sum())
    if n_train < min_train and fallback_end is not None:
        n_train = int((g[date_col] <= fallback_end).sum())
    if n_train < min_train or n_train >= len(g):
        return item, None, None
    order = order or select_order(y[:n_train], candidates, model_cls)
    X = g[exog_cols].to_numpy(float) if exog_cols else None
    try:
        pred = one_step_predictions(y, X, n_train, order, model_cls)
    except Exception:                                            # dự phòng: bỏ biến ngoại sinh
        pred = one_step_predictions(y, None, n_train, order, model_cls)
    out = pd.DataFrame({item_col: item, date_col: g[date_col].to_numpy()[n_train:], "pred": np.clip(pred, 0, None)})
    return item, order, out


def run_arima_panel(df: pd.DataFrame, exog_cols: Iterable[str], train_end, orders: dict | None = None,
                    candidates=((1, 0, 0), (1, 0, 1), (2, 0, 1)), item_col: str = "item_id",
                    date_col: str = "date", y_col: str = "demand", model_cls=None, n_jobs: int = 1,
                    fallback_end=None, min_train: int = 30) -> pd.DataFrame:
    """Chạy ARIMAX cho mọi mặt hàng. Trả về (item_id, date, pred) cho mọi ngày sau mốc ước lượng.

    Mốc ước lượng là `train_end`; mặt hàng lên kệ muộn (ít hơn `min_train` ngày trước `train_end`) được
    ước lượng đến `fallback_end` (cuối tập kiểm định) nếu có, để vẫn có dự báo cho tập kiểm tra.
    `orders` (mặt hàng → bậc) được dùng lại nếu có; nếu không, bậc được chọn bằng AIC và trả về trong
    thuộc tính `.attrs["orders"]` của kết quả để dùng cho các bộ biến khác.
    """
    exog_cols = list(exog_cols)
    train_end = pd.Timestamp(train_end)
    fallback_end = None if fallback_end is None else pd.Timestamp(fallback_end)
    groups = [(item, g[[date_col, y_col] + exog_cols]) for item, g in df.groupby(item_col, sort=False)]
    args = [(item, g, exog_cols, train_end, fallback_end, min_train, (orders or {}).get(item), candidates,
             model_cls, item_col, date_col, y_col) for item, g in groups]
    if n_jobs == 1:
        results = [_arima_item(*a) for a in args]
    else:  # pragma: no cover - song song trên Colab
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(_arima_item)(*a) for a in args)
    frames = [r[2] for r in results if r[2] is not None]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[item_col, date_col, "pred"])
    out.attrs["orders"] = {r[0]: r[1] for r in results if r[1] is not None}
    return out
