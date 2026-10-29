"""Baseline: Seasonal Naive (B0) và Croston–SBA (B1). Không cần huấn luyện."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def _frame(panel, window, pred: np.ndarray) -> pd.DataFrame:
    days = window.days
    n = panel.n
    pred = np.where(panel.avail[:, days - 1], pred, 0.0)
    return pd.DataFrame({
        "sid": np.repeat(np.arange(n), len(days)),
        "d": np.tile(days, n),
        "y_pred": np.clip(pred, 0, None).ravel().astype(np.float32),
    })


def seasonal_naive_matrix(Y: np.ndarray, origin: int, horizon: int = 28) -> np.ndarray:
    """ŷ(origin + h) = y(origin − 7 + ((h − 1) mod 7) + 1): lặp lại tuần cuối cùng đã quan sát."""
    last_week = Y[:, origin - 7: origin]            # ngày origin−6 ... origin
    reps = int(np.ceil(horizon / 7))
    return np.tile(last_week, (1, reps))[:, :horizon]


def seasonal_naive(panel, window) -> pd.DataFrame:
    pred = seasonal_naive_matrix(panel.Y, window.origin, len(window.days))
    return _frame(panel, window, pred)


def _ses(x: np.ndarray, alpha: float) -> float:
    """Làm trơn hàm mũ đơn giản, khởi tạo bằng giá trị đầu; trả về mức cuối cùng."""
    if len(x) == 0:
        return np.nan
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x[1:], zi=[(1.0 - alpha) * x[0]])
    return float(y[-1]) if len(y) else float(x[0])


def croston_sba_level(y: np.ndarray, alpha: float = 0.1) -> float:
    """Dự báo Croston có hiệu chỉnh Syntetos–Boylan: (1 − α/2) · z/p."""
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    sizes = y[nz].astype(float)
    intervals = np.diff(np.concatenate([[-1], nz])).astype(float)
    intervals[0] = nz[0] + 1
    z = _ses(sizes, alpha)
    p = _ses(intervals, alpha)
    return (1 - alpha / 2) * z / max(p, 1e-9)


def croston_sba(panel, window, alpha: float = 0.1) -> pd.DataFrame:
    launch = panel.launch_day()
    levels = np.empty(panel.n)
    for i in range(panel.n):
        start = max(int(launch[i]), 1) - 1
        levels[i] = croston_sba_level(panel.Y[i, start: window.origin], alpha)
    pred = np.repeat(levels[:, None], len(window.days), axis=1)
    return _frame(panel, window, pred)
