"""Chia dữ liệu theo nhiều mốc dự báo (rolling-origin), Bảng 10 trong báo cáo.

Mỗi cửa sổ có mốc `origin` = ngày cuối cùng đã biết doanh số; dự báo cho
origin + 1 ... origin + horizon. Mô hình được huấn luyện lại ở các cửa sổ có
retrain=True; ở các cửa sổ khác, mô hình cũ được dùng nhưng vẫn nhận doanh số
mới nhất đến origin làm đầu vào.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import LeakageError


@dataclass(frozen=True)
class Window:
    name: str
    role: str          # "tune" | "eval" | "test"
    origin: int        # ngày d cuối cùng đã biết
    start: int         # origin + 1
    end: int           # origin + horizon
    retrain: bool

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.start, self.end + 1)


def make_windows(
    horizon: int = 28,
    first_tune_start: int = 1522,
    n_tune: int = 2,
    first_eval_start: int = 1578,
    n_eval: int = 13,
    retrain_every: int = 3,
    max_day: int = 1941,
) -> list[Window]:
    windows: list[Window] = []
    for k in range(n_tune):
        s = first_tune_start + k * horizon
        windows.append(Window(f"tune{k + 1:02d}", "tune", s - 1, s, s + horizon - 1, retrain=(k == 0)))
    for k in range(n_eval):
        s = first_eval_start + k * horizon
        role = "test" if k == n_eval - 1 else "eval"
        windows.append(Window(f"eval{k + 1:02d}", role, s - 1, s, s + horizon - 1, retrain=(k % retrain_every == 0)))
    validate_windows(windows, max_day)
    return windows


def windows_from_config(cfg_windows: dict, max_day: int = 1941) -> list[Window]:
    keys = ("horizon", "first_tune_start", "n_tune", "first_eval_start", "n_eval", "retrain_every")
    return make_windows(**{k: cfg_windows[k] for k in keys if k in cfg_windows}, max_day=max_day)


def validate_windows(windows: list[Window], max_day: int = 1941) -> None:
    """Không chồng lấn, theo thứ tự thời gian, tinh chỉnh trước đánh giá, không vượt dữ liệu."""
    for a, b in zip(windows, windows[1:]):
        if b.start <= a.end:
            raise ValueError(f"Cửa sổ {a.name} và {b.name} chồng lấn")
    if windows and windows[-1].end > max_day:
        raise ValueError(f"Cửa sổ cuối kết thúc ở d_{windows[-1].end} > d_{max_day}")
    tune_end = max((w.end for w in windows if w.role == "tune"), default=0)
    eval_start = min((w.start for w in windows if w.role != "tune"), default=10**9)
    if tune_end >= eval_start:
        raise ValueError("Cửa sổ tinh chỉnh phải nằm hoàn toàn trước cửa sổ đánh giá")
    for w in windows:
        if w.end < w.start:
            raise ValueError(f"Cửa sổ {w.name} rỗng")


def training_cutoff(windows: list[Window], w: Window) -> int:
    """Mốc huấn luyện của mô hình được dùng cho cửa sổ w (lần retrain gần nhất trong cùng giai đoạn)."""
    same_phase = [x for x in windows if (x.role == "tune") == (w.role == "tune") and x.start <= w.start]
    retrains = [x for x in same_phase if x.retrain]
    if not retrains:
        raise ValueError(f"Không có cửa sổ retrain trước {w.name}")
    return retrains[-1].origin


def assert_train_before(cutoff: int, target_days) -> None:
    """Mọi ngày mục tiêu dùng để huấn luyện phải ≤ cutoff."""
    if np.max(target_days) > cutoff:
        raise LeakageError(f"Dòng huấn luyện có ngày mục tiêu d_{int(np.max(target_days))} > mốc d_{cutoff}")
