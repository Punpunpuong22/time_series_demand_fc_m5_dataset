"""Đọc dữ liệu M5 và xây dựng `Panel`: các ma trận chuỗi × ngày dùng chung cho mọi mô hình.

Quy ước chỉ số: cột j của mọi ma trận ứng với ngày d = j + 1 (d_1 ở cột 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_REQUIRED = [
    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]


@dataclass
class RawData:
    calendar: pd.DataFrame
    prices: pd.DataFrame
    sales: pd.DataFrame


def load_raw(raw_dir: str | Path) -> RawData:
    """Đọc ba tệp CSV của M5 và kiểm tra lược đồ (tên cột, kiểu, giá trị âm)."""
    raw_dir = Path(raw_dir)
    missing = [f for f in (SALES_FILE, CALENDAR_FILE, PRICES_FILE) if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu tệp {missing} trong '{raw_dir}'. Hãy chạy scripts/download_m5.sh "
            "hoặc sao chép dữ liệu M5 vào thư mục này (xem README, mục 'Dữ liệu')."
        )
    cal = pd.read_csv(raw_dir / CALENDAR_FILE)
    prices = pd.read_csv(
        raw_dir / PRICES_FILE,
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    sales = pd.read_csv(raw_dir / SALES_FILE)
    raw = RawData(cal, prices, sales)
    validate_raw(raw)
    return raw


def validate_raw(raw: RawData) -> None:
    """Kiểm tra lược đồ dữ liệu trước khi xử lý; báo lỗi dễ hiểu nếu sai."""
    cal_missing = [c for c in CAL_REQUIRED if c not in raw.calendar.columns]
    if cal_missing:
        raise ValueError(f"calendar.csv thiếu cột {cal_missing}")
    if not any(c.startswith("snap_") for c in raw.calendar.columns):
        raise ValueError("calendar.csv không có cột snap_<STATE>")
    for c in ("store_id", "item_id", "wm_yr_wk", "sell_price"):
        if c not in raw.prices.columns:
            raise ValueError(f"sell_prices.csv thiếu cột {c}")
    sales_missing = [c for c in ID_COLS if c not in raw.sales.columns]
    if sales_missing:
        raise ValueError(f"sales_train_evaluation.csv thiếu cột {sales_missing}")
    day_cols = [c for c in raw.sales.columns if c.startswith("d_")]
    expected = [f"d_{i + 1}" for i in range(len(day_cols))]
    if day_cols != expected:
        raise ValueError("Các cột ngày trong sales không liên tục từ d_1")
    if raw.sales["id"].duplicated().any():
        raise ValueError("sales có id trùng lặp")
    if (raw.sales[day_cols].to_numpy() < 0).any():
        raise ValueError("sales có giá trị âm")


@dataclass
class Panel:
    """Dữ liệu dạng ma trận cho n chuỗi.

    meta   : DataFrame n dòng, gồm ID_COLS và các cột <col>_code (mã số nguyên)
    Y      : float32 [n, D]   doanh số theo ngày (D = số ngày có doanh số, 1941)
    avail  : bool    [n, Dc]  True khi sản phẩm đã lên kệ (đã có giá) — Dc = số ngày lịch (1969)
    P      : float32 [n, Dc]  giá bán theo ngày (điền tiến trong khoảng thiếu; NaN trước khi lên kệ)
    P_gap  : bool    [n, Dc]  True nếu giá ngày đó được điền từ tuần trước (thiếu giá xen giữa)
    snap   : int8    [n, Dc]  chỉ báo SNAP của bang chứa cửa hàng
    calendar : DataFrame lịch, index = d (1..Dc)
    """

    meta: pd.DataFrame
    Y: np.ndarray
    avail: np.ndarray
    P: np.ndarray
    P_gap: np.ndarray
    snap: np.ndarray
    calendar: pd.DataFrame
    cardinalities: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.Y.shape[0]

    @property
    def n_days(self) -> int:
        return self.Y.shape[1]

    def subset(self, idx) -> "Panel":
        idx = np.asarray(idx)
        return Panel(
            meta=self.meta.iloc[idx].reset_index(drop=True),
            Y=self.Y[idx],
            avail=self.avail[idx],
            P=self.P[idx],
            P_gap=self.P_gap[idx],
            snap=self.snap[idx],
            calendar=self.calendar,
            cardinalities=self.cardinalities,
        )

    def launch_day(self) -> np.ndarray:
        """Ngày d đầu tiên sản phẩm có giá (lên kệ); n+1 nếu không bao giờ có giá."""
        has = self.avail.any(axis=1)
        first = self.avail.argmax(axis=1) + 1
        return np.where(has, first, self.avail.shape[1] + 1)


def build_panel(raw: RawData) -> Panel:
    """Chuyển dữ liệu thô thành Panel. Không có bước nào dùng thông tin tương lai."""
    sales = raw.sales.reset_index(drop=True)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    Y = sales[day_cols].to_numpy(dtype=np.float32)

    meta = sales[ID_COLS].copy()
    cardinalities = {}
    for c in CAT_COLS:
        cats = pd.Categorical(meta[c])
        meta[c + "_code"] = cats.codes.astype(np.int32)
        cardinalities[c] = len(cats.categories)

    cal = raw.calendar.copy()
    cal["d"] = cal["d"].str.slice(2).astype(int)
    cal = cal.set_index("d").sort_index()
    n_cal = int(cal.index.max())

    # ---- giá: tuần -> ngày ----
    pr = raw.prices
    pivot = pr.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price", aggfunc="mean")
    key = pd.MultiIndex.from_arrays([meta["store_id"].astype(str), meta["item_id"].astype(str)])
    pivot.index = pd.MultiIndex.from_arrays(
        [pivot.index.get_level_values(0).astype(str), pivot.index.get_level_values(1).astype(str)]
    )
    pivot = pivot.reindex(key)
    weeks = pivot.columns.to_numpy()
    day_week = cal["wm_yr_wk"].to_numpy()
    col_idx = np.searchsorted(weeks, day_week)
    valid = (col_idx < len(weeks)) & (weeks[np.clip(col_idx, 0, len(weeks) - 1)] == day_week)
    P_raw = np.full((len(meta), n_cal), np.nan, dtype=np.float32)
    P_raw[:, valid] = pivot.to_numpy(dtype=np.float32)[:, col_idx[valid]]

    has_price = ~np.isnan(P_raw)
    launched = np.maximum.accumulate(has_price, axis=1)  # True từ ngày có giá đầu tiên trở đi
    P_ff = pd.DataFrame(P_raw).ffill(axis=1).to_numpy(dtype=np.float32)
    P_gap = launched & ~has_price
    P = np.where(launched, P_ff, np.nan).astype(np.float32)

    # ---- SNAP theo bang của cửa hàng ----
    snap_cols = {c.split("_", 1)[1]: c for c in cal.columns if c.startswith("snap_")}
    snap = np.zeros((len(meta), n_cal), dtype=np.int8)
    for st, col in snap_cols.items():
        rows = (meta["state_id"].astype(str) == st).to_numpy()
        snap[rows] = cal[col].to_numpy(dtype=np.int8)[None, :]

    return Panel(
        meta=meta, Y=Y, avail=launched, P=P, P_gap=P_gap, snap=snap, calendar=cal, cardinalities=cardinalities
    )
