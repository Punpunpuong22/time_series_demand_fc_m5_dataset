"""Sinh bộ dữ liệu GIẢ LẬP có đúng định dạng M5.

Mục đích duy nhất: kiểm thử pipeline và chạy thử nhanh khi chưa tải dữ liệu thật.
Kết quả trên dữ liệu giả lập KHÔNG phải kết quả nghiên cứu và không được báo cáo
như kết quả thực nghiệm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEEKDAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SNAP_DAYS = {
    "CA": set(range(1, 11)),
    "TX": {1, 3, 5, 6, 7, 9, 11, 12, 13, 15},
    "WI": {2, 3, 5, 6, 8, 9, 11, 12, 14, 15},
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """weekday theo pandas: Monday=0 ... Sunday=6."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _events(years, rng) -> dict:
    """Danh sách sự kiện giả lập: một số cố định ngày, một số dịch chuyển hằng năm."""
    ev = {}
    for y in years:
        ev[pd.Timestamp(year=y, month=12, day=25)] = ("Christmas", "National")
        ev[pd.Timestamp(year=y, month=7, day=4)] = ("IndependenceDay", "National")
        ev[_nth_weekday(y, 11, 3, 4)] = ("Thanksgiving", "National")
        ev[_nth_weekday(y, 5, 6, 2)] = ("MothersDay", "Cultural")
        ev[_nth_weekday(y, 2, 6, 1)] = ("SuperBowl", "Sporting")
        # sự kiện tôn giáo dịch chuyển: ngày ngẫu nhiên trong tháng 3–4
        moving = pd.Timestamp(year=y, month=3, day=20) + pd.Timedelta(days=int(rng.integers(0, 30)))
        ev[moving] = ("Easter", "Religious")
    return ev


def make_synthetic_m5(
    out_dir: str | Path,
    n_items_per_dept: int = 4,
    depts: tuple = ("FOODS_1", "FOODS_2", "HOBBIES_1"),
    states: tuple = ("CA", "TX", "WI"),
    stores_per_state: int = 1,
    n_days: int = 1941,
    n_cal_days: int = 1969,
    seed: int = 0,
) -> Path:
    """Ghi calendar.csv, sell_prices.csv, sales_train_evaluation.csv vào out_dir."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2011-01-29", periods=n_cal_days, freq="D")
    day_idx = np.arange(n_cal_days)
    events = _events(range(2011, 2017), rng)

    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + day_idx // 7,
            "weekday": [WEEKDAYS[i % 7] for i in day_idx],
            "wday": day_idx % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in day_idx],
        }
    )
    cal["event_name_1"] = [events.get(ts, (None, None))[0] for ts in dates]
    cal["event_type_1"] = [events.get(ts, (None, None))[1] for ts in dates]
    cal["event_name_2"] = None
    cal["event_type_2"] = None
    for st in ("CA", "TX", "WI"):
        cal[f"snap_{st}"] = [int(ts.day in SNAP_DAYS[st]) for ts in dates]

    # hiệu ứng ngày: tuần, sự kiện, ngày trước sự kiện
    weekly = np.array([1.35, 1.30, 0.90, 0.85, 0.85, 0.90, 1.05])[day_idx % 7]
    ev_mult = np.ones(n_cal_days)
    type_effect = {"National": 1.6, "Cultural": 1.3, "Sporting": 1.4, "Religious": 1.5}
    for i, ts in enumerate(dates):
        if ts in events:
            ev_mult[i] *= type_effect[events[ts][1]]
            for k in (1, 2, 3):
                if i - k >= 0:
                    ev_mult[i - k] *= 1.15
        if ts.month == 12 and ts.day == 25:
            ev_mult[i] = 0.0  # cửa hàng đóng cửa ngày Giáng sinh (giả lập)

    rows, price_rows = [], []
    stores = [f"{st}_{k + 1}" for st in states for k in range(stores_per_state)]
    n_weeks = int(cal["wm_yr_wk"].max() - 11101 + 1)
    for dept in depts:
        cat = dept.split("_")[0]
        for j in range(n_items_per_dept):
            item = f"{dept}_{j + 1:03d}"
            base_level = rng.gamma(shape=1.2, scale=2.0)  # nhiều chuỗi có mức thấp -> gián đoạn
            launch_week = int(rng.integers(0, 60)) if rng.random() < 0.5 else 0
            base_price = round(float(rng.uniform(1, 15)), 2)
            for store in stores:
                st = store.split("_")[0]
                snap = cal[f"snap_{st}"].to_numpy()
                snap_mult = 1.0 + (0.25 * snap if cat == "FOODS" else 0.0)
                # giá theo tuần, thỉnh thoảng giảm giá
                week_price = np.full(n_weeks, base_price)
                promo = rng.random(n_weeks) < 0.08
                week_price[promo] *= 0.8
                price_day = week_price[day_idx // 7]
                price_mult = (price_day / base_price) ** -1.5
                trend = 1.0 + 0.0002 * day_idx
                lam = base_level * weekly * ev_mult * snap_mult * price_mult * trend
                lam *= rng.uniform(0.7, 1.3)
                y = rng.poisson(lam[:n_days]).astype(int)
                y[: launch_week * 7] = 0
                sid = f"{item}_{store}_evaluation"
                rows.append([sid, item, dept, cat, store, st] + y.tolist())
                for w in range(launch_week, n_weeks):
                    price_rows.append((store, item, 11101 + w, round(float(week_price[w]), 2)))

    day_cols = [f"d_{i + 1}" for i in range(n_days)]
    sales = pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + day_cols)
    prices = pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])

    cal.to_csv(out_dir / "calendar.csv", index=False)
    prices.to_csv(out_dir / "sell_prices.csv", index=False)
    sales.to_csv(out_dir / "sales_train_evaluation.csv", index=False)
    (out_dir / "SYNTHETIC_DATA_README.txt").write_text(
        "Day la du lieu GIA LAP de kiem thu pipeline. Khong phai du lieu M5 that.\n", encoding="utf-8"
    )
    return out_dir
