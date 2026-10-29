"""Ba nhóm yếu tố lịch dùng trong thiết kế mới (phiên bản 0.3.0).

Ngày tháng là một biến liên tục, tăng mãi theo thời gian: đưa thẳng nó vào mô hình thì mô hình chỉ thấy
"thời điểm thứ bao nhiêu", không thấy các chu kỳ lặp lại hay ngữ cảnh của ngày. Vì vậy thông tin ngày tháng
được phân rã và gắn ngữ cảnh thành ba nhóm thuộc tính:

- G1 · Nội sinh (phái sinh từ ngày): thứ, ngày trong tháng, tuần trong năm, tháng, mùa, cuối tuần.
- G2 · Ngoại sinh (sự kiện, lễ hội): loại sự kiện, ngày lễ, trước sự kiện, khoảng cách tới / từ sự kiện.
- G3 · Chính sách trợ cấp (SNAP): ngày phát SNAP và vị trí của ngày trong đợt phát.

Mỗi nhóm có ba cách biểu diễn, tùy mô hình:
- "tree"   : giá trị số nguyên / nhị phân cho LightGBM.
- "linear" : biến giả (one-hot) cho ARIMA có biến ngoại sinh (mô hình tuyến tính).
- "neural" : mã hóa sin–cos và biến đã chuẩn hóa cho LSTM và N-BEATS.

Bộ đặc trưng (cấu hình ablation):
- S0 = chỉ lịch sử nhu cầu
- S1 = S0 + G1, S2 = S1 + G2, S3 = S2 + G3 (lồng nhau, đo tác động tăng thêm)
- D1 = S3 − G1, D2 = S3 − G2 (bỏ một nhóm khỏi bộ đầy đủ, đo tác động biên; S3 − G3 chính là S2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_NAMES = {"G1": "Nội sinh (thứ, ngày, tháng, mùa)",
               "G2": "Ngoại sinh (sự kiện, lễ hội)",
               "G3": "Chính sách trợ cấp (SNAP)"}
EVENT_TYPES = ["National", "Religious", "Cultural", "Sporting"]
HOLIDAY_TYPES = ["National", "Religious"]
DATE_TYPES = ["Weekday", "Weekend", "Pre_Event", "Other_Event", "Holiday"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]          # bắc bán cầu: 12-2, 3-5, 6-8, 9-11

SET_GROUPS = {"S0": [], "S1": ["G1"], "S2": ["G1", "G2"], "S3": ["G1", "G2", "G3"],
              "D1": ["G2", "G3"], "D2": ["G1", "G3"]}
NESTED = ["S0", "S1", "S2", "S3"]


def _cyc(x, period):
    a = 2 * np.pi * np.asarray(x, float) / period
    return np.sin(a), np.cos(a)


def build_calendar_groups(calendar: pd.DataFrame, state: str = "CA", event_window: int = 14,
                          pre_event_days: tuple[int, int] = (1, 2)) -> pd.DataFrame:
    """Từ calendar.csv của M5, tạo mọi thuộc tính của ba nhóm (mỗi dòng một ngày).

    Chỉ dùng thông tin có trong lịch của chính ngày đó và các ngày sự kiện đã công bố, nên mọi thuộc tính
    đều biết trước tại thời điểm dự báo.
    """
    c = calendar.copy()
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")
    c = c.sort_values("date").reset_index(drop=True)

    # ---------------- G1: phân rã ngày tháng
    c["day_of_week"] = c["date"].dt.dayofweek                       # 0 = thứ Hai
    c["day_of_month"] = c["date"].dt.day
    c["week_of_year"] = c["date"].dt.isocalendar().week.astype(int)
    c["month"] = c["date"].dt.month
    c["season"] = ((c["month"] % 12) // 3).astype(int)              # 0 Đông, 1 Xuân, 2 Hạ, 3 Thu
    c["is_weekend"] = (c["day_of_week"] >= 5).astype(int)
    c["dow_sin"], c["dow_cos"] = _cyc(c["day_of_week"], 7)
    c["dom_sin"], c["dom_cos"] = _cyc(c["day_of_month"] - 1, 31)
    c["woy_sin"], c["woy_cos"] = _cyc(c["week_of_year"] - 1, 53)
    c["month_sin"], c["month_cos"] = _cyc(c["month"] - 1, 12)
    for k in range(1, 7):
        c[f"dow_{k}"] = (c["day_of_week"] == k).astype(int)          # thứ Hai làm mốc
    for k in range(2, 13):
        c[f"month_{k}"] = (c["month"] == k).astype(int)             # tháng 1 làm mốc
    for k, s in enumerate(SEASONS):
        c[f"season_{s}"] = (c["season"] == k).astype(int)

    # ---------------- G2: sự kiện, lễ hội
    t1 = c.get("event_type_1", pd.Series(index=c.index, dtype=object))
    t2 = c.get("event_type_2", pd.Series(index=c.index, dtype=object))
    for et in EVENT_TYPES:
        c[f"ev_{et}"] = ((t1 == et) | (t2 == et)).astype(int)
    c["has_event"] = (t1.notna() | t2.notna()).astype(int)
    c["is_holiday"] = (t1.isin(HOLIDAY_TYPES) | t2.isin(HOLIDAY_TYPES)).astype(int)
    ev_date = c["date"].where(c["has_event"] == 1)
    c["days_to_next_event"] = (ev_date.bfill() - c["date"]).dt.days.clip(upper=event_window).fillna(event_window)
    c["days_since_last_event"] = (c["date"] - ev_date.ffill()).dt.days.clip(upper=event_window).fillna(event_window)
    lo, hi = pre_event_days
    c["is_pre_event"] = c["days_to_next_event"].between(lo, hi).astype(int)
    c["dts"] = c["days_to_next_event"] / event_window
    c["dsl"] = c["days_since_last_event"] / event_window

    # ---------------- G3: chính sách trợ cấp SNAP của bang
    col = f"snap_{state}"
    c["snap"] = c[col].astype(int) if col in c else 0
    run = c["snap"].groupby((c["snap"] != c["snap"].shift()).cumsum()).cumsum()
    c["snap_run_day"] = np.where(c["snap"] == 1, run, 0).astype(int)  # ngày thứ mấy của đợt phát SNAP
    c["snap_run_scaled"] = c["snap_run_day"] / max(int(c["snap_run_day"].max()), 1)

    # ---------------- nhãn loại ngày: CHỈ dùng để phân tầng đánh giá, không làm đặc trưng
    c["date_type"] = np.select(
        [c["is_holiday"] == 1, c["has_event"] == 1, c["is_pre_event"] == 1, c["is_weekend"] == 1],
        ["Holiday", "Other_Event", "Pre_Event", "Weekend"], default="Weekday")
    return c


# Danh sách cột của từng nhóm, theo cách biểu diễn
GROUP_COLUMNS = {
    "tree": {
        "G1": ["day_of_week", "day_of_month", "week_of_year", "month", "season", "is_weekend"],
        "G2": [f"ev_{e}" for e in EVENT_TYPES] + ["has_event", "is_holiday", "is_pre_event",
                                                  "days_to_next_event", "days_since_last_event"],
        "G3": ["snap", "snap_run_day"],
    },
    "linear": {
        "G1": [f"dow_{k}" for k in range(1, 7)] + [f"month_{k}" for k in range(2, 13)],
        "G2": [f"ev_{e}" for e in EVENT_TYPES] + ["is_pre_event"],
        "G3": ["snap", "snap_run_scaled"],
    },
    "neural": {
        "G1": ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "woy_sin", "woy_cos", "month_sin", "month_cos",
               "is_weekend"] + [f"season_{s}" for s in SEASONS],
        "G2": [f"ev_{e}" for e in EVENT_TYPES] + ["has_event", "is_holiday", "is_pre_event", "dts", "dsl"],
        "G3": ["snap", "snap_run_scaled"],
    },
}


def set_columns(set_name: str, kind: str) -> list[str]:
    """Danh sách cột lịch của một bộ đặc trưng (không gồm biến lịch sử nhu cầu)."""
    if set_name not in SET_GROUPS:
        raise KeyError(f"Bộ không tồn tại: {set_name}; có {list(SET_GROUPS)}")
    cols: list[str] = []
    for g in SET_GROUPS[set_name]:
        cols += GROUP_COLUMNS[kind][g]
    return cols


def all_calendar_columns() -> list[str]:
    cols = []
    for kind in GROUP_COLUMNS.values():
        for g in kind.values():
            cols += g
    return list(dict.fromkeys(cols)) + ["date_type", "season", "snap_run_day"]
