"""Đặc trưng lịch và phân loại ngày (Bảng 9 trong báo cáo).

Hiệu ứng lịch ĐỀU ĐẶN: thứ trong tuần, cuối tuần (lặp chu kỳ 7 ngày).
Hiệu ứng lịch BẤT THƯỜNG: sự kiện (dịch chuyển giữa các năm), ngày trước sự kiện,
SNAP theo bang. Toàn bộ thông tin ở đây là thông tin lịch biết trước.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]
WEEKEND = {"Saturday", "Sunday"}
# Thứ tự ưu tiên khi gán MỘT nhãn chính cho mỗi ngày
PRIMARY_ORDER = ["event_<Type>", "pre_event", "snap", "weekend", "weekday"]
REGULAR_LABELS = {"weekday", "weekend"}


def _distance_to_event(has_event: np.ndarray, cap: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Số ngày tới sự kiện kế tiếp và số ngày kể từ sự kiện trước (0 nếu hôm nay có sự kiện)."""
    n = len(has_event)
    pos = np.flatnonzero(has_event)
    idx = np.arange(n)
    if len(pos) == 0:
        return np.full(n, cap, np.float32), np.full(n, cap, np.float32)
    nxt = np.searchsorted(pos, idx, side="left")
    to_next = np.where(nxt < len(pos), pos[np.clip(nxt, 0, len(pos) - 1)] - idx, cap)
    prv = np.searchsorted(pos, idx, side="right") - 1
    since = np.where(prv >= 0, idx - pos[np.clip(prv, 0, len(pos) - 1)], cap)
    return np.minimum(to_next, cap).astype(np.float32), np.minimum(since, cap).astype(np.float32)


def calendar_features(calendar: pd.DataFrame, pre_event_days: int = 3) -> pd.DataFrame:
    """Đặc trưng lịch cấp ngày (không phụ thuộc bang). Index = d."""
    cal = calendar
    out = pd.DataFrame(index=cal.index)
    weekday = cal["weekday"].astype(str)
    out["wday"] = cal["wday"].astype(np.int8)                    # M5: 1 = thứ Bảy ... 7 = thứ Sáu
    out["is_weekend"] = weekday.isin(WEEKEND).astype(np.int8)
    out["month"] = cal["month"].astype(np.int8)
    out["month_sin"] = np.sin(2 * np.pi * (cal["month"] - 1) / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * (cal["month"] - 1) / 12).astype(np.float32)

    t1 = cal["event_type_1"].fillna("").astype(str)
    t2 = cal["event_type_2"].fillna("").astype(str)
    for et in EVENT_TYPES:
        out[f"ev_{et}"] = ((t1 == et) | (t2 == et)).astype(np.int8)
    has_event = ((t1 != "") | (t2 != "")).to_numpy()
    out["any_event"] = has_event.astype(np.int8)

    names = cal["event_name_1"].fillna("none").astype(str)
    out["event_name_code"] = pd.Categorical(names).codes.astype(np.int16)

    to_next, since = _distance_to_event(has_event)
    out["days_to_next_event"] = to_next
    out["days_since_event"] = since
    out["pre_event"] = ((to_next >= 1) & (to_next <= pre_event_days)).astype(np.int8)
    out["event_type_primary"] = t1.where(t1 != "", t2).replace("", "none")
    return out


def primary_labels(calf: pd.DataFrame, snap_row: np.ndarray) -> np.ndarray:
    """Nhãn chính cho một chuỗi ngày (một bang): ưu tiên sự kiện > trước sự kiện > SNAP > cuối tuần > ngày thường."""
    n = len(calf)
    lab = np.where(calf["is_weekend"].to_numpy() == 1, "weekend", "weekday").astype(object)
    lab = np.where(snap_row[:n] == 1, "snap", lab)
    lab = np.where(calf["pre_event"].to_numpy() == 1, "pre_event", lab)
    # Ngày có sự kiện: dùng event_type_1 (hoặc event_type_2 nếu ô thứ nhất trống)
    ev_type = calf["event_type_primary"].to_numpy()
    lab = np.where(ev_type != "none", np.char.add("event_", ev_type.astype(str)), lab)
    return lab.astype(str)


def daytype_matrix(panel, calf: pd.DataFrame, days: np.ndarray) -> np.ndarray:
    """Nhãn chính [n_series, len(days)] (phụ thuộc bang vì SNAP)."""
    days = np.asarray(days)
    sub = calf.loc[days]
    out = np.empty((panel.n, len(days)), dtype=object)
    # tính theo từng mẫu SNAP duy nhất (mỗi bang một mẫu) cho nhanh
    snap_days = panel.snap[:, days - 1]
    uniq, inv = np.unique(snap_days, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for k, row in enumerate(uniq):
        out[inv == k] = primary_labels(sub, row)[None, :]
    return out.astype(str)


def daytype_flags(panel, calf: pd.DataFrame, days: np.ndarray) -> dict:
    """Nhãn KHÔNG loại trừ: mỗi khóa là một ma trận bool [n_series, len(days)]."""
    days = np.asarray(days)
    sub = calf.loc[days]
    n = panel.n
    flags = {
        "weekend": np.repeat(sub["is_weekend"].to_numpy()[None, :] == 1, n, axis=0),
        "any_event": np.repeat(sub["any_event"].to_numpy()[None, :] == 1, n, axis=0),
        "pre_event": np.repeat(sub["pre_event"].to_numpy()[None, :] == 1, n, axis=0),
        "snap": panel.snap[:, days - 1] == 1,
    }
    for et in EVENT_TYPES:
        flags[f"event_{et}"] = np.repeat(sub[f"ev_{et}"].to_numpy()[None, :] == 1, n, axis=0)
    return flags


def regularity(label: str) -> str:
    """'regular' cho ngày thường/cuối tuần, 'irregular' cho sự kiện, trước sự kiện, SNAP."""
    return "regular" if label in REGULAR_LABELS else "irregular"
