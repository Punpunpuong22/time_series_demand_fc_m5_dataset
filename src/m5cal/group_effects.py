"""Tác động của từng nhóm yếu tố lịch lên sai số của từng mô hình.

Cột dự báo trong khung đánh giá đặt tên "<mô hình>-<bộ>", ví dụ "LightGBM-S2".
Hai loại tác động:
- tăng thêm (incremental): thêm nhóm theo thứ tự G1 → G2 → G3 (S0→S1, S1→S2, S2→S3);
- biên (marginal): bỏ đúng một nhóm khỏi bộ đầy đủ S3 (D1→S3 cho G1, D2→S3 cho G2, S2→S3 cho G3).
Tác động được tính trên mọi ngày và trên "ngày liên quan" của nhóm (G2: ngày lễ, sự kiện và trước sự kiện;
G3: ngày SNAP; G1: mọi ngày).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .supplementary import dm_test_daily, wape

COMPARISONS = [("G1", "tăng thêm", "S0", "S1"), ("G2", "tăng thêm", "S1", "S2"), ("G3", "tăng thêm", "S2", "S3"),
               ("G1", "biên", "D1", "S3"), ("G2", "biên", "D2", "S3"), ("G3", "biên", "S2", "S3")]


def relevant_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "G2":
        return df["date_type"].isin(["Holiday", "Other_Event", "Pre_Event"])
    if group == "G3":
        return df["snap"] == 1
    return pd.Series(True, index=df.index)


def group_effects(df: pd.DataFrame, models, y_col: str = "demand", lag: int = 7) -> pd.DataFrame:
    rows = []
    for m in models:
        for g, kind, before, after in COMPARISONS:
            cb, ca = f"{m}-{before}", f"{m}-{after}"
            if cb not in df.columns or ca not in df.columns:
                continue
            for scope, mask in (("mọi ngày", pd.Series(True, index=df.index)), ("ngày liên quan", relevant_mask(df, g))):
                if scope == "ngày liên quan" and g == "G1":
                    continue
                sub = df[mask]
                wb, wa = 100 * wape(sub[y_col], sub[cb]), 100 * wape(sub[y_col], sub[ca])
                s_abs, p_abs = dm_test_daily(sub, cb, ca, "abs", lag)
                s_sq, p_sq = dm_test_daily(sub, cb, ca, "sq", lag)
                rows.append({"model": m, "group": g, "type": kind, "scope": scope, "before": before, "after": after,
                             "days": int(sub["date"].nunique()), "WAPE before (%)": round(wb, 2),
                             "WAPE after (%)": round(wa, 2), "delta (pp)": round(wa - wb, 2),
                             "delta (%)": round(100 * (wa / wb - 1), 2) if wb > 0 else np.nan,
                             "DM abs": round(s_abs, 3), "p abs": round(p_abs, 4),
                             "DM sq": round(s_sq, 3), "p sq": round(p_sq, 4),
                             "helps (5%)": bool(s_abs > 0 and p_abs < 0.05)})
    return pd.DataFrame(rows)


def effect_matrix(effects: pd.DataFrame, kind: str = "biên", scope: str = "mọi ngày",
                  value: str = "delta (%)") -> pd.DataFrame:
    """Bảng mô hình × nhóm (giá trị âm = nhóm làm giảm sai số)."""
    sub = effects[(effects["type"] == kind) & (effects["scope"] == scope)]
    return sub.pivot(index="model", columns="group", values=value)
