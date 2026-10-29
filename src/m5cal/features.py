"""Xây dựng đặc trưng cho LightGBM (dạng bảng) và cửa sổ chuỗi cho LSTM.

NGUYÊN TẮC CHỐNG RÒ RỈ (mục 7.1 của báo cáo)
1. Tại mốc `cutoff`, doanh số sau cutoff bị che (NaN) TRƯỚC khi tính bất kỳ đặc trưng nào.
2. LightGBM trực tiếp: một mô hình cho mỗi nhóm tầm dự báo [a, b]; độ trễ tối thiểu L = b
   nên mọi giá trị trễ của ngày mục tiêu t = cutoff + h (h ≤ b) đều ≤ cutoff.
3. Chuẩn hóa giá chỉ dùng giá đến cutoff; chuẩn hóa chuỗi của LSTM chỉ dùng cửa sổ quá khứ.
4. Lịch, SNAP và giá kế hoạch là biến biết trước (giả định được nêu trong báo cáo);
   price_mode="last_known" tắt giả định giá biết trước để phân tích độ nhạy.
Các nguyên tắc này được kiểm tra tự động trong tests/test_leakage.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CAT_COLS
from .daytypes import EVENT_TYPES
from .utils import LeakageError

HORIZON_GROUPS = {
    "weekly": [(1, 7), (8, 14), (15, 21), (22, 28)],
    "lag28": [(1, 28)],
    "daily": [(h, h) for h in range(1, 29)],
}

FEATURE_SETS = {
    "H": ["id", "lag", "roll"],
    "HC": ["id", "lag", "roll", "cal_regular", "cal_irregular", "snap"],
    "HCP": ["id", "lag", "roll", "cal_regular", "cal_irregular", "snap", "price"],
}

NON_FEATURE_COLS = ("sid", "d", "y", "avail")
LAG_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
ROLL_WINDOWS = (7, 28, 56)
CAL_REGULAR_COLS = ["wday", "is_weekend", "month"]
CAL_IRREGULAR_COLS = [f"ev_{e}" for e in EVENT_TYPES] + [
    "any_event", "event_name_code", "days_to_next_event", "days_since_event", "pre_event",
]
CATEGORICAL_FEATURES = list(CAT_COLS) + ["event_name_code"]


# ----------------------------------------------------------------------------- helpers
def masked_sales(panel, cutoff: int) -> np.ndarray:
    """Doanh số với mọi ngày > cutoff và mọi ngày trước khi lên kệ đặt thành NaN."""
    Y = panel.Y.astype(np.float32, copy=True)
    Y[:, cutoff:] = np.nan
    Y[~panel.avail[:, : Y.shape[1]]] = np.nan
    return Y


def price_reference(panel, cutoff: int) -> np.ndarray:
    """Giá trung bình của từng chuỗi, CHỈ tính trên các ngày ≤ cutoff."""
    P = panel.P[:, :cutoff]
    with np.errstate(all="ignore"):
        ref = np.nanmean(np.where(np.isnan(P), np.nan, P), axis=1)
    return ref.astype(np.float32)


def _pad_left(M: np.ndarray, pad: int) -> np.ndarray:
    return np.concatenate([np.full((M.shape[0], pad), np.nan, dtype=M.dtype), M], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def feature_group_of(col: str) -> str:
    if col in CAT_COLS:
        return "id"
    if col.startswith("lag_"):
        return "lag"
    if col.startswith("roll_") or col.startswith("zero_share"):
        return "roll"
    if col in CAL_REGULAR_COLS:
        return "cal_regular"
    if col in CAL_IRREGULAR_COLS:
        return "cal_irregular"
    if col == "snap":
        return "snap"
    if col.startswith("price"):
        return "price"
    raise KeyError(col)


# ----------------------------------------------------------------------------- LightGBM
def lgbm_frame(
    panel,
    calf: pd.DataFrame,
    cutoff: int,
    min_lag: int,
    target_days,
    feature_set: str,
    price_mode: str = "known",
) -> pd.DataFrame:
    """Một dòng cho mỗi (chuỗi, ngày mục tiêu). Đặc trưng chỉ dùng doanh số ≤ cutoff và ≤ d − min_lag.

    Dùng cho cả dòng huấn luyện (target_days ≤ cutoff) và dòng dự báo (target_days > cutoff).
    Cột 'y' là nhãn (chỉ dùng để huấn luyện/đánh giá, không bao giờ là đặc trưng).
    """
    target_days = np.asarray(target_days, dtype=int)
    if target_days.max() - min_lag > cutoff:
        raise LeakageError(
            f"Ngày mục tiêu d_{target_days.max()} với lag tối thiểu {min_lag} cần doanh số sau mốc d_{cutoff}"
        )
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"feature_set phải là một trong {list(FEATURE_SETS)}")
    groups = FEATURE_SETS[feature_set]
    n, T = panel.n, len(target_days)

    Ym = masked_sales(panel, cutoff)
    pad = min_lag + max(LAG_OFFSETS) + max(ROLL_WINDOWS) + 8
    Yp = _pad_left(Ym, pad)
    base = target_days - 1 + pad          # cột của ngày mục tiêu trong ma trận đã đệm

    cols: dict[str, np.ndarray] = {
        "sid": np.repeat(np.arange(n, dtype=np.int32), T),
        "d": np.tile(target_days.astype(np.int32), n),
    }
    if "id" in groups:
        for c in CAT_COLS:
            cols[c] = np.repeat(panel.meta[c + "_code"].to_numpy(np.int32), T)
    if "lag" in groups:
        for o in LAG_OFFSETS:
            cols[f"lag_p{o}"] = Yp[:, base - (min_lag + o)].ravel()
    if "roll" in groups:
        Z = np.nan_to_num(Yp).astype(np.float64)
        C = (~np.isnan(Yp)).astype(np.float64)
        zero = ((Yp == 0) & ~np.isnan(Yp)).astype(np.float64)
        cz = np.concatenate([np.zeros((n, 1)), np.cumsum(Z, axis=1)], axis=1)
        cz2 = np.concatenate([np.zeros((n, 1)), np.cumsum(Z * Z, axis=1)], axis=1)
        cc = np.concatenate([np.zeros((n, 1)), np.cumsum(C, axis=1)], axis=1)
        c0 = np.concatenate([np.zeros((n, 1)), np.cumsum(zero, axis=1)], axis=1)
        end = base - min_lag + 1          # tổng trên các cột (end - w, end]
        for w in ROLL_WINDOWS:
            s = cz[:, end] - cz[:, end - w]
            s2 = cz2[:, end] - cz2[:, end - w]
            k = cc[:, end] - cc[:, end - w]
            with np.errstate(all="ignore"):
                mean = np.where(k > 0, s / k, np.nan)
                var = np.where(k > 1, (s2 - k * mean * mean) / np.maximum(k - 1, 1), np.nan)
            cols[f"roll_mean_{w}"] = mean.ravel().astype(np.float32)
            if w in (7, 28):
                cols[f"roll_std_{w}"] = np.sqrt(np.clip(var, 0, None)).ravel().astype(np.float32)
        k28 = cc[:, end] - cc[:, end - 28]
        with np.errstate(all="ignore"):
            cols["zero_share_28"] = np.where(k28 > 0, (c0[:, end] - c0[:, end - 28]) / k28, np.nan).ravel()

    sub = calf.loc[target_days]
    if "cal_regular" in groups:
        for c in CAL_REGULAR_COLS:
            cols[c] = np.tile(sub[c].to_numpy(), n)
    if "cal_irregular" in groups:
        for c in CAL_IRREGULAR_COLS:
            cols[c] = np.tile(sub[c].to_numpy(), n)
    if "snap" in groups:
        cols["snap"] = panel.snap[:, target_days - 1].ravel()
    if "price" in groups:
        if price_mode == "known":
            pdays = target_days
        elif price_mode == "last_known":
            pdays = np.minimum(target_days, cutoff)
        else:
            raise ValueError("price_mode phải là 'known' hoặc 'last_known'")
        Pd = panel.P[:, pdays - 1]
        Pprev = panel.P[:, np.clip(pdays - 1 - 7, 0, None)]
        ref = price_reference(panel, cutoff)[:, None]
        with np.errstate(all="ignore"):
            cols["price_rel"] = (Pd / ref).ravel().astype(np.float32)
            cols["price_change_7"] = (Pd / Pprev - 1.0).ravel().astype(np.float32)
        cols["price_gap"] = panel.P_gap[:, pdays - 1].ravel().astype(np.int8)

    cols["y"] = panel.Y[:, np.clip(target_days - 1, 0, panel.n_days - 1)].ravel()
    if target_days.max() > panel.n_days:
        cols["y"] = np.where(np.tile(target_days, n) > panel.n_days, np.nan, cols["y"])
    cols["avail"] = panel.avail[:, target_days - 1].ravel()
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------- LSTM
class SeqBuilder:
    """Tạo mini-batch cửa sổ trượt cho LSTM encoder–decoder (chỉ dùng numpy, không cần torch).

    Đầu vào quá khứ: L ngày (origin − L + 1 ... origin). Đầu vào tương lai: biến biết trước
    của origin + 1 ... origin + H. Mục tiêu: doanh số origin + 1 ... origin + H (chỉ dùng cho loss).
    """

    def __init__(self, panel, calf: pd.DataFrame, feature_set: str, cutoff: int,
                 input_len: int = 112, horizon: int = 28, price_mode: str = "known"):
        self.panel = panel
        self.cutoff = int(cutoff)
        self.L = int(input_len)
        self.H = int(horizon)
        self.price_mode = price_mode
        groups = FEATURE_SETS[feature_set]
        self.groups = groups

        day_chans, day_names, day_groups = [], [], []
        if "cal_regular" in groups:
            wday = calf["wday"].to_numpy()
            for k in range(1, 8):
                day_chans.append((wday == k).astype(np.float32)); day_names.append(f"wday_{k}"); day_groups.append("cal_regular")
            for c in ("is_weekend", "month_sin", "month_cos"):
                day_chans.append(calf[c].to_numpy(np.float32)); day_names.append(c); day_groups.append("cal_regular")
        if "cal_irregular" in groups:
            for c in [f"ev_{e}" for e in EVENT_TYPES] + ["pre_event"]:
                day_chans.append(calf[c].to_numpy(np.float32)); day_names.append(c); day_groups.append("cal_irregular")
            for c in ("days_to_next_event", "days_since_event"):
                day_chans.append(calf[c].to_numpy(np.float32) / 60.0); day_names.append(c); day_groups.append("cal_irregular")
        n_cal = len(calf)
        self.CAL = np.stack(day_chans, axis=1) if day_chans else np.zeros((n_cal, 0), np.float32)

        self.series_names, self.series_groups = [], []
        if "snap" in groups:
            self.series_names.append("snap"); self.series_groups.append("snap")
        if "price" in groups:
            ref = price_reference(panel, cutoff)
            with np.errstate(all="ignore"):
                self.PR = np.nan_to_num(panel.P / ref[:, None], nan=0.0).astype(np.float32)
            self.series_names += ["price_rel", "price_gap"]; self.series_groups += ["price", "price"]

        self.channel_names = day_names + self.series_names
        self.channel_groups = day_groups + self.series_groups
        self.cats = np.stack([panel.meta[c + "_code"].to_numpy(np.int64) for c in CAT_COLS], axis=1)
        self.launch = panel.launch_day()

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def _cov(self, rows, cols, origins):
        """Biến biết trước tại các cột (ngày) cho từng dòng: [B, len, c]."""
        B, Lc = cols.shape
        safe = np.clip(cols, 0, self.CAL.shape[0] - 1)
        parts = [self.CAL[safe]]
        if "snap" in self.groups:
            parts.append(self.panel.snap[rows[:, None], safe][..., None].astype(np.float32))
        if "price" in self.groups:
            pcols = safe if self.price_mode == "known" else np.minimum(safe, origins[:, None] - 1)
            parts.append(self.PR[rows[:, None], pcols][..., None])
            parts.append(self.panel.P_gap[rows[:, None], pcols][..., None].astype(np.float32))
        return np.concatenate(parts, axis=2) if parts else np.zeros((B, Lc, 0), np.float32)

    def batch(self, rows, origins, with_target: bool = True, permute: str | None = None, rng=None) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        origins = np.asarray(origins, dtype=np.int64)
        past_cols = origins[:, None] - self.L + np.arange(self.L)[None, :]   # ngày origin−L+1..origin
        fut_cols = origins[:, None] + np.arange(self.H)[None, :]             # ngày origin+1..origin+H
        Y = self.panel.Y
        y_past = Y[rows[:, None], np.clip(past_cols, 0, None)].astype(np.float32)
        past_ok = (past_cols >= 0) & (past_cols + 1 >= self.launch[rows][:, None])
        y_past = np.where(past_ok, y_past, 0.0)
        n_ok = np.maximum(past_ok.sum(axis=1), 1)
        scale = 1.0 + y_past.sum(axis=1) / n_ok                               # chỉ dùng quá khứ
        out = {
            "y_past": (y_past / scale[:, None]).astype(np.float32),
            "past_mask": past_ok.astype(np.float32),
            "c_past": self._cov(rows, past_cols, origins),
            "c_fut": self._cov(rows, fut_cols, origins),
            "cats": self.cats[rows],
            "scale": scale.astype(np.float32),
            "rows": rows,
            "origins": origins,
        }
        if permute is not None:
            rng = rng or np.random.default_rng(0)
            ch = [i for i, g in enumerate(self.channel_groups) if g == permute]
            if ch:
                perm = rng.permutation(len(rows))
                out["c_past"][:, :, ch] = out["c_past"][perm][:, :, ch]
                out["c_fut"][:, :, ch] = out["c_fut"][perm][:, :, ch]
        if with_target:
            if fut_cols.max() >= Y.shape[1]:
                raise ValueError("Không có nhãn cho ngày sau d_{}".format(Y.shape[1]))
            y_fut = Y[rows[:, None], fut_cols].astype(np.float32)
            out["y_fut"] = y_fut / scale[:, None]
            out["fut_mask"] = self.panel.avail[rows[:, None], fut_cols].astype(np.float32)
        return out

    def sample_train_pairs(self, n_samples: int, train_years: float, rng) -> tuple[np.ndarray, np.ndarray]:
        """Lấy ngẫu nhiên (chuỗi, mốc) để huấn luyện, mọi mục tiêu ≤ cutoff."""
        hi = self.cutoff - self.H
        lo_global = int(self.cutoff - self.H - 364 * train_years + 1)
        lo = np.maximum(self.launch + self.L - 1, lo_global)
        span = np.maximum(hi - lo + 1, 0)
        if span.sum() == 0:
            raise ValueError("Không có mốc huấn luyện hợp lệ; giảm input_len hoặc tăng train_years")
        p = span / span.sum()
        rows = rng.choice(len(span), size=n_samples, p=p)
        origins = lo[rows] + (rng.random(n_samples) * span[rows]).astype(np.int64)
        if origins.max() + self.H > self.cutoff:
            raise LeakageError("Mốc huấn luyện có mục tiêu sau cutoff")
        return rows.astype(np.int64), origins.astype(np.int64)
