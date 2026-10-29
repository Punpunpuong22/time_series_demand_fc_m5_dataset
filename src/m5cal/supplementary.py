"""Thí nghiệm bổ sung cho thí nghiệm chính (notebook v3: CA_1 FOODS, 100 mặt hàng, dự báo 1 ngày tới).

Mọi hàm ở đây chỉ cần numpy/pandas/scipy và làm việc trên khung `test_eval` của notebook:
một dòng cho mỗi (item_id, date), cột `demand`, `date`, `date_type`, ... và một cột dự báo cho mỗi mô hình.
Notebook v3 chép nguyên văn tệp này vào một ô, nên mã trong notebook và mã được kiểm thử là một.

Nội dung
- wape, bias, dm_test_daily      : giống hệt định nghĩa trong notebook v2 (để số liệu khớp)
- dm_pairs_table                 : bảng Diebold–Mariano cho nhiều cặp mô hình
- block_bootstrap_wape_diff      : khoảng tin cậy cho chênh lệch WAPE, lấy mẫu lại theo khối tuần
- permutation_importance         : hoán vị theo nhóm biến, dùng chung cho LightGBM (DataFrame) và LSTM (mảng)
- permute_frame_columns, permute_array_channels : bộ hoán vị cho hai loại đầu vào
- Timer, add_ensemble            : đo thời gian chạy; trung bình dự báo nhiều mô hình
"""
from __future__ import annotations

import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- độ đo (giống notebook v2)
def wape(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.abs(y - p).sum() / (np.abs(y).sum() + 1e-8))


def bias(y, p) -> float:
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def dm_test_daily(df: pd.DataFrame, col1: str, col2: str, loss: str = "abs", lag: int = 7,
                  y_col: str = "demand", date_col: str = "date") -> tuple[float, float]:
    """Diebold–Mariano trên chênh lệch tổn thất cộng gộp theo ngày, phương sai Bartlett đến độ trễ `lag`.

    stat > 0 nghĩa là col1 có tổn thất CAO hơn (kém hơn) col2. Giống hàm dm_test của notebook v2.
    """
    e1 = df[y_col] - df[col1]
    e2 = df[y_col] - df[col2]
    if loss == "abs":
        l1, l2 = e1.abs(), e2.abs()
    elif loss == "sq":
        l1, l2 = e1 ** 2, e2 ** 2
    else:
        raise ValueError("loss phải là 'abs' hoặc 'sq'")
    d = (l1 - l2).groupby(df[date_col]).sum().to_numpy()
    n, dm = len(d), d.mean()
    var = np.var(d)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.mean((d[k:] - dm) * (d[:-k] - dm))
    if var <= 0:
        return float("nan"), float("nan")
    stat = dm / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def dm_pairs_table(df: pd.DataFrame, pairs: Sequence[tuple[str, str]], losses=("abs", "sq"),
                   lag: int = 7, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        for loss in losses:
            s, p = dm_test_daily(df, a, b, loss, lag)
            rows.append({"model 1": a, "model 2": b, "loss": loss, "DM stat": round(s, 3),
                         "p-value": round(p, 4), f"significant ({int(alpha * 100)}%)": bool(p < alpha),
                         "better": b if s > 0 else a})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- bootstrap theo khối tuần
def _week_key(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize().to_numpy()


def block_bootstrap_wape_diff(df: pd.DataFrame, col_a: str, col_b: str, group_col: str | None = None,
                              n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                              y_col: str = "demand", date_col: str = "date") -> pd.DataFrame:
    """Khoảng tin cậy cho WAPE(col_b) − WAPE(col_a), tính bằng điểm phần trăm.

    Lấy mẫu lại có hoàn lại các TUẦN lịch (khối 7 ngày) để giữ tương quan trong tuần; mọi dòng của tuần
    được chọn đi cùng nhau. Khi có `group_col` (ví dụ `date_type`), WAPE được tính riêng trong từng nhóm
    trên cùng một mẫu tuần. Giá trị âm nghĩa là col_b tốt hơn col_a.

    p_two_sided là tỷ lệ bootstrap hai phía quanh 0 (xấp xỉ, dùng để tham khảo cùng khoảng tin cậy).
    """
    rng = np.random.default_rng(seed)
    work = pd.DataFrame({
        "week": _week_key(df[date_col]),
        "g": df[group_col].to_numpy() if group_col else "Tất cả",
        "ae_a": (df[y_col] - df[col_a]).abs().to_numpy(float),
        "ae_b": (df[y_col] - df[col_b]).abs().to_numpy(float),
        "y": df[y_col].abs().to_numpy(float),
    })
    agg = work.groupby(["week", "g"], sort=True)[["ae_a", "ae_b", "y"]].sum()
    weeks = agg.index.get_level_values(0).unique()
    groups = list(pd.unique(work["g"]))
    # ma trận tuần × nhóm cho ba tổng
    cube = {k: agg[k].unstack("g").reindex(index=weeks, columns=groups).fillna(0.0).to_numpy() for k in ("ae_a", "ae_b", "y")}
    W = len(weeks)
    idx = rng.integers(0, W, size=(n_boot, W))
    out = []
    for j, g in enumerate(groups):
        A, B, Y = cube["ae_a"][:, j], cube["ae_b"][:, j], cube["y"][:, j]
        if Y.sum() <= 0:
            continue
        point = 100 * (B.sum() / Y.sum() - A.sum() / Y.sum())
        ys = Y[idx].sum(axis=1)
        ok = ys > 0
        diffs = 100 * (B[idx].sum(axis=1)[ok] - A[idx].sum(axis=1)[ok]) / ys[ok]
        lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        out.append({"group": g, "model a": col_a, "model b": col_b,
                    "WAPE a (%)": round(100 * A.sum() / Y.sum(), 2), "WAPE b (%)": round(100 * B.sum() / Y.sum(), 2),
                    "diff b-a (pp)": round(point, 2), "CI low": round(float(lo), 2), "CI high": round(float(hi), 2),
                    "p_two_sided": round(float(min(p, 1.0)), 4),
                    "CI excludes 0": bool(lo > 0 or hi < 0),
                    "weeks with data": int((Y > 0).sum()),
                    "days": int(df.loc[(df[group_col] == g) if group_col else slice(None), date_col].nunique())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- permutation importance theo nhóm
Permuter = Callable[[object, np.random.Generator], object]


def permute_frame_columns(cols: Sequence[str]) -> Permuter:
    """Hoán vị CÙNG MỘT thứ tự dòng cho mọi cột trong nhóm (giữ quan hệ giữa các cột trong nhóm)."""
    cols = list(cols)

    def _perm(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        Xp = X.copy()
        order = rng.permutation(len(X))
        for c in cols:
            vals = X[c].to_numpy()
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                Xp[c] = pd.Categorical.from_codes(X[c].cat.codes.to_numpy()[order], dtype=X[c].dtype)
            else:
                Xp[c] = vals[order]
        return Xp

    return _perm


def permute_array_channels(spec: Mapping[str, Sequence[int] | None]) -> Permuter:
    """Hoán vị mẫu (trục 0) cho các kênh được chọn của nhiều mảng đầu vào, dùng chung một thứ tự.

    spec: {tên đầu vào: danh sách chỉ số kênh ở trục cuối, hoặc None = toàn bộ mảng}.
    Ví dụ nhóm lịch của LSTM-C: {"hist": [1, ..., 15], "future": [0, ..., 14]}.
    """
    spec = {k: (None if v is None else list(v)) for k, v in spec.items()}

    def _perm(inputs: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict:
        n = len(next(iter(inputs.values())))
        order = rng.permutation(n)
        out = dict(inputs)
        for name, ch in spec.items():
            arr = np.array(inputs[name], copy=True)
            if ch is None:
                arr = arr[order]
            else:
                arr[..., ch] = arr[order][..., ch]
            out[name] = arr
        return out

    return _perm


def permutation_importance(predict_fn: Callable[[object], np.ndarray], inputs, y,
                           permuters: Mapping[str, Permuter], metric: Callable = wape,
                           n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Mức tăng của `metric` (điểm phần trăm nếu metric là wape) khi hoán vị từng nhóm biến.

    predict_fn nhận đúng kiểu `inputs` (DataFrame cho LightGBM, dict mảng cho LSTM) và trả về dự báo.
    Dự báo được cắt tại 0 như trong notebook.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    base = metric(y, np.clip(np.asarray(predict_fn(inputs), float).ravel(), 0, None))
    rows = []
    for name, perm in permuters.items():
        scores = []
        for _ in range(n_repeats):
            pred = np.clip(np.asarray(predict_fn(perm(inputs, rng)), float).ravel(), 0, None)
            scores.append(metric(y, pred))
        scores = np.asarray(scores)
        rows.append({"group": name, "base": 100 * base, "permuted mean": 100 * scores.mean(),
                     "increase (pp)": 100 * (scores.mean() - base), "increase std (pp)": 100 * scores.std(ddof=0),
                     "repeats": n_repeats})
    return pd.DataFrame(rows).sort_values("increase (pp)", ascending=False).round(3).reset_index(drop=True)


# --------------------------------------------------------------------------- tiện ích
class Timer:
    """with Timer() as t: ...;  t.seconds"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.seconds = float("nan")
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


def add_ensemble(df: pd.DataFrame, cols: Sequence[str], name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột dự báo: {missing}")
    df[name] = df[list(cols)].mean(axis=1)
    return df
