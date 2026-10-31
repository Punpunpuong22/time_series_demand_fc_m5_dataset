"""Kiểm thử cho thiết kế 0.3.0: ba nhóm yếu tố lịch, ARIMAX, NBEATSx (giả lập) và bảng tác động theo nhóm."""
import numpy as np
import pandas as pd
import pytest

from m5cal.arima_exog import one_step_predictions, run_arima_panel
from m5cal.calendar_groups import (GROUP_COLUMNS, NESTED, SET_GROUPS, build_calendar_groups, set_columns)
from m5cal.group_effects import effect_matrix, group_effects
from m5cal.nbeats_exog import attach, build_nf_frame, nbeatsx_cross_validation


def _cal(raw_dir):
    return build_calendar_groups(pd.read_csv(raw_dir / "calendar.csv"), state="CA")


# ---------------------------------------------------------------- nhóm yếu tố lịch
def test_every_group_column_exists_and_is_finite(raw_dir):
    cal = _cal(raw_dir)
    for kind, groups in GROUP_COLUMNS.items():
        for g, cols in groups.items():
            missing = [c for c in cols if c not in cal.columns]
            assert not missing, (kind, g, missing)
            assert np.isfinite(cal[cols].to_numpy(float)).all(), (kind, g)


def test_g1_decomposition(raw_dir):
    cal = _cal(raw_dir)
    d = cal["date"]
    assert (cal["day_of_week"] == d.dt.dayofweek).all()
    assert set(cal.loc[d.dt.month.isin([12, 1, 2]), "season"]) == {0}
    assert set(cal.loc[d.dt.month.isin([6, 7, 8]), "season"]) == {2}
    np.testing.assert_allclose(cal["dow_sin"] ** 2 + cal["dow_cos"] ** 2, 1.0)
    # one-hot thứ: đúng một cột bằng 1, trừ thứ Hai (mốc)
    dows = cal[[f"dow_{k}" for k in range(1, 7)]].sum(axis=1)
    assert ((dows == 1) | (cal["day_of_week"] == 0)).all() and dows.max() == 1


def test_g2_events_and_date_type_priority(raw_dir):
    cal = _cal(raw_dir)
    ev = cal[cal["has_event"] == 1]
    assert len(ev) > 0
    assert (ev["days_to_next_event"] == 0).all() and (ev["days_since_last_event"] == 0).all()
    assert (cal.loc[cal["is_holiday"] == 1, "date_type"] == "Holiday").all()
    pre = cal[(cal["is_pre_event"] == 1) & (cal["has_event"] == 0)]
    assert (pre["date_type"] == "Pre_Event").all()
    assert set(cal["date_type"]) <= {"Weekday", "Weekend", "Pre_Event", "Other_Event", "Holiday"}


def test_g3_snap_run_day(raw_dir):
    cal = _cal(raw_dir)
    s = cal["snap"].to_numpy()
    r = cal["snap_run_day"].to_numpy()
    assert ((r == 0) == (s == 0)).all()
    starts = (s == 1) & (np.r_[0, s[:-1]] == 0)
    assert (r[starts] == 1).all()
    inside = (s == 1) & ~starts
    assert (r[inside] == np.r_[0, r[:-1]][inside] + 1).all()


def test_feature_sets_are_nested_and_drop_one_correct():
    for kind in GROUP_COLUMNS:
        cols = [set(set_columns(s, kind)) for s in NESTED]
        assert all(a < b for a, b in zip(cols, cols[1:])), kind
        assert set(set_columns("D1", kind)).isdisjoint(GROUP_COLUMNS[kind]["G1"])
        assert set(set_columns("D2", kind)).isdisjoint(GROUP_COLUMNS[kind]["G2"])
        assert set(set_columns("S3", kind)) == set(set_columns("D1", kind)) | set(GROUP_COLUMNS[kind]["G1"])
    assert SET_GROUPS["S0"] == [] and "date_type" not in set_columns("S3", "tree")
    with pytest.raises(KeyError):
        set_columns("S9", "tree")


# ---------------------------------------------------------------- ARIMAX (mô hình giả lập)
class FakeSARIMAX:
    """Dự báo một bước = y[t−1] + tổng các biến ngoại sinh của ngày t; ghi lại dữ liệu được dùng để ước lượng."""
    fitted_on = []

    def __init__(self, y, exog=None, **kw):
        self.y, self.exog = np.asarray(y, float), exog
        FakeSARIMAX.fitted_on.append((len(self.y), None if exog is None else np.asarray(exog).shape[1]))

    def fit(self, **kw):
        return _FakeRes(self.y, self.exog)


class _FakeRes:
    aic = 1.0

    def __init__(self, y, X):
        self.y, self.X = y, X

    def apply(self, y, exog=None):
        return _FakeRes(np.asarray(y, float), exog)

    def get_prediction(self, start):
        X = np.zeros((len(self.y), 1)) if self.X is None else np.asarray(self.X, float)
        pred = self.y[start - 1:-1] + X[start:].sum(axis=1)
        return type("P", (), {"predicted_mean": pred})()


def test_arima_one_step_uses_only_past_y_and_same_day_exog():
    y = np.arange(20, dtype=float)
    X = np.zeros((20, 2)); X[3, 0] = 2.0; X[15, 0] = 5.0; X[:, 1] = 1.0           # cột 2 là hằng → phải bị bỏ
    FakeSARIMAX.fitted_on.clear()
    pred = one_step_predictions(y, X, 12, (1, 0, 0), FakeSARIMAX)
    assert len(pred) == 8
    expect = y[11:19] + X[12:, 0]
    np.testing.assert_allclose(pred, expect)
    assert FakeSARIMAX.fitted_on[0] == (12, 1)                     # chỉ ước lượng trên 12 ngày đầu, 1 cột


def test_run_arima_panel_dates_and_orders():
    dates = pd.date_range("2015-01-01", periods=60)
    df = pd.DataFrame([(i, d, float(k % 7 + j)) for j, i in enumerate(["a", "b"]) for k, d in enumerate(dates)],
                      columns=["item_id", "date", "demand"])
    df["x"] = (df["date"].dt.dayofweek >= 5).astype(float)
    out = run_arima_panel(df, ["x"], dates[39], orders=None, candidates=((1, 0, 0),), model_cls=FakeSARIMAX)
    assert set(out["item_id"]) == {"a", "b"} and len(out) == 2 * 20
    assert out["date"].min() == dates[40] and (out["pred"] >= 0).all()
    assert out.attrs["orders"] == {"a": (1, 0, 0), "b": (1, 0, 0)}
    # mặt hàng lên kệ muộn: ước lượng đến cuối tập kiểm định, vẫn có dự báo cho tập kiểm tra
    late = df[(df["item_id"] == "b") & (df["date"] >= dates[20])]
    df2 = pd.concat([df[df["item_id"] == "a"], late])
    o2 = run_arima_panel(df2, ["x"], dates[39], candidates=((1, 0, 0),), model_cls=FakeSARIMAX, fallback_end=dates[49])
    assert o2[o2["item_id"] == "b"]["date"].min() == dates[50]
    o3 = run_arima_panel(df2, ["x"], dates[39], candidates=((1, 0, 0),), model_cls=FakeSARIMAX)
    assert set(o3["item_id"]) == {"a"}


# ---------------------------------------------------------------- NBEATSx (thư viện giả lập)
class FakeNBEATSx:
    def __init__(self, **kw):
        self.kw = kw


class FakeNF:
    last = None

    def __init__(self, models, freq):
        self.models = models
        FakeNF.last = self

    def cross_validation(self, df, n_windows, test_size, val_size, step_size, refit):
        assert n_windows is None and step_size == 1 and refit is False
        out = []
        for uid, g in df.groupby("unique_id"):
            g = g.sort_values("ds")
            pred = g["y"].shift(1).iloc[-test_size:]
            out.append(pd.DataFrame({"unique_id": uid, "ds": g["ds"].iloc[-test_size:], "cutoff": g["ds"].iloc[-test_size - 1:-1].values,
                                     "y": g["y"].iloc[-test_size:], "NBEATSx": pred.values}))
        return pd.concat(out, ignore_index=True)


def test_nbeatsx_pipeline_with_fake_library():
    dates = pd.date_range("2015-01-01", periods=50)
    df = pd.DataFrame([(i, d, float(k)) for i in ["a", "b"] for k, d in enumerate(dates)], columns=["item_id", "date", "demand"])
    df["ev"] = 0.0
    nf_df = build_nf_frame(df, ["ev"])
    assert list(nf_df.columns[:3]) == ["unique_id", "ds", "y"] and nf_df["ev"].dtype == np.float32
    preds = nbeatsx_cross_validation(nf_df, ["ev"], test_size=10, val_size=10, seed=7, params={"input_size": 14},
                                     nf_cls=FakeNF, model_cls=FakeNBEATSx, loss="mse")
    kw = FakeNF.last.models[0].kw
    assert kw["h"] == 1 and kw["futr_exog_list"] == ["ev"] and kw["random_seed"] == 7 and kw["input_size"] == 14
    assert kw["accelerator"] == "auto"
    test_eval = df[df["date"] > dates[39]][["item_id", "date", "demand"]].reset_index(drop=True)
    out = attach(test_eval, preds, "N-BEATS-S2")
    np.testing.assert_allclose(out["N-BEATS-S2"], out["demand"] - 1)
    with pytest.raises(ValueError):
        attach(test_eval, preds.iloc[:-1], "N-BEATS-S2")
    none = nbeatsx_cross_validation(nf_df, [], 10, 10, nf_cls=FakeNF, model_cls=FakeNBEATSx, loss="mse")
    assert FakeNF.last.models[0].kw["futr_exog_list"] is None and len(none) == 20

    # thiếu ít dòng: điền bằng mô hình cơ sở; thiếu nhiều dòng: vẫn báo lỗi
    test_eval["MA28"] = 99.0
    filled = attach(test_eval, preds.iloc[:-1], "N-BEATS-S2", fallback_col="MA28", max_missing=0.1)
    assert filled["N-BEATS-S2"].iloc[-1] == 99.0 and filled.attrs["filled"]["N-BEATS-S2"] == 1
    with pytest.raises(ValueError):
        attach(test_eval, preds.iloc[:5], "N-BEATS-S2", fallback_col="MA28", max_missing=0.1)
    # chuỗi quá ngắn bị bỏ trước khi gọi cross_validation
    short = nf_df[~((nf_df["unique_id"] == nf_df["unique_id"].iloc[0]) & (nf_df["ds"] < nf_df["ds"].max() - pd.Timedelta(days=15)))]
    part = nbeatsx_cross_validation(short, ["ev"], 10, 10, nf_cls=FakeNF, model_cls=FakeNBEATSx, loss="mse")
    assert part["unique_id"].nunique() == nf_df["unique_id"].nunique() - 1


# ---------------------------------------------------------------- bảng tác động
def test_group_effects_signs_and_relevant_days():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-08-10", periods=120)
    df = pd.DataFrame([(i, d) for i in range(10) for d in dates], columns=["item_id", "date"])
    df["demand"] = rng.poisson(3, len(df)).astype(float)
    df["date_type"] = np.where(df["date"].dt.day % 10 == 0, "Holiday", "Weekday")
    df["snap"] = (df["date"].dt.day <= 10).astype(int)
    noise = lambda s: df["demand"] + rng.normal(0, s, len(df))
    for st, s in {"S0": 2.0, "S1": 1.0, "S2": 1.0, "S3": 1.0, "D1": 2.0, "D2": 1.0}.items():
        df[f"M-{st}"] = noise(s).clip(lower=0)
    eff = group_effects(df, ["M", "absent"])
    assert set(eff["model"]) == {"M"}
    g1 = eff[(eff["group"] == "G1") & (eff["type"] == "tăng thêm")].iloc[0]
    assert g1["delta (pp)"] < 0 and g1["helps (5%)"]
    rel = eff[(eff["group"] == "G2") & (eff["scope"] == "ngày liên quan")]
    assert len(rel) == 2 and (rel["days"] == 12).all()
    assert not ((eff["group"] == "G1") & (eff["scope"] == "ngày liên quan")).any()
    mat = effect_matrix(eff)
    assert list(mat.columns) == ["G1", "G2", "G3"] and mat.loc["M", "G1"] < 0
