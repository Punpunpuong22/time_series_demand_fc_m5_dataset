"""Bộ kiểm thử CHỐNG RÒ RỈ THÔNG TIN (mục 7.1 và 7.9 của báo cáo).

Ý tưởng chung: thay đổi tùy ý dữ liệu SAU mốc dự báo. Nếu bất kỳ đặc trưng, phép chuẩn hóa
hoặc đầu vào mô hình nào thay đổi theo, nghĩa là có rò rỉ -> kiểm thử thất bại.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from m5cal.features import (
    HORIZON_GROUPS,
    SeqBuilder,
    feature_columns,
    lgbm_frame,
    price_reference,
)
from m5cal.splits import assert_train_before
from m5cal.utils import LeakageError

CUTOFF = 1600


def perturb_future(panel, cutoff, seed=0):
    """Bản sao panel với doanh số và giá sau cutoff bị thay bằng giá trị ngẫu nhiên."""
    p = copy.copy(panel)
    rng = np.random.default_rng(seed)
    Y = panel.Y.copy()
    Y[:, cutoff:] = rng.integers(0, 1000, size=Y[:, cutoff:].shape)
    p.Y = Y
    P = panel.P.copy()
    P[:, cutoff:] = np.where(np.isnan(P[:, cutoff:]), np.nan, rng.uniform(0.1, 100, size=P[:, cutoff:].shape))
    p.P = P
    return p


@pytest.mark.parametrize("groups", ["weekly", "lag28", "daily"])
@pytest.mark.parametrize("feature_set", ["H", "HC"])
def test_lgbm_prediction_features_ignore_future_sales(panel, calf, groups, feature_set):
    """Đặc trưng của dòng dự báo không đổi khi doanh số sau mốc thay đổi."""
    future = perturb_future(panel, CUTOFF)
    for a, b in HORIZON_GROUPS[groups][:2]:
        days = CUTOFF + np.arange(a, b + 1)
        f1 = lgbm_frame(panel, calf, CUTOFF, b, days, feature_set)
        f2 = lgbm_frame(future, calf, CUTOFF, b, days, feature_set)
        cols = feature_columns(f1)
        pd.testing.assert_frame_equal(f1[cols], f2[cols])


def test_lgbm_training_features_ignore_future(panel, calf):
    """Đặc trưng của dòng huấn luyện (mục tiêu ≤ mốc) không đổi khi dữ liệu sau mốc thay đổi."""
    future = perturb_future(panel, CUTOFF)
    days = np.arange(CUTOFF - 60, CUTOFF + 1)
    for fs in ("H", "HC"):
        f1 = lgbm_frame(panel, calf, CUTOFF, 7, days, fs)
        f2 = lgbm_frame(future, calf, CUTOFF, 7, days, fs)
        cols = feature_columns(f1) + ["y"]
        pd.testing.assert_frame_equal(f1[cols], f2[cols])


def test_price_normalisation_uses_only_past(panel):
    """Giá tham chiếu chỉ tính trên các ngày ≤ mốc."""
    future = perturb_future(panel, CUTOFF)
    np.testing.assert_allclose(price_reference(panel, CUTOFF), price_reference(future, CUTOFF), equal_nan=True)


def test_price_features_on_training_rows_ignore_future_prices(panel, calf):
    future = perturb_future(panel, CUTOFF)
    days = np.arange(CUTOFF - 30, CUTOFF + 1)
    f1 = lgbm_frame(panel, calf, CUTOFF, 7, days, "HCP")
    f2 = lgbm_frame(future, calf, CUTOFF, 7, days, "HCP")
    cols = [c for c in feature_columns(f1) if c.startswith("price")]
    pd.testing.assert_frame_equal(f1[cols], f2[cols])


def test_last_known_price_mode_blocks_future_prices(panel, calf):
    """Ở chế độ phân tích độ nhạy 'last_known', dòng dự báo không thấy giá tương lai."""
    future = perturb_future(panel, CUTOFF)
    days = CUTOFF + np.arange(1, 8)
    f1 = lgbm_frame(panel, calf, CUTOFF, 7, days, "HCP", price_mode="last_known")
    f2 = lgbm_frame(future, calf, CUTOFF, 7, days, "HCP", price_mode="last_known")
    cols = [c for c in feature_columns(f1) if c.startswith("price")]
    pd.testing.assert_frame_equal(f1[cols], f2[cols])


def test_insufficient_lag_raises(panel, calf):
    """Lag tối thiểu nhỏ hơn tầm dự báo phải bị chặn (lỗi), không được âm thầm dùng dữ liệu tương lai."""
    with pytest.raises(LeakageError):
        lgbm_frame(panel, calf, CUTOFF, 7, CUTOFF + np.arange(1, 15), "H")


def test_lag_values_are_correct(panel, calf):
    """lag_p0 của ngày d với lag tối thiểu L bằng doanh số ngày d − L."""
    days = np.arange(CUTOFF - 10, CUTOFF + 1)
    f = lgbm_frame(panel, calf, CUTOFF, 7, days, "H")
    row = f[(f["sid"] == 3) & (f["d"] == CUTOFF)].iloc[0]
    expected = panel.Y[3, CUTOFF - 7 - 1] if panel.avail[3, CUTOFF - 7 - 1] else np.nan
    assert row["lag_p0"] == pytest.approx(expected, nan_ok=True)


def test_rolling_mean_correct(panel, calf):
    sid, d, L = 5, CUTOFF, 14
    f = lgbm_frame(panel, calf, CUTOFF, L, np.array([d]), "H")
    row = f[f["sid"] == sid].iloc[0]
    window = panel.Y[sid, d - L - 7: d - L].astype(float)        # ngày d−L−6 ... d−L
    window = np.where(panel.avail[sid, d - L - 7: d - L], window, np.nan)
    assert row["roll_mean_7"] == pytest.approx(np.nanmean(window), rel=1e-5, nan_ok=True)


def test_masked_sales_hides_future_and_prelaunch(panel):
    """Lớp bảo vệ thứ hai: ma trận doanh số dùng để tính đặc trưng không chứa ngày > mốc."""
    from m5cal.features import masked_sales

    Ym = masked_sales(panel, CUTOFF)
    assert np.isnan(Ym[:, CUTOFF:]).all()
    assert np.isnan(Ym[~panel.avail[:, : Ym.shape[1]]]).all()


def test_training_rows_must_not_pass_cutoff():
    with pytest.raises(LeakageError):
        assert_train_before(100, np.arange(90, 102))


# ----------------------------------------------------------------------------- LSTM inputs
@pytest.mark.parametrize("feature_set", ["H", "HC", "HCP"])
def test_seq_inputs_ignore_future_sales(panel, calf, feature_set):
    """Đầu vào và hệ số chuẩn hóa của LSTM tại mốc T không phụ thuộc doanh số sau T."""
    origin = CUTOFF
    future = perturb_future(panel, origin)
    b1 = SeqBuilder(panel, calf, feature_set, cutoff=origin, input_len=56)
    b2 = SeqBuilder(future, calf, feature_set, cutoff=origin, input_len=56)
    rows = np.arange(panel.n)
    x1 = b1.batch(rows, np.full(panel.n, origin), with_target=False)
    x2 = b2.batch(rows, np.full(panel.n, origin), with_target=False)
    for k in ("y_past", "past_mask", "scale", "cats"):
        np.testing.assert_array_equal(x1[k], x2[k])
    if feature_set != "HCP":   # giá tương lai là biến biết trước theo giả định -> chỉ so khi không dùng giá
        np.testing.assert_array_equal(x1["c_fut"], x2["c_fut"])
    np.testing.assert_array_equal(x1["c_past"], x2["c_past"])


def test_seq_targets_are_strictly_future(panel, calf):
    origin = CUTOFF
    b = SeqBuilder(panel, calf, "H", cutoff=origin, input_len=28)
    x = b.batch(np.array([0, 1]), np.array([origin, origin]))
    np.testing.assert_allclose(x["y_fut"] * x["scale"][:, None], panel.Y[[0, 1], origin: origin + 28], rtol=1e-5)
    np.testing.assert_allclose(x["y_past"][:, -1] * x["scale"],
                               np.where(panel.avail[[0, 1], origin - 1], panel.Y[[0, 1], origin - 1], 0), rtol=1e-5)


def test_seq_training_pairs_targets_before_cutoff(panel, calf):
    b = SeqBuilder(panel, calf, "HC", cutoff=CUTOFF, input_len=56)
    rows, origins = b.sample_train_pairs(5000, 3, np.random.default_rng(0))
    assert (origins + 28 <= CUTOFF).all()
    assert (origins - 56 + 1 >= b.launch[rows]).all()


def test_seq_price_scaling_uses_only_past(panel, calf):
    future = perturb_future(panel, CUTOFF)
    b1 = SeqBuilder(panel, calf, "HCP", cutoff=CUTOFF, input_len=28)
    b2 = SeqBuilder(future, calf, "HCP", cutoff=CUTOFF, input_len=28)
    np.testing.assert_array_equal(b1.PR[:, :CUTOFF], b2.PR[:, :CUTOFF])
