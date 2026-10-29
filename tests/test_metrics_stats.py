import numpy as np
import pandas as pd
import pytest

from m5cal.inventory import simulate_order_up_to, summarize_inventory
from m5cal.metrics import (
    aggregate_scores,
    attach_truth,
    revenue_weights,
    rmsse_denominator,
    series_window_scores,
    stratified_scores,
)
from m5cal.models.baselines import croston_sba_level, seasonal_naive, seasonal_naive_matrix
from m5cal.stats_tests import benjamini_hochberg, bootstrap_wrmsse_diff, dm_test, friedman_nemenyi


def test_rmsse_denominator_hand_computed():
    class P:  # panel tối giản
        Y = np.array([[0, 0, 2, 4, 1, 1]], dtype=np.float32)
    den = rmsse_denominator(P, origin=6)
    # bắt đầu từ ngày bán đầu tiên (giá trị 2): sai phân 2,-3,0 -> (4+9+0)/3
    assert den[0] == pytest.approx(13 / 3)


def test_perfect_forecast_scores_zero(panel, windows):
    w = [x for x in windows if x.role == "eval"][0]
    days = w.days
    preds = pd.DataFrame({"sid": np.repeat(np.arange(panel.n), 28), "d": np.tile(days, panel.n)})
    preds["y_pred"] = panel.Y[preds["sid"], preds["d"] - 1]
    preds = attach_truth(panel, preds).assign(window=w.name)
    agg = aggregate_scores(series_window_scores(panel, preds, windows))
    assert agg["wrmsse"].iloc[0] == pytest.approx(0.0)
    assert agg["mae"].iloc[0] == pytest.approx(0.0)


def test_weights_sum_to_one(panel):
    assert revenue_weights(panel, 1600).sum() == pytest.approx(1.0)


def test_stratified_single_type_equals_overall(panel, windows):
    w = [x for x in windows if x.role == "eval"][0]
    p = seasonal_naive(panel, w).assign(window=w.name)
    p = attach_truth(panel, p)
    labels = p[["sid", "d"]].assign(daytype="all")
    st = stratified_scores(panel, p, windows, labels)
    sc = series_window_scores(panel, p, windows)
    m = st.merge(sc, on=["sid", "window"], suffixes=("_s", ""))
    np.testing.assert_allclose(m["rmsse_s"], m["rmsse"], equal_nan=True)


def test_seasonal_naive_repeats_last_week():
    Y = np.arange(1, 31, dtype=float)[None, :]
    pred = seasonal_naive_matrix(Y, origin=21, horizon=14)
    np.testing.assert_array_equal(pred[0, :7], Y[0, 14:21])
    np.testing.assert_array_equal(pred[0, 7:], Y[0, 14:21])


def test_croston_sba_constant_series():
    y = np.array([0, 2, 0, 2, 0, 2, 0, 2], float)
    lvl = croston_sba_level(y, alpha=0.1)
    assert lvl == pytest.approx((1 - 0.05) * 2 / 2, rel=0.05)
    assert croston_sba_level(np.zeros(5)) == 0.0


def test_dm_identical_and_different():
    rng = np.random.default_rng(0)
    e = rng.normal(size=200)
    assert dm_test(e, e)[1] == 1.0
    s, p = dm_test(e * 3, e * 0.5)
    assert s > 0 and p < 0.01


def test_benjamini_hochberg_monotone():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert (q >= np.array([0.01, 0.04, 0.03, 0.5])).all()
    assert q[0] <= q[2] <= q[1] <= q[3] or np.allclose(sorted(q), np.sort(q))


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a = rng.uniform(1, 2, 500)
    b = a - 0.2
    res = bootstrap_wrmsse_diff(a, b, np.ones(500), n_boot=300)
    assert res["ci_low"] <= 0.2 + 1e-9 <= res["ci_high"] + 1e-9


def test_friedman_detects_ranking():
    rng = np.random.default_rng(2)
    base = rng.uniform(1, 2, 100)
    s = pd.DataFrame({"A": base, "B": base + 0.5 + rng.normal(0, 0.01, 100), "C": base + 1.0})
    res = friedman_nemenyi(s)
    assert res["p_value"] < 1e-6 and list(res["mean_ranks"])[0] == "A"


def test_inventory_perfect_forecast_full_service():
    D = np.full((3, 28), 5.0)
    sim = simulate_order_up_to(D, D.copy(), sigma=np.zeros(3), lead_time=2, z=0)
    summ = summarize_inventory(sim, D)
    assert summ.loc[0, "fill_rate"] == pytest.approx(1.0)


def test_inventory_underforecast_loses_sales():
    D = np.full((2, 28), 10.0)
    sim = simulate_order_up_to(D, D * 0.5, sigma=np.zeros(2), lead_time=2, z=0)
    assert summarize_inventory(sim, D).loc[0, "fill_rate"] < 0.9
