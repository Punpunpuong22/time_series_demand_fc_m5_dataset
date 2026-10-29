"""Kiểm thử cho các thí nghiệm bổ sung (không cần dữ liệu M5, chỉ dữ liệu giả lập nhỏ)."""
import numpy as np
import pandas as pd
import pytest

from m5cal.supplementary import (Timer, add_ensemble, block_bootstrap_wape_diff, dm_pairs_table, dm_test_daily,
                                 permutation_importance, permute_array_channels, permute_frame_columns, wape)


def _frame(n_items=20, n_days=140, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-08-10", periods=n_days, freq="D")
    df = pd.DataFrame([(i, d) for i in range(n_items) for d in dates], columns=["item_id", "date"])
    lam = 2 + 2 * (df["date"].dt.dayofweek >= 5)
    df["demand"] = rng.poisson(lam).astype(float)
    df["date_type"] = np.where(df["date"].dt.dayofweek >= 5, "Weekend", "Weekday")
    df["good"] = lam.astype(float)                       # biết đúng kỳ vọng
    df["bad"] = df["good"] + rng.normal(0, 1.5, len(df)).clip(-2, None)
    df["same"] = df["good"]
    return df


def test_dm_sign_and_zero_variance():
    df = _frame()
    s, p = dm_test_daily(df, "bad", "good", "abs")
    assert s > 0 and p < 0.05                           # 'bad' kém hơn => stat > 0
    s2, p2 = dm_test_daily(df, "good", "same")
    assert np.isnan(s2) and np.isnan(p2)                # hai dự báo trùng nhau => không kiểm định được
    tab = dm_pairs_table(df, [("bad", "good"), ("bad", "missing")])
    assert len(tab) == 2 and set(tab["better"]) == {"good"}


def test_bootstrap_point_matches_wape_and_ci_contains_point():
    df = _frame()
    res = block_bootstrap_wape_diff(df, "bad", "good", group_col="date_type", n_boot=500, seed=1)
    assert set(res["group"]) == {"Weekday", "Weekend"}
    for _, r in res.iterrows():
        sub = df[df["date_type"] == r["group"]]
        expect = 100 * (wape(sub["demand"], sub["good"]) - wape(sub["demand"], sub["bad"]))
        assert r["diff b-a (pp)"] == pytest.approx(expect, abs=0.01)
        assert r["CI low"] <= r["diff b-a (pp)"] <= r["CI high"]
        assert r["CI excludes 0"] and r["diff b-a (pp)"] < 0
    overall = block_bootstrap_wape_diff(df, "good", "same", n_boot=200)
    assert overall.loc[0, "diff b-a (pp)"] == 0 and overall.loc[0, "CI low"] == 0


def test_bootstrap_is_reproducible():
    df = _frame()
    a = block_bootstrap_wape_diff(df, "bad", "good", n_boot=300, seed=7)
    b = block_bootstrap_wape_diff(df, "bad", "good", n_boot=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_permutation_importance_frame_finds_the_useful_group():
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({"x_useful": rng.normal(size=n), "x_noise": rng.normal(size=n),
                      "cat": pd.Categorical(rng.choice(["a", "b"], n), categories=["a", "b"])})
    y = np.exp(0.5 * X["x_useful"]).to_numpy()
    predict = lambda Z: np.exp(0.5 * Z["x_useful"].to_numpy())   # mô hình "biết" x_useful
    res = permutation_importance(predict, X, y, {"useful": permute_frame_columns(["x_useful"]),
                                                 "noise": permute_frame_columns(["x_noise", "cat"])},
                                 n_repeats=2, seed=0)
    assert res.iloc[0]["group"] == "useful" and res.iloc[0]["increase (pp)"] > 10
    assert res.set_index("group").loc["noise", "increase (pp)"] == pytest.approx(0, abs=1e-9)
    assert X["cat"].dtype == pd.CategoricalDtype(["a", "b"])        # đầu vào gốc không bị đổi


def test_permute_array_channels_keeps_other_channels_and_shape():
    rng = np.random.default_rng(0)
    inputs = {"hist": rng.normal(size=(50, 28, 4)).astype(np.float32),
              "future": rng.normal(size=(50, 1, 3)).astype(np.float32),
              "item": np.arange(50).reshape(-1, 1)}
    perm = permute_array_channels({"hist": [1, 2], "future": [0]})
    out = perm(inputs, np.random.default_rng(1))
    assert out["hist"].shape == inputs["hist"].shape
    np.testing.assert_array_equal(out["hist"][..., [0, 3]], inputs["hist"][..., [0, 3]])   # kênh khác giữ nguyên
    np.testing.assert_array_equal(out["item"], inputs["item"])
    # cùng một thứ tự hoán vị cho mọi mảng trong nhóm
    order = [int(np.where((inputs["hist"][:, :, 1] == out["hist"][k, :, 1]).all(axis=1))[0][0]) for k in range(50)]
    np.testing.assert_array_equal(out["future"][:, :, 0], inputs["future"][order][:, :, 0])
    assert not np.array_equal(out["hist"], inputs["hist"])
    whole = permute_array_channels({"item": None})(inputs, np.random.default_rng(2))
    assert sorted(whole["item"].ravel()) == list(range(50))


def test_timer_and_ensemble():
    with Timer() as t:
        sum(range(1000))
    assert t.seconds >= 0
    df = pd.DataFrame({"a": [1.0, 3.0], "b": [3.0, 5.0]})
    add_ensemble(df, ["a", "b"], "ens")
    assert df["ens"].tolist() == [2.0, 4.0]
    with pytest.raises(KeyError):
        add_ensemble(df, ["a", "zzz"], "x")
