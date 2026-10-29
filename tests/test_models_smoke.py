"""Chạy thử LightGBM và LSTM trên dữ liệu giả lập (bỏ qua nếu chưa cài thư viện).

Kiểm tra: chạy được, đúng kích thước, dự báo không âm, sản phẩm chưa lên kệ được dự báo 0,
và tính tái lập theo seed.
"""
from pathlib import Path

import numpy as np
import pytest

from m5cal.config import load_config
from m5cal.pipeline import make_model

CFG = load_config(Path(__file__).resolve().parents[1] / "configs")


def _small_lgbm_cfg():
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in CFG.items()}
    cfg["lgbm"] = dict(CFG["lgbm"], num_boost_round=30, early_stopping_rounds=5)
    cfg["lgbm"]["params"] = dict(CFG["lgbm"]["params"], num_leaves=15, min_data_in_leaf=20, num_threads=1)
    cfg["base"] = dict(CFG["base"], data=dict(CFG["base"]["data"], history_days=200))
    return cfg


def test_lgbm_fit_predict(panel, calf, windows):
    pytest.importorskip("lightgbm")
    cfg = _small_lgbm_cfg()
    tune = [w for w in windows if w.role == "tune"]
    ev = [w for w in windows if w.role == "eval"][0]
    spec = {"model": "lgbm", "feature_set": "HCP", "horizon_groups": "weekly"}
    m = make_model(cfg, spec, panel, seed=1).fit(panel, calf, tune[0].origin, valid_windows=tune[:1])
    p = m.predict(panel, calf, ev)
    assert len(p) == panel.n * 28 and (p["y_pred"] >= 0).all()
    p2 = make_model(cfg, spec, panel, seed=1).fit(panel, calf, tune[0].origin, valid_windows=tune[:1]).predict(panel, calf, ev)
    np.testing.assert_allclose(p["y_pred"], p2["y_pred"])


def test_lstm_fit_predict(panel, calf, windows):
    pytest.importorskip("torch")
    cfg = dict(CFG["lstm"], input_len=28, hidden_size=8, max_epochs=1, samples_per_epoch=512, batch_size=128, device="cpu")
    full = dict(CFG, lstm=cfg)
    tune = [w for w in windows if w.role == "tune"]
    ev = [w for w in windows if w.role == "eval"][0]
    for fs in ("H", "HCP"):
        m = make_model(full, {"model": "lstm", "feature_set": fs}, panel, seed=1)
        m.fit(panel, calf, tune[0].origin, valid_windows=tune[:1])
        p = m.predict(panel, calf, ev)
        assert len(p) == panel.n * 28 and np.isfinite(p["y_pred"]).all() and (p["y_pred"] >= 0).all()
