"""Điều phối thí nghiệm: chuẩn bị dữ liệu, chạy một cấu hình trên mọi cửa sổ, đánh giá.

Được gọi bởi experiments/run.py, experiments/evaluate.py và các notebook Colab.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import experiment_spec
from .daytypes import calendar_features, daytype_matrix
from .data import build_panel, load_raw
from .metrics import aggregate_scores, aggregate_stratified, attach_truth, series_window_scores, stratified_scores
from .sampling import demand_classes, stratified_sample
from .splits import training_cutoff, windows_from_config
from .utils import environment_info, get_logger, save_json, set_seed

LOG = get_logger()


# ----------------------------------------------------------------------------- data
def prepare(cfg: dict, raw_dir: str | None = None, n_series: int | None = -1, out_dir: str | None = None):
    """Đọc dữ liệu, phân nhóm nhu cầu, lấy mẫu phân tầng (lưu danh sách id để tái lập)."""
    base = cfg["base"]
    raw_dir = raw_dir or base["paths"]["raw_dir"]
    out_dir = Path(out_dir or base["paths"]["out_dir"])
    n_series = base["data"]["n_series"] if n_series == -1 else n_series

    LOG.info(f"Đọc dữ liệu từ {raw_dir}")
    raw = load_raw(raw_dir)
    panel = build_panel(raw)
    classes = demand_classes(panel, int(base["data"]["classify_until_d"]))
    panel.meta = pd.concat([panel.meta, classes], axis=1)

    ids_file = out_dir / "series_ids.csv"
    if ids_file.exists():
        ids = pd.read_csv(ids_file)["id"]
        idx = np.flatnonzero(panel.meta["id"].isin(set(ids)).to_numpy())
        LOG.info(f"Dùng lại {len(idx)} chuỗi từ {ids_file}")
    else:
        eligible = np.flatnonzero((panel.meta["demand_class"] != "no_sales").to_numpy())
        sub = stratified_sample(panel.meta.iloc[eligible].reset_index(drop=True), n_series,
                                base["data"]["stratify_by"], int(base["data"]["sample_seed"]))
        idx = eligible[sub]
        ids_file.parent.mkdir(parents=True, exist_ok=True)
        panel.meta.iloc[idx][["id", "store_id", "cat_id", "demand_class"]].to_csv(ids_file, index=False)
        LOG.info(f"Lấy mẫu phân tầng {len(idx)} chuỗi, lưu tại {ids_file}")
    panel = panel.subset(idx)
    calf = calendar_features(panel.calendar, int(base["daytypes"]["pre_event_days"]))
    windows = windows_from_config(base["windows"], max_day=panel.n_days)
    return panel, calf, windows


# ----------------------------------------------------------------------------- models
def make_model(cfg: dict, spec: dict, panel, seed: int):
    m = spec["model"]
    price_mode = spec.get("price_mode", "known")
    if m == "lgbm":
        from .models.lgbm import LGBMDirect

        lc = cfg["lgbm"]
        return LGBMDirect(lc["params"], spec["feature_set"], spec.get("horizon_groups", "weekly"),
                          lc["num_boost_round"], lc["early_stopping_rounds"],
                          cfg["base"]["data"]["history_days"], price_mode, seed)
    if m == "lstm":
        from .models.lstm import LSTMForecaster

        return LSTMForecaster(cfg["lstm"], spec["feature_set"], panel.cardinalities, seed, price_mode)
    if m == "deepar":
        from .models.deepar import DeepARForecaster

        return DeepARForecaster(cfg["lstm"], spec["feature_set"], panel.cardinalities, seed)
    raise ValueError(m)


def run_experiment(cfg: dict, code: str, seed: int, panel, calf, windows, out_dir: str | Path = "results",
                   max_eval: int | None = None, include_tune: bool = True) -> pd.DataFrame:
    """Chạy một cấu hình trên các cửa sổ; lưu và trả về bảng dự báo (sid, d, window, role, y_pred, y_true)."""
    spec = experiment_spec(cfg, code)
    set_seed(seed)
    out_dir = Path(out_dir)
    tune_w = [w for w in windows if w.role == "tune"]
    eval_w = [w for w in windows if w.role != "tune"]
    if max_eval:
        eval_w = eval_w[:max_eval]
    t0 = time.time()
    preds = []

    if spec["model"] in ("snaive", "croston_sba"):
        from .models.baselines import croston_sba, seasonal_naive

        fn = seasonal_naive if spec["model"] == "snaive" else croston_sba
        for w in (tune_w if include_tune else []) + eval_w:
            preds.append(fn(panel, w).assign(window=w.name, role=w.role))
        meta = {}
    else:
        model = make_model(cfg, spec, panel, seed)
        LOG.info(f"[{code} seed={seed}] Huấn luyện tại d_{tune_w[0].origin}, early stopping trên cửa sổ tinh chỉnh")
        model.fit(panel, calf, tune_w[0].origin, valid_windows=tune_w[:1], log=LOG)
        rounds = getattr(model, "best_iter", None)
        epochs = getattr(model, "best_epochs", None)
        if include_tune:
            for w in tune_w:
                preds.append(model.predict(panel, calf, w).assign(window=w.name, role=w.role))
        for w in eval_w:
            if w.retrain:
                LOG.info(f"[{code} seed={seed}] Huấn luyện lại tại d_{w.origin}")
                model = make_model(cfg, spec, panel, seed)
                if spec["model"] == "lgbm":
                    model.fit(panel, calf, w.origin, rounds=rounds, log=LOG)
                else:
                    model.fit(panel, calf, w.origin, epochs=epochs, log=LOG)
            assert training_cutoff(windows, w) <= w.origin
            preds.append(model.predict(panel, calf, w).assign(window=w.name, role=w.role))
        meta = {"best_rounds": {str(k): v for k, v in (rounds or {}).items()}, "best_epochs": epochs}

    df = attach_truth(panel, pd.concat(preds, ignore_index=True))
    df["exp"], df["seed"] = code, seed
    path = out_dir / "preds" / f"{code}__seed{seed}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    save_json({"exp": spec, "seed": seed, "seconds": round(time.time() - t0, 1), **meta,
               "n_series": panel.n, "env": environment_info()}, out_dir / "runs" / f"{code}__seed{seed}.json")
    LOG.info(f"[{code} seed={seed}] Xong sau {time.time() - t0:.0f}s -> {path}")
    return df


# ----------------------------------------------------------------------------- evaluation
def load_predictions(out_dir: str | Path = "results") -> pd.DataFrame:
    files = sorted(Path(out_dir, "preds").glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"Không có dự báo trong {out_dir}/preds. Hãy chạy experiments/run.py trước.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def daytype_table(panel, calf, windows) -> pd.DataFrame:
    days = np.concatenate([w.days for w in windows])
    lab = daytype_matrix(panel, calf, days)
    return pd.DataFrame({"sid": np.repeat(np.arange(panel.n), len(days)), "d": np.tile(days, panel.n),
                         "daytype": lab.ravel()})


def evaluate(panel, calf, windows, preds: pd.DataFrame, roles=("eval", "test")) -> dict:
    """Bảng tổng hợp: theo cửa sổ, tổng thể, theo loại ngày, theo nhóm nhu cầu."""
    preds = preds[preds["role"].isin(roles)]
    labels = daytype_table(panel, calf, [w for w in windows if w.role in roles])
    tables = {"by_window": [], "series": [], "by_daytype": [], "strat_series": []}
    for (exp, seed), g in preds.groupby(["exp", "seed"]):
        sc = series_window_scores(panel, g, windows)
        sc["exp"], sc["seed"] = exp, seed
        tables["series"].append(sc)
        agg = aggregate_scores(sc)
        agg["exp"], agg["seed"] = exp, seed
        tables["by_window"].append(agg)
        st = stratified_scores(panel, g, windows, labels)
        st["exp"], st["seed"] = exp, seed
        tables["strat_series"].append(st)
        sa = aggregate_stratified(st)
        sa["exp"], sa["seed"] = exp, seed
        tables["by_daytype"].append(sa)
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    bw = out["by_window"]
    overall = bw.groupby(["exp", "seed"])[["wrmsse", "rmsse_mean", "mae", "rmse", "bias"]].mean().reset_index()
    out["overall"] = overall.groupby("exp").agg(
        wrmsse=("wrmsse", "mean"), wrmsse_sd=("wrmsse", "std"), rmsse_mean=("rmsse_mean", "mean"),
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"), n_seeds=("seed", "nunique"),
    ).reset_index().sort_values("wrmsse")
    s = out["series"].merge(panel.meta[["demand_class"]].reset_index().rename(columns={"index": "sid"}), on="sid")
    out["by_demand_class"] = s.groupby(["exp", "demand_class"])["rmsse"].mean().unstack().reset_index()
    return out
