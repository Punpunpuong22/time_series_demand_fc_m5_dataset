"""LightGBM toàn cục, dự báo trực tiếp theo nhóm tầm (L-H, L-HC, L-HCP, R2).

Mỗi nhóm tầm [a, b] có một booster riêng với độ trễ tối thiểu b, nên tại mốc T
mô hình chỉ dùng doanh số ≤ T (xem features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import CATEGORICAL_FEATURES, HORIZON_GROUPS, feature_columns, feature_group_of, lgbm_frame
from ..splits import assert_train_before


def _lgb():
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần cài lightgbm: pip install lightgbm (Colab đã cài sẵn)") from e
    return lgb


class LGBMDirect:
    def __init__(self, params: dict, feature_set: str, horizon_groups: str = "weekly",
                 num_boost_round: int = 2000, early_stopping_rounds: int = 100,
                 history_days: int = 1095, price_mode: str = "known", seed: int = 1):
        self.params = dict(params)
        self.params.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed})
        self.feature_set = feature_set
        self.groups = HORIZON_GROUPS[horizon_groups]
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.history_days = history_days
        self.price_mode = price_mode
        self.boosters: dict = {}
        self.best_iter: dict = {}
        self.features: dict = {}

    # --------------------------------------------------------------- training
    def _train_frame(self, panel, calf, cutoff, a, b):
        start = max(cutoff - self.history_days + 1, b + 1)
        days = np.arange(start, cutoff + 1)
        assert_train_before(cutoff, days)
        df = lgbm_frame(panel, calf, cutoff, b, days, self.feature_set, self.price_mode)
        df = df[df["avail"] & df["y"].notna() & df["lag_p0"].notna()] if "lag_p0" in df else df[df["avail"]]
        return df

    def _window_frame(self, panel, calf, window, a, b):
        days = window.origin + np.arange(a, b + 1)
        return lgbm_frame(panel, calf, window.origin, b, days, self.feature_set, self.price_mode)

    def fit(self, panel, calf, cutoff: int, valid_windows=None, rounds: dict | None = None, log=None):
        """Huấn luyện mọi nhóm tầm với dữ liệu ≤ cutoff.

        valid_windows: các cửa sổ tinh chỉnh để early stopping (giai đoạn tune).
        rounds: số vòng cố định cho từng nhóm (giai đoạn đánh giá, lấy từ tune).
        """
        lgb = _lgb()
        for a, b in self.groups:
            tr = self._train_frame(panel, calf, cutoff, a, b)
            feats = feature_columns(tr)
            cats = [c for c in CATEGORICAL_FEATURES if c in feats]
            dtrain = lgb.Dataset(tr[feats], label=tr["y"], categorical_feature=cats, free_raw_data=True)
            valid_sets, callbacks = [], []
            n_round = self.num_boost_round if rounds is None else int(rounds[(a, b)])
            if valid_windows:
                va = pd.concat([self._window_frame(panel, calf, w, a, b) for w in valid_windows])
                va = va[va["avail"]]
                dvalid = lgb.Dataset(va[feats], label=va["y"], categorical_feature=cats, reference=dtrain)
                valid_sets = [dvalid]
                callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            booster = lgb.train(self.params, dtrain, num_boost_round=n_round,
                                valid_sets=valid_sets or None, callbacks=callbacks)
            self.boosters[(a, b)] = booster
            self.best_iter[(a, b)] = booster.best_iteration if valid_windows and booster.best_iteration else n_round
            self.features[(a, b)] = feats
            if log:
                log.info(f"  LightGBM nhóm h={a}-{b}: {len(tr):,} dòng, {self.best_iter[(a, b)]} vòng")
        return self

    # --------------------------------------------------------------- prediction
    def predict(self, panel, calf, window, permute: str | None = None, rng=None,
                return_contrib: bool = False) -> pd.DataFrame:
        out = []
        for a, b in self.groups:
            df = self._window_frame(panel, calf, window, a, b)
            feats = self.features[(a, b)]
            X = df[feats].copy()
            if permute is not None:
                rng = rng or np.random.default_rng(0)
                cols = [c for c in feats if feature_group_of(c) == permute]
                if cols:
                    perm = rng.permutation(len(X))
                    X[cols] = X[cols].to_numpy()[perm]
            booster = self.boosters[(a, b)]
            pred = booster.predict(X, num_iteration=self.best_iter[(a, b)])
            res = pd.DataFrame({"sid": df["sid"].to_numpy(), "d": df["d"].to_numpy(),
                                "y_pred": np.where(df["avail"], np.clip(pred, 0, None), 0.0)})
            if return_contrib:
                contrib = booster.predict(X, num_iteration=self.best_iter[(a, b)], pred_contrib=True)
                res = pd.concat([res, pd.DataFrame(contrib[:, :-1], columns=feats)], axis=1)
            out.append(res)
        return pd.concat(out, ignore_index=True)


def tune_lgbm(panel, calf, tune_windows, base_params: dict, space: dict, feature_set: str,
              horizon_groups: str, n_trials: int, history_days: int, score_fn, seed: int = 1, log=None):
    """Tìm siêu tham số bằng Optuna; điểm = score_fn(dự báo trên các cửa sổ tinh chỉnh) (càng thấp càng tốt)."""
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần cài optuna: pip install optuna") from e

    def objective(trial):
        params = dict(base_params)
        for name, (lo, hi, kind) in space.items():
            if kind == "log":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            elif kind == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)
        m = LGBMDirect(params, feature_set, horizon_groups, history_days=history_days, seed=seed)
        m.fit(panel, calf, tune_windows[0].origin, valid_windows=tune_windows[:1])
        preds = pd.concat([m.predict(panel, calf, w).assign(window=w.name) for w in tune_windows])
        return score_fn(preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(base_params)
    best.update(study.best_params)
    if log:
        log.info(f"Optuna LightGBM: best={study.best_value:.4f} params={study.best_params}")
    return best, study
