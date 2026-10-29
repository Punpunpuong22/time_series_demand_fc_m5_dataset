"""R1 — DeepAR (Salinas et al., 2020) qua GluonTS, dùng làm kiểm tra độ bền cho nhánh hồi quy.

THỬ NGHIỆM: cần `pip install "gluonts[torch]"`. API của GluonTS có thể thay đổi giữa các
phiên bản; mã này viết cho gluonts >= 0.14 (gluonts.torch). Nếu lỗi, ghi lại phiên bản và
xem tài liệu GluonTS. Không bắt buộc cho các thí nghiệm chính (B0–S-HCP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import CAT_COLS
from ..features import SeqBuilder
from ..utils import d_to_date


class DeepARForecaster:
    def __init__(self, cfg: dict, feature_set: str, cardinalities: dict, seed: int = 1):
        self.cfg = cfg
        self.feature_set = feature_set
        self.cardinalities = cardinalities
        self.seed = seed
        self.predictor = None

    def _dataset(self, panel, calf, upto: int, cov_upto: int):
        b = SeqBuilder(panel, calf, self.feature_set, cutoff=upto, input_len=1, horizon=1)
        days = np.arange(1, cov_upto + 1)
        launch = panel.launch_day()
        data = []
        for i in range(panel.n):
            start = max(int(launch[i]), 1)
            cols = days[start - 1:] - 1
            cov = b._cov(np.array([i]), cols[None, :], np.array([upto]))[0].T  # [c, len]
            entry = {
                "start": pd.Period(d_to_date(start), freq="D"),
                "target": panel.Y[i, start - 1: upto].astype(np.float32),
                "feat_static_cat": [int(panel.meta[c + "_code"].iloc[i]) for c in CAT_COLS],
            }
            if cov.shape[0] > 0:
                entry["feat_dynamic_real"] = cov.astype(np.float32)
            data.append(entry)
        return data, b.n_channels

    def fit(self, panel, calf, cutoff: int, valid_windows=None, epochs: int | None = None, log=None):
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.torch.distributions import NegativeBinomialOutput
            from gluonts.torch.model.deepar import DeepAREstimator
        except ImportError as e:  # pragma: no cover
            raise ImportError('R1 cần GluonTS: pip install "gluonts[torch]"') from e
        data, n_dyn = self._dataset(panel, calf, cutoff, cutoff)
        est = DeepAREstimator(
            freq="D", prediction_length=28, context_length=int(self.cfg["input_len"]),
            num_layers=2, hidden_size=int(self.cfg["hidden_size"]), lr=float(self.cfg["lr"]),
            num_feat_dynamic_real=n_dyn, num_feat_static_cat=len(CAT_COLS),
            cardinality=[self.cardinalities[c] + 1 for c in CAT_COLS],
            distr_output=NegativeBinomialOutput(), batch_size=int(self.cfg["batch_size"]),
            trainer_kwargs={"max_epochs": int(epochs or self.cfg["max_epochs"]), "enable_progress_bar": False},
        )
        self.predictor = est.train(ListDataset(data, freq="D"))
        self.best_epochs = int(epochs or self.cfg["max_epochs"])
        return self

    def predict(self, panel, calf, window, permute=None, rng=None) -> pd.DataFrame:
        from gluonts.dataset.common import ListDataset

        data, _ = self._dataset(panel, calf, window.origin, window.end)
        fcsts = list(self.predictor.predict(ListDataset(data, freq="D"), num_samples=100))
        P = np.stack([f.mean[: len(window.days)] for f in fcsts])
        days = window.days
        P = np.where(panel.avail[:, days - 1], P, 0.0)
        return pd.DataFrame({"sid": np.repeat(np.arange(panel.n), len(days)), "d": np.tile(days, panel.n),
                             "y_pred": np.clip(P, 0, None).ravel()})
