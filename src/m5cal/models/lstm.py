"""LSTM encoder–decoder toàn cục, dự báo trực tiếp 28 giá trị (S-H, S-HC, S-HCP).

Encoder đọc L ngày quá khứ (doanh số đã chuẩn hóa theo chuỗi + biến biết trước của các ngày đó);
decoder nhận biến biết trước của 28 ngày tới; trạng thái ẩn của encoder khởi tạo decoder.
Định danh phân cấp đi qua lớp embedding. Không có bước đệ quy dùng dự báo của chính mô hình.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from ..data import CAT_COLS
from ..features import SeqBuilder


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần cài PyTorch: pip install torch (Colab đã cài sẵn)") from e
    return torch


def build_network(cardinalities: dict, emb_dims: dict, n_channels: int, hidden_size: int,
                  num_layers: int, dropout: float, horizon: int):
    torch = _torch()
    nn = torch.nn

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList(
                [nn.Embedding(cardinalities[c] + 1, emb_dims.get(c, 4)) for c in CAT_COLS]
            )
            static = sum(emb_dims.get(c, 4) for c in CAT_COLS)
            drop = dropout if num_layers > 1 else 0.0
            self.encoder = nn.LSTM(2 + n_channels + static, hidden_size, num_layers, batch_first=True, dropout=drop)
            self.decoder = nn.LSTM(1 + n_channels + static, hidden_size, num_layers, batch_first=True, dropout=drop)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)
            self.horizon = horizon

        def forward(self, y_past, past_mask, c_past, c_fut, cats):
            s = torch.cat([emb(cats[:, k]) for k, emb in enumerate(self.embs)], dim=1)
            L = y_past.shape[1]
            enc_in = torch.cat([y_past.unsqueeze(-1), past_mask.unsqueeze(-1), c_past,
                                s.unsqueeze(1).expand(-1, L, -1)], dim=2)
            _, state = self.encoder(enc_in)
            H = c_fut.shape[1]
            ones = torch.ones(c_fut.shape[0], H, 1, device=c_fut.device)
            dec_in = torch.cat([ones, c_fut, s.unsqueeze(1).expand(-1, H, -1)], dim=2)
            out, _ = self.decoder(dec_in, state)
            return self.head(self.dropout(out)).squeeze(-1)   # [B, H]: log-mean (tweedie) hoặc giá trị (mse)

    return Seq2Seq()


def tweedie_loss(f, y, mask, rho: float):
    """Âm log-likelihood Tweedie (bỏ hằng số), f = log(mu)."""
    torch = _torch()
    loss = -y * torch.exp(f * (1 - rho)) / (1 - rho) + torch.exp(f * (2 - rho)) / (2 - rho)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class LSTMForecaster:
    def __init__(self, cfg: dict, feature_set: str, cardinalities: dict, seed: int = 1, price_mode: str = "known"):
        self.cfg = cfg
        self.feature_set = feature_set
        self.cardinalities = cardinalities
        self.seed = seed
        self.price_mode = price_mode
        self.net = None
        self.builder = None
        self.best_epochs = None
        torch = _torch()
        dev = cfg.get("device", "auto")
        self.device = torch.device("cuda" if (dev == "auto" and torch.cuda.is_available()) else ("cpu" if dev == "auto" else dev))

    # ------------------------------------------------------------------ utils
    def _to_tensors(self, b: dict):
        torch = _torch()
        t = {k: torch.as_tensor(v, device=self.device) for k, v in b.items() if k not in ("rows", "origins")}
        t["cats"] = t["cats"].long()
        return t

    def _forward(self, t):
        return self.net(t["y_past"], t["past_mask"], t["c_past"], t["c_fut"], t["cats"])

    def _loss(self, f, t):
        if self.cfg.get("loss", "tweedie") == "tweedie":
            return tweedie_loss(f, t["y_fut"], t["fut_mask"], float(self.cfg.get("tweedie_rho", 1.5)))
        torch = _torch()
        pred = torch.nn.functional.softplus(f)
        return (((pred - t["y_fut"]) ** 2) * t["fut_mask"]).sum() / t["fut_mask"].sum().clamp(min=1.0)

    def _point(self, f):
        torch = _torch()
        return torch.exp(f) if self.cfg.get("loss", "tweedie") == "tweedie" else torch.nn.functional.softplus(f)

    def _valid_loss(self, windows) -> float:
        torch = _torch()
        self.net.eval()
        tot, cnt = 0.0, 0
        bs = int(self.cfg.get("batch_size", 512))
        with torch.no_grad():
            for w in windows:
                rows = np.arange(self.builder.panel.n)
                for s in range(0, len(rows), bs):
                    r = rows[s: s + bs]
                    b = self.builder.batch(r, np.full(len(r), w.origin))
                    t = self._to_tensors(b)
                    tot += float(self._loss(self._forward(t), t)) * len(r)
                    cnt += len(r)
        return tot / max(cnt, 1)

    # ------------------------------------------------------------------ fit
    def fit(self, panel, calf, cutoff: int, valid_windows=None, epochs: int | None = None, log=None):
        torch = _torch()
        from ..utils import set_seed

        set_seed(self.seed)
        cfg = self.cfg
        self.builder = SeqBuilder(panel, calf, self.feature_set, cutoff, int(cfg["input_len"]), 28, self.price_mode)
        self.net = build_network(self.cardinalities, cfg.get("emb_dims", {}), self.builder.n_channels,
                                 int(cfg["hidden_size"]), int(cfg["num_layers"]), float(cfg["dropout"]), 28).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg["lr"]))
        rng = np.random.default_rng(self.seed)
        n_epochs = int(epochs or cfg["max_epochs"])
        bs = int(cfg["batch_size"])
        spe = int(cfg.get("samples_per_epoch", 100_000))
        best, best_state, best_epoch, bad = np.inf, None, n_epochs, 0
        for ep in range(1, n_epochs + 1):
            t0 = time.time()
            self.net.train()
            rows, origins = self.builder.sample_train_pairs(spe, float(cfg.get("train_years", 3)), rng)
            losses = []
            for s in range(0, spe, bs):
                b = self.builder.batch(rows[s: s + bs], origins[s: s + bs])
                t = self._to_tensors(b)
                opt.zero_grad()
                loss = self._loss(self._forward(t), t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(cfg.get("grad_clip", 1.0)))
                opt.step()
                losses.append(float(loss))
            msg = f"  LSTM epoch {ep}: train={np.mean(losses):.4f}"
            if valid_windows:
                vl = self._valid_loss(valid_windows)
                msg += f" valid={vl:.4f}"
                if vl < best - 1e-5:
                    best, best_state, best_epoch, bad = vl, copy.deepcopy(self.net.state_dict()), ep, 0
                else:
                    bad += 1
                    if bad >= int(cfg.get("patience", 3)):
                        if log:
                            log.info(msg + " -> early stopping")
                        break
            if log:
                log.info(msg + f" ({time.time() - t0:.0f}s)")
        if valid_windows and best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_epochs = best_epoch
        else:
            self.best_epochs = n_epochs
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, panel, calf, window, permute: str | None = None, rng=None) -> pd.DataFrame:
        torch = _torch()
        self.net.eval()
        bs = int(self.cfg.get("batch_size", 512))
        rows_all = np.arange(panel.n)
        preds = []
        with torch.no_grad():
            for s in range(0, len(rows_all), bs):
                r = rows_all[s: s + bs]
                b = self.builder.batch(r, np.full(len(r), window.origin), with_target=False,
                                       permute=permute, rng=rng)
                t = self._to_tensors(b)
                p = self._point(self._forward(t)).cpu().numpy() * b["scale"][:, None]
                preds.append(p)
        P = np.concatenate(preds, axis=0)[:, : len(window.days)]
        days = window.days
        P = np.where(panel.avail[:, days - 1], P, 0.0)
        return pd.DataFrame({"sid": np.repeat(rows_all, len(days)), "d": np.tile(days, panel.n),
                             "y_pred": np.clip(P, 0, None).ravel()})


def tune_lstm(panel, calf, tune_windows, base_cfg: dict, space: dict, feature_set: str, n_trials: int,
              score_fn, seed: int = 1, log=None):
    """Tìm siêu tham số LSTM bằng Optuna với cùng ngân sách và cùng cửa sổ tinh chỉnh như LightGBM.

    space: {tên: [thấp, cao, kiểu]} với kiểu 'log' | 'int' | 'float' | 'choice' (khi 'choice', hai giá trị đầu là các lựa chọn).
    """
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần cài optuna: pip install optuna") from e

    def objective(trial):
        cfg = dict(base_cfg)
        for name, (lo, hi, kind) in space.items():
            if kind == "log":
                cfg[name] = trial.suggest_float(name, lo, hi, log=True)
            elif kind == "int":
                cfg[name] = trial.suggest_int(name, lo, hi)
            elif kind == "choice":
                cfg[name] = trial.suggest_categorical(name, [lo, hi])
            else:
                cfg[name] = trial.suggest_float(name, lo, hi)
        m = LSTMForecaster(cfg, feature_set, panel.cardinalities, seed)
        m.fit(panel, calf, tune_windows[0].origin, valid_windows=tune_windows[:1])
        preds = pd.concat([m.predict(panel, calf, w).assign(window=w.name) for w in tune_windows])
        return score_fn(preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(base_cfg)
    best.update(study.best_params)
    if log:
        log.info(f"Optuna LSTM: best={study.best_value:.4f} params={study.best_params}")
    return best, study
