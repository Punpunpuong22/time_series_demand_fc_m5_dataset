"""Mô phỏng tồn kho tối giản (mục 7.8): chính sách order-up-to xét duyệt hằng ngày.

Mỗi ngày t: nhận hàng đến hạn → đáp ứng nhu cầu (phần thiếu là mất doanh số) →
đặt hàng q_t = max(0, S_t − vị thế tồn kho), hàng về sau `lead_time` ngày.
S_t = tổng dự báo của t+1 … t+lead_time+1 + z · σ · sqrt(lead_time + 1).
Đây là so sánh TƯƠNG ĐỐI giữa các mô hình dưới giả định đơn giản, không phải chi phí thực.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_order_up_to(demand: np.ndarray, forecast: np.ndarray, sigma: np.ndarray,
                         lead_time: int = 2, z: float = 1.645) -> dict:
    """demand, forecast: [n, H]; sigma: [n] độ lệch chuẩn sai số dự báo ngày (ước lượng trên tinh chỉnh)."""
    demand = np.asarray(demand, float)
    forecast = np.asarray(forecast, float)
    n, H = demand.shape
    cover = lead_time + 1
    tail = forecast[:, -7:].mean(axis=1, keepdims=True)
    f_ext = np.concatenate([forecast, np.repeat(tail, cover + 1, axis=1)], axis=1)
    safety = z * np.asarray(sigma, float) * np.sqrt(cover)

    S0 = f_ext[:, :cover].sum(axis=1) + safety
    on_hand = S0.copy()
    pipeline = np.zeros((n, lead_time + 1))   # pipeline[:, k] về sau k ngày nữa
    served = np.zeros((n, H)); end_inv = np.zeros((n, H)); short = np.zeros((n, H))
    for t in range(H):
        on_hand += pipeline[:, 0]
        pipeline = np.concatenate([pipeline[:, 1:], np.zeros((n, 1))], axis=1)
        s = np.minimum(on_hand, demand[:, t])
        served[:, t] = s
        short[:, t] = demand[:, t] - s
        on_hand -= s
        end_inv[:, t] = on_hand
        S_t = f_ext[:, t + 1: t + 1 + cover].sum(axis=1) + safety
        position = on_hand + pipeline.sum(axis=1)
        q = np.maximum(0.0, S_t - position)
        pipeline[:, lead_time - 1 if lead_time > 0 else 0] += q
    return {"served": served, "short": short, "end_inventory": end_inv}


def summarize_inventory(sim: dict, demand: np.ndarray, labels: np.ndarray | None = None) -> pd.DataFrame:
    """Fill rate, tồn kho trung bình, tỷ lệ ngày hết hàng; theo từng loại ngày nếu có labels [n, H]."""
    demand = np.asarray(demand, float)
    stockout = (sim["short"] > 1e-9)

    def _row(mask):
        d = demand[mask].sum()
        return {
            "fill_rate": float(sim["served"][mask].sum() / d) if d > 0 else np.nan,
            "avg_inventory": float(sim["end_inventory"][mask].mean()),
            "stockout_day_share": float(stockout[mask].mean()),
            "n_obs": int(mask.sum()),
        }

    rows = [{"daytype": "all", **_row(np.ones_like(demand, bool))}]
    if labels is not None:
        for lab in np.unique(labels):
            rows.append({"daytype": lab, **_row(labels == lab)})
    return pd.DataFrame(rows)
