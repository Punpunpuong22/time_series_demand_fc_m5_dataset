"""Phân nhóm nhu cầu theo ADI/CV² (Syntetos, Boylan & Croston, 2005) và lấy mẫu phân tầng.

ADI và CV² chỉ được tính trên dữ liệu đến `until_d` (giai đoạn huấn luyện ban đầu)
để việc chọn mẫu không dùng thông tin của giai đoạn đánh giá.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT = 1.32
CV2_CUT = 0.49


def demand_classes(panel, until_d: int) -> pd.DataFrame:
    """Trả về DataFrame: adi, cv2, demand_class, zero_share cho từng chuỗi (tính trên d ≤ until_d)."""
    Y = panel.Y[:, :until_d]
    launch = panel.launch_day()
    rows = []
    for i in range(panel.n):
        start = max(int(launch[i]), 1) - 1
        y = Y[i, start:]
        nz = y[y > 0]
        n_days = len(y)
        if len(nz) == 0 or n_days == 0:
            rows.append((np.inf, np.nan, "no_sales", 1.0))
            continue
        # ADI = số ngày từ lần bán đầu tiên / số ngày có bán
        first_sale = int(np.argmax(y > 0))
        adi = (n_days - first_sale) / len(nz)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        if adi < ADI_CUT:
            cls = "smooth" if cv2 < CV2_CUT else "erratic"
        else:
            cls = "intermittent" if cv2 < CV2_CUT else "lumpy"
        rows.append((adi, cv2, cls, float((y == 0).mean())))
    return pd.DataFrame(rows, columns=["adi", "cv2", "demand_class", "zero_share"])


def stratified_sample(meta: pd.DataFrame, n: int | None, by: list[str], seed: int = 2026) -> np.ndarray:
    """Chỉ số các chuỗi được chọn, phân bổ theo tỷ lệ của từng tầng (ít nhất 1 chuỗi/tầng)."""
    if n is None or n >= len(meta):
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    groups = meta.groupby(by, observed=True).indices
    total = len(meta)
    chosen = []
    for _, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
        k = max(1, int(round(n * len(idx) / total)))
        k = min(k, len(idx))
        chosen.extend(rng.choice(idx, size=k, replace=False).tolist())
    chosen = np.array(sorted(chosen))
    if len(chosen) > n:  # cắt bớt phần dư do làm tròn, vẫn ngẫu nhiên có seed
        chosen = np.sort(rng.choice(chosen, size=n, replace=False))
    return chosen
