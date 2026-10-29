"""Tiện ích dùng chung: seed, log, chuyển đổi chỉ số ngày, lưu kết quả."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np

M5_DAY1 = date(2011, 1, 29)  # d_1


class LeakageError(RuntimeError):
    """Được ném ra khi một phép tính có nguy cơ dùng thông tin sau mốc dự báo."""


def d_to_date(d: int) -> date:
    """Chuyển chỉ số ngày M5 (d_1 = 29/01/2011) sang ngày dương lịch."""
    return M5_DAY1 + timedelta(days=int(d) - 1)


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy và (nếu có) PyTorch để kết quả tái lập được."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "m5cal") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def environment_info() -> dict:
    """Ghi lại phiên bản thư viện và phần cứng để báo cáo trong mục tái lập."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scipy", "lightgbm", "torch", "statsmodels"):
        try:
            module = __import__(pkg)
            info[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["cuda"] = False
    return info


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
