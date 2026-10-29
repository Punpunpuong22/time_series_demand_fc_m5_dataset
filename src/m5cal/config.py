"""Đọc các tệp cấu hình YAML trong thư mục configs/."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path = "configs") -> dict:
    """Trả về dict gồm base, experiments, lgbm, lstm."""
    config_dir = Path(config_dir)
    cfg = {
        "base": load_yaml(config_dir / "base.yaml"),
        "experiments": load_yaml(config_dir / "experiments.yaml"),
        "lgbm": load_yaml(config_dir / "lgbm.yaml"),
        "lstm": load_yaml(config_dir / "lstm.yaml"),
    }
    return cfg


def experiment_spec(cfg: dict, code: str) -> dict:
    """Lấy đặc tả của một thí nghiệm theo mã (ví dụ 'L-HCP')."""
    exps = cfg["experiments"]
    if code not in exps:
        raise KeyError(f"Không có thí nghiệm '{code}'. Các mã hợp lệ: {sorted(exps)}")
    spec = dict(exps[code])
    spec["code"] = code
    return spec
