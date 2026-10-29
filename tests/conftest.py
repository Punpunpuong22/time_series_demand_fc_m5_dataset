"""Fixture dùng chung: một bộ dữ liệu giả lập nhỏ đúng định dạng M5 (27 chuỗi × 1941 ngày)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5cal.data import build_panel, load_raw  # noqa: E402
from m5cal.daytypes import calendar_features  # noqa: E402
from m5cal.splits import make_windows  # noqa: E402
from m5cal.synthetic import make_synthetic_m5  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_m5")
    make_synthetic_m5(d, n_items_per_dept=3, seed=123)
    return d


@pytest.fixture(scope="session")
def panel(raw_dir):
    return build_panel(load_raw(raw_dir))


@pytest.fixture(scope="session")
def calf(panel):
    return calendar_features(panel.calendar, pre_event_days=3)


@pytest.fixture(scope="session")
def windows():
    return make_windows()
