"""Sinh dữ liệu GIẢ LẬP định dạng M5 để chạy thử pipeline (không phải dữ liệu thật).

    python scripts/make_synthetic.py --out data/synthetic --items 6
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5cal.synthetic import make_synthetic_m5  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--items", type=int, default=6, help="Số sản phẩm mỗi bộ phận")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = make_synthetic_m5(a.out, n_items_per_dept=a.items, seed=a.seed)
    print(f"Đã sinh dữ liệu giả lập tại {p}")
