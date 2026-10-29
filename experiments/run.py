"""Chạy một hoặc nhiều thí nghiệm.

Ví dụ:
    python experiments/run.py --exp B0 B1                       # baseline
    python experiments/run.py --exp L-HCP --seeds 1 2 3 4 5     # LightGBM đầy đủ, 5 seed
    python experiments/run.py --exp S-HC --seeds 1 --max-eval 3 # chạy thử 3 cửa sổ đầu
    python experiments/run.py --exp all --seeds 1               # toàn bộ ma trận, 1 seed
    python experiments/run.py --exp B0 L-H --raw-dir data/synthetic --n-series 30 --out results_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5cal.config import load_config  # noqa: E402
from m5cal.pipeline import prepare, run_experiment  # noqa: E402

BASELINES = {"B0", "B1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", required=True, help="Mã thí nghiệm trong configs/experiments.yaml hoặc 'all'")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="Mặc định: seeds trong base.yaml")
    ap.add_argument("--config-dir", default=str(ROOT / "configs"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-series", type=int, default=-1, help="-1 = theo base.yaml; 0 = toàn bộ chuỗi")
    ap.add_argument("--out", default=None, help="Thư mục kết quả (mặc định results/)")
    ap.add_argument("--max-eval", type=int, default=None, help="Chỉ chạy N cửa sổ đánh giá đầu (chạy thử)")
    args = ap.parse_args()

    cfg = load_config(args.config_dir)
    codes = list(cfg["experiments"]) if args.exp == ["all"] else args.exp
    seeds = args.seeds or cfg["base"]["seeds"]
    out = args.out or cfg["base"]["paths"]["out_dir"]
    n_series = None if args.n_series == 0 else args.n_series

    panel, calf, windows = prepare(cfg, args.raw_dir, n_series, out)
    for code in codes:
        for seed in ([1] if code in BASELINES else seeds):   # baseline không có tính ngẫu nhiên
            run_experiment(cfg, code, seed, panel, calf, windows, out, max_eval=args.max_eval)


if __name__ == "__main__":
    main()
