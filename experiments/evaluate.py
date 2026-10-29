"""Tính độ đo, kiểm định và mô phỏng tồn kho cho mọi dự báo đã lưu trong results/preds/.

Ví dụ:
    python experiments/evaluate.py
    python experiments/evaluate.py --out results_smoke --raw-dir data/synthetic --pairs L-HCP:S-HCP L-HC:S-HC
Kết quả ghi vào <out>/tables/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5cal.config import load_config  # noqa: E402
from m5cal.daytypes import daytype_matrix  # noqa: E402
from m5cal.inventory import simulate_order_up_to, summarize_inventory  # noqa: E402
from m5cal.pipeline import evaluate, load_predictions, prepare  # noqa: E402
from m5cal.stats_tests import bootstrap_wrmsse_diff, dm_by_series, friedman_nemenyi  # noqa: E402


def pivot_series(series: pd.DataFrame, exps) -> pd.DataFrame:
    s = series[series["exp"].isin(exps)]
    return s.groupby(["sid", "exp"])["rmsse"].mean().unstack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(ROOT / "configs"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs", nargs="*", default=["L-HCP:S-HCP", "L-HC:S-HC", "L-H:S-H", "L-HCP:B0", "S-HCP:B0"])
    ap.add_argument("--inventory", nargs="*", default=["B0", "L-HCP", "S-HCP"])
    args = ap.parse_args()

    cfg = load_config(args.config_dir)
    out = Path(args.out or cfg["base"]["paths"]["out_dir"])
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    panel, calf, windows = prepare(cfg, args.raw_dir, -1, out)
    preds = load_predictions(out)
    res = evaluate(panel, calf, windows, preds)
    for name in ("overall", "by_window", "by_daytype", "by_demand_class"):
        res[name].to_csv(tdir / f"{name}.csv", index=False)
    print("\n=== Tổng thể (trung bình qua cửa sổ và seed) ===")
    print(res["overall"].to_string(index=False))

    # ---- Diebold–Mariano theo chuỗi và bootstrap WRMSSE (seed 1) ----
    ev = preds[preds["role"].isin(["eval", "test"]) & (preds["seed"] == 1)]
    dm_rows, boot_rows = [], []
    for pair in args.pairs:
        a, b = pair.split(":")
        if a not in set(ev["exp"]) or b not in set(ev["exp"]):
            continue
        dm = dm_by_series(ev[ev["exp"] == a], ev[ev["exp"] == b], h=28)
        dm_rows.append({"A": a, "B": b, "share_A_better": (dm["better"] == "A").mean(),
                        "share_B_better": (dm["better"] == "B").mean(), "share_tie": (dm["better"] == "tie").mean(),
                        "n_series": len(dm)})
        s = res["series"][res["series"]["seed"] == 1]
        ra = s[s["exp"] == a].groupby("sid")["rmsse"].mean()
        rb = s[s["exp"] == b].groupby("sid")["rmsse"].mean()
        wt = s[s["exp"] == a].groupby("sid")["weight"].mean()
        common = ra.index.intersection(rb.index)
        boot = bootstrap_wrmsse_diff(ra[common], rb[common], wt[common])
        boot_rows.append({"A": a, "B": b, **boot})
    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(tdir / "dm_tests.csv", index=False)
        pd.DataFrame(boot_rows).to_csv(tdir / "bootstrap_wrmsse_diff.csv", index=False)
        print("\n=== Diebold–Mariano (tỷ lệ chuỗi, sau hiệu chỉnh BH) ===")
        print(pd.DataFrame(dm_rows).to_string(index=False))

    # ---- Friedman + Nemenyi trên RMSSE trung bình theo chuỗi ----
    exps = sorted(res["series"]["exp"].unique())
    if len(exps) >= 3:
        fr = friedman_nemenyi(pivot_series(res["series"][res["series"]["seed"] == 1], exps))
        with open(tdir / "friedman_nemenyi.json", "w", encoding="utf-8") as f:
            json.dump(fr, f, indent=2)
        print("\n=== Friedman–Nemenyi ===")
        print(json.dumps(fr, indent=2))

    # ---- Mô phỏng tồn kho (RQ4) ----
    inv_cfg = cfg["base"]["inventory"]
    inv_rows = []
    tune = preds[(preds["role"] == "tune") & (preds["seed"] == 1)]
    for exp in args.inventory:
        if exp not in set(ev["exp"]):
            continue
        e_t = tune[tune["exp"] == exp]
        sigma = (e_t.assign(err=e_t["y_pred"] - e_t["y_true"]).groupby("sid")["err"].std()
                 .reindex(range(panel.n)).fillna(0).to_numpy())
        for wname, g in ev[ev["exp"] == exp].groupby("window"):
            g = g.sort_values(["sid", "d"])
            days = np.sort(g["d"].unique())
            D = g["y_true"].to_numpy().reshape(panel.n, len(days))
            F = g["y_pred"].to_numpy().reshape(panel.n, len(days))
            sim = simulate_order_up_to(D, F, sigma, int(inv_cfg["lead_time"]), float(inv_cfg["service_z"]))
            labels = daytype_matrix(panel, calf, days)
            summ = summarize_inventory(sim, D, labels)
            summ["exp"], summ["window"] = exp, wname
            inv_rows.append(summ)
    if inv_rows:
        inv = pd.concat(inv_rows)
        inv_tab = inv.groupby(["exp", "daytype"])[["fill_rate", "avg_inventory", "stockout_day_share"]].mean().reset_index()
        inv_tab.to_csv(tdir / "inventory.csv", index=False)
        print("\n=== Tồn kho (mô phỏng, trung bình qua cửa sổ) ===")
        print(inv_tab[inv_tab["daytype"] == "all"].to_string(index=False))
    print(f"\nĐã ghi các bảng vào {tdir}")


if __name__ == "__main__":
    main()
