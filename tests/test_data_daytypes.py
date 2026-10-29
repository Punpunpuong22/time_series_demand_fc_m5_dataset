import numpy as np
import pandas as pd

from m5cal.daytypes import daytype_flags, daytype_matrix, regularity
from m5cal.sampling import demand_classes, stratified_sample


def test_panel_shapes(panel):
    assert panel.Y.shape[1] == 1941
    assert panel.P.shape[1] == 1969 and panel.avail.shape == panel.P.shape
    assert panel.Y.min() >= 0


def test_no_price_before_launch_and_ffill_after(panel):
    launch = panel.launch_day()
    for i in range(panel.n):
        if launch[i] > 1:
            assert np.isnan(panel.P[i, : launch[i] - 1]).all()
        assert not np.isnan(panel.P[i, launch[i] - 1:]).any()


def test_snap_follows_store_state(panel):
    cal = panel.calendar
    for st in ("CA", "TX", "WI"):
        rows = np.flatnonzero(panel.meta["state_id"] == st)
        np.testing.assert_array_equal(panel.snap[rows[0]], cal[f"snap_{st}"].to_numpy())


def test_weekend_and_event_flags(calf, panel):
    cal = panel.calendar
    sat = cal.index[cal["weekday"] == "Saturday"][0]
    assert calf.loc[sat, "is_weekend"] == 1
    ev_day = cal.index[cal["event_type_1"].notna()][0]
    assert calf.loc[ev_day, "any_event"] == 1 and calf.loc[ev_day, "days_to_next_event"] == 0
    assert calf.loc[ev_day - 1, "pre_event"] == 1
    assert calf.loc[ev_day - 1, "days_to_next_event"] == 1


def test_primary_label_priority(panel, calf):
    cal = panel.calendar
    days = np.arange(1500, 1941)
    lab = daytype_matrix(panel, calf, days)
    ev_days = set(cal.index[cal["event_type_1"].notna()])
    for j, d in enumerate(days):
        if d in ev_days:
            assert lab[0, j].startswith("event_")
    # nhãn khác nhau giữa các bang chỉ do SNAP
    ca = np.flatnonzero(panel.meta["state_id"] == "CA")[0]
    tx = np.flatnonzero(panel.meta["state_id"] == "TX")[0]
    diff = lab[ca] != lab[tx]
    assert set(lab[ca][diff]) | set(lab[tx][diff]) <= {"snap", "weekday", "weekend"}


def test_flags_non_exclusive(panel, calf):
    days = np.arange(1578, 1606)
    f = daytype_flags(panel, calf, days)
    assert f["snap"].shape == (panel.n, len(days))
    assert regularity("weekend") == "regular" and regularity("snap") == "irregular"


def test_demand_classes_and_sampling(panel):
    cls = demand_classes(panel, 1521)
    assert set(cls["demand_class"]) <= {"smooth", "erratic", "lumpy", "intermittent", "no_sales"}
    meta = pd.concat([panel.meta, cls], axis=1)
    idx = stratified_sample(meta, 10, ["store_id", "cat_id"], seed=1)
    assert len(idx) <= 10 and len(set(idx)) == len(idx)
    idx2 = stratified_sample(meta, 10, ["store_id", "cat_id"], seed=1)
    np.testing.assert_array_equal(idx, idx2)   # tái lập được
    assert len(stratified_sample(meta, None, ["store_id"])) == panel.n
