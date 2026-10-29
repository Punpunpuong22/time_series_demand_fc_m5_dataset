import numpy as np
import pytest

from m5cal.splits import Window, make_windows, training_cutoff, validate_windows


def test_window_layout_matches_report(windows):
    tune = [w for w in windows if w.role == "tune"]
    ev = [w for w in windows if w.role != "tune"]
    assert len(tune) == 2 and len(ev) == 13
    assert (tune[0].start, tune[-1].end) == (1522, 1577)
    assert (ev[0].start, ev[-1].end) == (1578, 1941)
    assert ev[-1].role == "test" and (ev[-1].start, ev[-1].end) == (1914, 1941)
    assert all(len(w.days) == 28 for w in windows)


def test_windows_contiguous_and_non_overlapping(windows):
    for a, b in zip(windows, windows[1:]):
        assert b.start == a.end + 1


def test_retrain_every_three(windows):
    ev = [w for w in windows if w.role != "tune"]
    assert [w.retrain for w in ev] == [k % 3 == 0 for k in range(13)]


def test_training_cutoff_never_after_origin(windows):
    for w in windows:
        assert training_cutoff(windows, w) <= w.origin


def test_eval_models_never_trained_on_eval_targets(windows):
    """Mốc huấn luyện của cửa sổ k luôn nhỏ hơn ngày bắt đầu của cửa sổ k."""
    for w in windows:
        assert training_cutoff(windows, w) < w.start


def test_overlap_rejected():
    ws = [Window("a", "eval", 10, 11, 20, True), Window("b", "eval", 15, 16, 30, False)]
    with pytest.raises(ValueError):
        validate_windows(ws, 100)


def test_beyond_data_rejected():
    with pytest.raises(ValueError):
        make_windows(n_eval=14)
