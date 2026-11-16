"""Added 7 Oct 2026 (lead): daily-log transforms ('log_daily', 'log1p_daily') average the daily logs."""
import numpy as np
from model.har import _tf, rolling_nanmean


def test_tf_nonpositive_is_nan():
    x = np.array([[2.0, 0.0, -1.0, 1.0]])
    out = _tf(x, "log")
    assert np.isclose(out[0, 0], np.log(2.0)) and np.isnan(out[0, 1]) and np.isnan(out[0, 2]) and out[0, 3] == 0.0
    out1 = _tf(np.array([[0.0, -1.0, -2.0]]), "log1p")
    assert out1[0, 0] == 0.0 and np.isnan(out1[0, 1]) and np.isnan(out1[0, 2])


def test_log_then_average_differs_from_average_then_log():
    x = np.array([[1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0]])
    daily = rolling_nanmean(_tf(x, "log1p"), 7)[0, -1]
    after = _tf(rolling_nanmean(x, 7), "log1p")[0, -1]
    assert np.isclose(daily, np.mean(np.log1p(x[0])))
    assert np.isclose(after, np.log1p(np.mean(x[0])))
    assert daily < after
