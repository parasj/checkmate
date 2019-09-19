import time

import numpy as np

from remat.core.utils.timer import Timer


def test_timer():
    for t in np.arange(0.05, 0.2, 0.02):
        with Timer("test") as timer:
            time.sleep(t)
        measured_time_err = abs(float(timer.elapsed) - t) / t
        assert measured_time_err <= 0.25  # 25% error is generous, but else this test is flaky
        assert float(timer.elapsed) >= t
