import time

import numpy as np

from remat.core.utils.timer import Timer


def test_timer():
    for t in np.arange(0.05, 0.2, 0.02):
        with Timer("test") as timer:
            time.sleep(t)
        assert float(timer.elapsed) >= t
