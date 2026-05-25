import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import create_spectrograms


def test_create_spectrograms():
    windows = np.random.default_rng(42).standard_normal((3, 4096))
    specs = create_spectrograms(windows)

    assert specs.shape == (3, 257, 61, 1)
    assert specs.dtype == np.float32
