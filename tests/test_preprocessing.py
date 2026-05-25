import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import normalize_windows, segment_signal, windows_to_model_input


def test_segment_signal():
    data = np.arange(8192 + 10)
    windows = segment_signal(data, window_size=4096)

    assert windows.shape == (2, 4096)
    assert windows[0, 0] == 0
    assert windows[1, 0] == 4096


def test_normalize_windows():
    windows = np.random.randn(2, 4096) * 10 + 5
    normalized = normalize_windows(windows)

    assert np.allclose(np.mean(normalized, axis=1), 0, atol=1e-6)
    assert np.allclose(np.std(normalized, axis=1), 1, atol=1e-6)


def test_windows_to_model_input():
    windows = np.random.default_rng(42).standard_normal((3, 4096))
    specs = windows_to_model_input(windows)

    assert specs.shape == (3, 257, 61, 1)
    assert specs.dtype == np.float32
    assert np.allclose(np.mean(specs, axis=(1, 2, 3)), 0, atol=1e-6)
