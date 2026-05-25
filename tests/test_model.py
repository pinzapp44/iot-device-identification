import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import NUM_CLASSES
from model import build_cnn


def test_build_cnn():
    input_shape = (257, 61, 1)
    model = build_cnn(input_shape, NUM_CLASSES)

    assert model is not None
    assert model.output_shape == (None, NUM_CLASSES)
    assert model.loss == "categorical_crossentropy"
    assert model.layers[-1].activation.__name__ == "softmax"
