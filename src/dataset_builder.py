"""Compatibility wrappers for the canonical dataset implementation.

New code should import ``build_dataset`` and ``split_dataset`` from
``dataset`` directly. This module remains for notebook-era callers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from config import CLASSES, NUM_CLASSES, SEED
from dataset import build_dataset as _build_dataset
from dataset import split_dataset


def build_dataset(data_dir: Path = Path("data/raw")) -> tuple[np.ndarray, np.ndarray]:
    """Build the balanced configured dataset through the canonical pipeline."""
    return _build_dataset(
        data_dir,
        classes=CLASSES,
        balance=True,
        seed=SEED,
    )


def get_train_val_test_splits(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    np.ndarray,
]:
    """Return the legacy tuple shape using canonical stratified splitting."""
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x,
        y,
        test_size=0.10,
        validation_size=0.10,
        seed=SEED,
    )
    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train_cat), (x_val, y_val_cat), (x_test, y_test_cat), y_test
