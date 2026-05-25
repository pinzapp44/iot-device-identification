"""Create spectrogram preview plots from raw class files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES, DATA_FILES, WINDOW_SIZE
from preprocessing import segment_signal, windows_to_model_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/spectrograms"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    return parser.parse_args()


def save_spectrogram(window: np.ndarray, title: str, output_path: Path) -> None:
    """Plot the normalized spectrogram representation supplied to the CNN."""
    model_input = windows_to_model_input(window[np.newaxis, :])[0, ..., 0]

    plt.figure(figsize=(5, 4))
    plt.imshow(model_input, aspect="auto", origin="lower", cmap="viridis")
    plt.title(title)
    plt.xlabel("Time frame")
    plt.ylabel("Frequency bin")
    plt.colorbar(label="Normalized log spectrum")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        data_path = args.data_dir / DATA_FILES[class_name]
        raw_signal = np.load(data_path)
        windows = segment_signal(raw_signal, window_size=args.window_size)
        if len(windows) == 0:
            raise ValueError(f"{data_path} has fewer than {args.window_size} samples")

        output_path = args.output_dir / f"{class_name}.png"
        save_spectrogram(windows[0], class_name.title(), output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
