# Architecture

## System Overview

The current executable pipeline is organized around these modules:

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Class order, default hyperparameters, and spectrogram configuration |
| `src/dataset.py` | Locate labeled signal files, segment, preprocess, balance, shuffle, and split data |
| `src/preprocessing.py` | Windowing, per-window normalization, and spectrogram conversion |
| `src/model.py` | Define and compile the Keras CNN |
| `src/train.py` | Train, checkpoint, evaluate the held-out split, and write model artifacts |
| `src/infer.py` | Shared sliding-window whole-signal prediction logic |
| `src/predict.py` | Single-file prediction command |
| `src/test.py` | File-level external test reporting |
| `src/evaluate.py` | Window-level evaluation for a labeled directory |
| `src/evaluation.py` | Classification reports, confusion matrices, and training-curve plots |

`src/dataset_builder.py` and `src/spectrogram.py` are compatibility exports for
notebook-era callers and now delegate to the canonical `dataset.py` and
`preprocessing.py` logic. `src/data_loader.py` and `src/utils.py` are retained
legacy utilities and are not used by the executable pipeline.

## Training Flow

![RF classification data and execution flow](flowchart.png)

1. `dataset.build_dataset()` loads each class's one-dimensional `float32`
   signal from `data/raw/<class>.npy`.
2. The signal is divided into non-overlapping `4096`-sample windows.
3. Each window is standardized independently and converted to a log-compressed
   SciPy spectrogram using the configuration in `config.py`.
4. When balancing is enabled, each class is truncated to the lowest available
   number of windows; examples are then shuffled.
5. `split_dataset()` creates stratified 80% train, 10% validation, and 10%
   held-out test window splits.
6. `train.py` trains the CNN and writes model/report artifacts to `models/`.

## Inference Flow

Single-file and external testing use `infer.predict_signal()`:

1. Load a raw `.npy` signal.
2. Form overlapping `4096`-sample windows using a default step of `1024`.
3. Apply the same normalization and spectrogram transformation used during
   training.
4. Predict six probabilities for every window.
5. Average probabilities across all windows and select the highest averaged
   probability as the file prediction.

## CNN Architecture

![Implemented CNN architecture](architecture_diagram.png)

Given the recorded input shape in `models/metadata.json`, the layer path is:

| Stage | Layer/operation | Output shape |
| --- | --- | --- |
| Input | Normalized spectrogram | `257 x 61 x 1` |
| Augmentation | `GaussianNoise(0.003)` during training | `257 x 61 x 1` |
| Block 1 | `Conv2D(32, 3x3, same)` + BatchNorm + ReLU + MaxPool | `128 x 30 x 32` |
| Block 2 | `Conv2D(64, 3x3, same)` + BatchNorm + ReLU + MaxPool | `64 x 15 x 64` |
| Block 3 | `Conv2D(128, 3x3, same)` + BatchNorm + ReLU + MaxPool | `32 x 7 x 128` |
| Block 4 | `Conv2D(256, 3x3, same)` + BatchNorm + ReLU | `32 x 7 x 256` |
| Head | Flatten + Dense(256, ReLU) + Dropout(0.25) | `256` |
| Output | Dense(6, softmax) | `6` probabilities |

The model uses Adam with learning rate `3e-4`, categorical cross-entropy, and
accuracy as its configured training metric.

## Artifact Paths

| Producer | Artifacts |
| --- | --- |
| `src/train.py` | `models/best_iot_classifier.h5`, `final_iot_classifier.h5`, `metadata.json`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt` |
| `src/evaluate.py` | `results/evaluation/metrics.json` with declared evaluation role, `predictions.csv`, `confusion_matrix.png`, `classification_report.txt` |
| `src/test.py` | `results/test/metrics.json` with file-level confidence interval, `predictions.csv`, `probabilities.json`, `confusion_matrix.png`, `classification_report.txt` |
| `src/visualize.py` | Per-class spectrogram images under `results/spectrograms/` |
