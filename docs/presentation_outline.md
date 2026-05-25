# Presentation Outline

## 1. Project Goal

- Classify IoT RF recordings into six classes using signal-derived images and
  a CNN.
- Classes: dooralarm, lora, microphone, mbus, sigfox, and miwi.

## 2. Dataset and Evaluation Units

- Raw and external test inputs are one-dimensional `.npy` RF recordings.
- Local reviewed files contain 10,000,000 samples each.
- Clarify the difference between window-level and file-level evaluation before
  showing any accuracy value.

## 3. End-to-End Pipeline

- Show `docs/flowchart.png`.
- Highlight the shared normalization/spectrogram path used for training and
  inference.
- Contrast non-overlapping training windows with overlapping inference windows.

## 4. Feature Representation

- Segment length: 4096 signal samples.
- Hann spectrogram configuration: `nperseg=256`, `noverlap=192`, `nfft=512`.
- Per-window normalization, log-compression, and normalized `257 x 61 x 1`
  model input.

## 5. CNN Design

- Show `docs/architecture_diagram.png`.
- Four convolutional stages scale channels from 32 to 256.
- Dense classification head outputs six softmax probabilities.

## 6. Training Protocol

- Balanced classes and stratified 80/10/10 window split.
- Adam optimizer, categorical cross-entropy, up to 15 epochs.
- Early stopping, learning-rate reduction, and best-validation checkpointing.

## 7. Results

| Measurement | Accuracy | Interpretation |
| --- | ---: | --- |
| Training held-out window split | 94.54% | Internal window-level test |
| Full raw-source window evaluation | 99.00% | Diagnostic; not independent |
| External whole-file ensemble test | 6/6 files correct; 95% CI `[0.610, 1.000]` | Small external smoke test |

## 8. Inference Demonstration

- Load one unseen-format-compatible `.npy` recording.
- Divide it into overlapping windows and average CNN probabilities.
- Display predicted class, confidence, and per-class probability distribution.

## 9. Review Findings and Limitations

- `evaluate.py` now requires an explicit diagnostic or external evaluation
  role in its metrics.
- Compatibility wrappers and unit tests now use the canonical transformation
  and model APIs.
- More independent recordings are required for defensible generalization
  claims.

## 10. Next Engineering Steps

- Add recording-level evaluation sets and report robust external metrics.
- Pin and automate the validated training/evaluation environment.
