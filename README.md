# IoT RF Device Classification

This project classifies recorded RF signals from six IoT device/protocol
classes with a spectrogram-based convolutional neural network (CNN):
`dooralarm`, `lora`, `microphone`, `mbus`, `sigfox`, and `miwi`.

The implemented pipeline converts one-dimensional `.npy` signals into
normalized `257 x 61 x 1` spectrogram inputs, trains a six-class CNN, and
supports both window-level evaluation and whole-file inference using averaged
sliding-window probabilities.

## Documentation

- [Documentation index](docs/index.md)
- [Architecture and diagrams](docs/architecture.md)
- [Methodology and recorded results](docs/methodology.md)
- [Code review findings](docs/code_review.md)
- [Presentation outline](docs/presentation_outline.md)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | Training, preprocessing, CNN definition, inference, and evaluation code |
| `data/raw/` | Training-source files named `<class>.npy` |
| `data/test/` | External test files named `<class>_test.npy` |
| `models/` | Saved model binaries when present, metadata, training plots, and held-out report |
| `results/evaluation/` | Window-level evaluation reports |
| `results/test/` | File-level external test reports and probabilities |
| `docs/` | Project documentation and generated architecture diagrams |
| `tests/` | Unit tests; see the documented review findings before relying on them |

Large `.npy` signal files and `.h5`/`.keras` model binaries are excluded from
Git by `.gitignore`.

## Setup

Create an environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected input files:

```text
data/raw/{dooralarm,lora,microphone,mbus,sigfox,miwi}.npy
data/test/{dooralarm,lora,microphone,mbus,sigfox,miwi}_test.npy
```

## Commands

Train a model from the raw labeled signals:

```bash
python src/train.py
```

This saves `models/best_iot_classifier.h5`,
`models/final_iot_classifier.h5`, `models/metadata.json`, training curves,
a confusion matrix, and a classification report.

Evaluate labeled windows from a directory:

```bash
python src/evaluate.py \
  --data-dir data/raw \
  --evaluation-role training-source-diagnostic
```

Run external file-level testing with sliding-window probability averaging:

```bash
python src/test.py --data-dir data/test
```

Classify one raw signal file:

```bash
python src/predict.py --input data/test/miwi_test.npy
```

Create sample spectrogram visualizations:

```bash
python src/visualize.py
```

Regenerate the documentation diagrams:

```bash
python docs/generate_diagrams.py
```

## Recorded Outputs

The checked-in metadata reports `0.9454` test accuracy for the random held-out
window split created during training. `results/evaluation/` reports `0.9900`
accuracy on `data/raw`, which is the same source directory used to build the
training set and should not be presented as independent generalization.
`results/test/` reports six correct external file predictions, one file per
class, using ensemble inference; its metrics include a file-level 95% Wilson
confidence interval to expose the small sample size.

Code review findings and their implemented remediations are tracked in
[docs/code_review.md](docs/code_review.md).
