# RF Signal Dataset Generator

## Overview

This script generates a synthetic RF signal dataset consisting of six wireless protocol classes under three different noise conditions. The generated signals are stored as NumPy (`.npy`) files and can be used for:

* RF signal classification
* Spectrum sensing research
* Signal processing experiments
* Spectrogram-based machine learning
* Educational demonstrations of wireless protocols

The dataset contains **18 recordings** in total.

---

# Dataset Structure

```text
data/
└── recordings/
    ├── dooralarm/
    │   ├── baseline_capture01.npy
    │   ├── background_capture01.npy
    │   └── upstairs_capture01.npy
    │
    ├── lora/
    │   ├── baseline_capture01.npy
    │   ├── background_capture01.npy
    │   └── upstairs_capture01.npy
    │
    ├── microphone/
    ├── mbus/
    ├── miwi/
    └── sigfox/
```

Total generated files:

```text
6 protocol classes × 3 scenarios = 18 recordings
```

---

# Signal Configuration

## Sampling Rate

```python
fs = 10e6
```

* Sampling frequency: **10 MHz**

---

## Recording Length

```python
N = 500_000
```

* Samples per recording: **500,000**

---

## Recording Duration

```text
500,000 / 10,000,000 = 0.05 seconds
```

Each recording contains **50 ms** of RF data.

---

# Noise Scenarios

The script generates recordings under three signal-to-noise ratio (SNR) conditions.

| Scenario   | SNR   |
| ---------- | ----- |
| baseline   | 25 dB |
| background | 10 dB |
| upstairs   | 5 dB  |

Noise is generated using additive white Gaussian noise (AWGN).

---

# Output Format

Each file is saved as:

```python
float32
shape = (500000,)
```

Example:

```python
array([
    0.000184,
   -0.000097,
    0.000213,
    ...
], dtype=float32)
```

Each `.npy` file contains only raw waveform samples.

The following information is **not stored**:

* Spectrograms
* Peak frequencies
* Bandwidth estimates
* Signal power values
* Metadata
* Bit streams

---

# Protocol Classes

## DoorAlarm

### Description

Simple On-Off Keying (OOK) transmission.

### Frequency Region

```text
-100 kHz
```

### Characteristics

* Narrowband
* Long symbols
* Carrier appears and disappears

### Signal Model

```text
s(t) = A × bit(t) × cos(2πf₀t)
```

Parameters:

```text
Amplitude = 0.0002
Carrier = -100 kHz
```

---

## LoRa

### Description

Simplified chirp spread spectrum signal.

### Frequency Region

```text
+50 kHz
```

### Bandwidth

```text
125 kHz
```

### Characteristics

* Continuous chirps
* Frequency sweeps upward
* Wide spectral signature

### Spectrogram Appearance

```text
Frequency

125k |         /
     |       /
     |     /
     |   /
 50k |__/
     +-------------> Time
```

---

## Microphone

### Description

Audio-like signal generated from filtered white noise.

### Generation Process

1. Generate white Gaussian noise.
2. Apply Butterworth bandpass filter.
3. Apply amplitude envelope modulation.

### Characteristics

* Wideband signal
* Speech-like variations
* No RF carrier tone

### Spectrogram Appearance

```text
██████
██████
██████
```

Diffuse energy across a broad frequency range.

---

## MBus

### Description

Binary Frequency Shift Keying (FSK).

### Frequencies

| Bit | Frequency |
| --- | --------- |
| 0   | 150 kHz   |
| 1   | 200 kHz   |

### Characteristics

* Constant envelope
* Frequency hopping between two tones
* Narrowband

### Spectrogram Appearance

```text
200k ──────┐ ┌─────
           │ │
150k ──────┘ └─────
```

---

## Sigfox

### Description

Simplified BPSK-like transmission.

### Frequency Region

```text
+250 kHz
```

### Characteristics

* Constant amplitude
* Phase reversals
* Very narrowband

### Symbol Mapping

| Symbol | Phase |
| ------ | ----- |
| +1     | 0°    |
| -1     | 180°  |

---

## MiWi

### Description

Simplified Direct Sequence Spread Spectrum (DSSS) signal.

### Frequency Region

```text
-200 kHz
```

### Spreading Code

```text
[1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]
```

### Characteristics

* Spread spectrum
* Rapid sign transitions
* Wider occupied bandwidth

---

# Signal Amplitudes

Approximate signal amplitudes used during generation:

| Protocol   | Amplitude |
| ---------- | --------- |
| DoorAlarm  | 0.0002    |
| LoRa       | 0.0015    |
| Microphone | 0.0008    |
| MBus       | 0.0003    |
| Sigfox     | 0.0020    |
| MiWi       | 0.0005    |

Noise is added afterward according to the selected SNR scenario.

---

# Frequency Allocation

Each protocol occupies a unique frequency region to improve class separability.

| Protocol   | Frequency Region     |
| ---------- | -------------------- |
| DoorAlarm  | -100 kHz             |
| LoRa       | +50 kHz              |
| Microphone | Baseband / Broadband |
| MBus       | 150–200 kHz          |
| MiWi       | -200 kHz             |
| Sigfox     | +250 kHz             |

---

# Diagnostic Metrics

For each generated recording, the script computes diagnostic information before saving.

## Peak Frequency

Estimated strongest frequency component:

```python
f_peak
```

---

## Occupied Bandwidth

Estimated using a 10% power threshold:

```python
bw
```

---

## Signal Power

Computed as:

```python
10 * log10(mean(signal²))
```

Reported in dBW.

---

# Example Console Output

```text
dooralarm    baseline   peak=-101.6kHz bw=39.1kHz  pwr=-77.2dBW
lora         baseline   peak=+50.8kHz  bw=156.2kHz pwr=-59.5dBW
microphone   baseline   peak=+0.0kHz   bw=820.3kHz pwr=-66.3dBW
mbus         baseline   peak=+156.2kHz bw=78.1kHz  pwr=-73.4dBW
sigfox       baseline   peak=+250.0kHz bw=39.1kHz  pwr=-57.0dBW
miwi         baseline   peak=-195.3kHz bw=117.2kHz pwr=-69.1dBW
```

Actual values vary because the generated signals and noise are random.

---

# Loading a Recording

```python
import numpy as np

signal = np.load(
    "data/recordings/lora/baseline_capture01.npy"
)

print(signal.shape)
print(signal.dtype)
print(signal[:10])
```

Example output:

```text
(500000,)
float32

[0.00142 0.00138 0.00131 0.00125 ...]
```

---

# Machine Learning Workflow

Typical usage:

```text
Raw .npy waveform
        ↓
Windowing
        ↓
STFT / Spectrogram
        ↓
Feature Extraction
        ↓
Classifier (CNN, Transformer, etc.)
        ↓
Protocol Prediction
```

---

# Limitations

This dataset is synthetic and intended for experimentation.

The generated signals are inspired by real protocols but are **not standards-compliant implementations** of:

* LoRa
* Sigfox
* Wireless M-Bus
* MiWi

The goal is to create visually and spectrally distinct signal classes for machine learning and signal-processing experiments rather than exact protocol emulation.
