# IoT-Based Embedded Platform for Bioacoustic Surveillance of Mosquito Species

**Thesis Project — Mindanao State University – Iligan Institute of Technology (MSU-IIT)**
Bachelor of Science in Computer Applications

**Authors:** Chine Hazel M. Hudaya, Danissa Mae M. Colanze, Neilroie T. Gapito
**Adviser:** Prof. Leah A. Alindayo

---

## Overview

This system performs real-time bioacoustic surveillance of mosquito species using an IoT-based two-node architecture. A field transmitter node captures wingbeat audio, extracts mel spectrograms on-device, and transmits classification results via LoRa to a base station running a Python receiver and web dashboard. The primary surveillance target is *Aedes* mosquitoes in support of dengue control programs in Iligan City, Lanao del Norte, Philippines.

---

## Repository Structure

```
Mosquito_detect_mac/
│   mosquito_receiver.py              # Python LoRa gateway receiver (v4.2)
│   mosquito_dashboard_simple.py      # Dash-based web dashboard
│
├── Models/                           # TFLite inference models
│       mosquito_vs_background.tflite     # Model 1 — binary mosquito detection (primary)
│       model1_v3.tflite                  # Model 1 v3
│       mosquito_vs_background_v1.tflite  # Model 1 v1
│       mosquito_species_classifier.tflite # Model 2 — 6-class species classification
│       model2_v3.tflite                  # Model 2 v3
│
├── logs/                             # Runtime logs (generated at runtime)
│       stats.json                        # Transmission statistics
│       live_log.txt                      # Live classification log
│
├── captured_data/                    # Captured mel spectrogram data (generated at runtime)
│   └── YYYYMMDD_HHMMSS/
│           capture_XXXX.npy              # Per-transmission mel spectrogram (NumPy)
│           captures_analysis.csv         # Analysis summary for the session
│
├── mosquito_receiver_arduino/        # Arduino gateway/receiver firmware
│   └── mosquito_receiver/
│           mosquito_receiver.ino
│
└── mosquito_transmitter_arduino/     # Arduino field node firmware (v3.4)
    └── mosquito_transmitter/
            mosquito_transmitter.ino      # Main transmitter sketch
            mel_spectrogram.h             # On-device mel spectrogram computation
```

---

## System Architecture

| Node | Hardware |
|------|----------|
| Field Transmitter | Arduino Nano 33 BLE Sense Rev2, ICS-43434 MEMS microphone, RFM95W LoRa (915 MHz), Li-ion battery, AMS1117 3.3V regulator |
| Gateway Receiver | Arduino (LoRa RX) → USB → Base Station PC |
| Base Station | Python receiver + Dash web dashboard |

### Audio Pipeline (Transmitter)
- Capture: 2-second clips, 16 kHz, 32,000 samples
- Pre-filter: Energy ratio E (0.3–20.0), spectral ratio R (0.039–0.10, 200–950 Hz), dB threshold −15.0, P/A ratio 2.0, spectral flatness 0.03–0.15
- Mel spectrogram: 128 mel bins, FFT 1024, hop length 256, Hamming window, normalized as (melDB + 80) / 80 → 128×122 feature map
- Inference: Model 1 (binary detection) → Model 2 (species classification) if mosquito detected

### Model Performance
| Model | Accuracy | Notes |
|-------|----------|-------|
| Model 1 (mosquito vs. background) | 91.83% | Precision 95.72%, Recall 88.00% |
| Model 2 (species classification) | 80.40% | 6 classes |

**Species covered by Model 2:** *Ae. aegypti*, *Ae. albopictus*, *An. arabiensis*, *An. gambiae*, *Cx. pipiens*, *Cx. quinquefasciatus*

### LoRa Communication
- Frequency: 915 MHz
- Packet reception rate: 97.0% across 31 test cycles

---

## Getting Started

### Requirements

- Python 3.8+
- Install dependencies:
  ```
  pip install numpy pandas dash plotly tflite-runtime pyserial
  ```

### Running the Receiver and Dashboard

1. Connect the Arduino receiver node via USB.
2. Start the Python receiver:
   ```
   python mosquito_receiver.py
   ```
3. In a separate terminal, start the dashboard:
   ```
   python mosquito_dashboard_simple.py
   ```
4. Open your browser at `http://127.0.0.1:8050`

### Arduino Firmware

- Open `mosquito_transmitter_arduino/mosquito_transmitter/mosquito_transmitter.ino` in Arduino IDE.
- Open `mosquito_receiver_arduino/mosquito_receiver/mosquito_receiver.ino` for the gateway node.
- Flash each to the respective Arduino board.

---

## Notes

- Captured `.npy` files and runtime logs are excluded from version control (see `.gitignore`). The `captured_data/` and `logs/` folders are generated automatically at runtime.
- Models were trained on the HumBugDB dataset (Kiskin et al., 2021) using MATLAB and converted to TFLite for deployment.
- Due to domain shift between the HumBugDB training data (condenser microphone) and the ICS-43434 MEMS deployment microphone, live detections are labeled *Unknown/Non-dengue Carrier Mosquito* pending retraining on MEMS-captured data.

---

## Citation

If referencing the training dataset:

> Kiskin, I., et al. (2021). *HumBugDB: A Large-scale Acoustic Mosquito Dataset*. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2110.07607.
