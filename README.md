# IoT-Based Embedded Platform for Bioacoustic Surveillance of Mosquito Species

An IoT-based embedded platform for real-time mosquito surveillance using bioacoustic sensing, embedded signal processing, LoRa communication, and TinyML.

**Bachelor of Science in Computer Applications Thesis**  
**Mindanao State University – Iligan Institute of Technology (MSU-IIT)**

**Authors**
- Chine Hazel M. Hudaya
- Danissa Mae M. Colanze
- Neilroie T. Gapito

**Adviser**
- Prof. Leah A. Alindayo

---

# Overview

This repository contains the complete implementation of an IoT-based embedded platform developed for bioacoustic surveillance of mosquito species in support of dengue control programs.

The system consists of a field transmitter node that captures mosquito wingbeat sounds using an ICS-43434 digital MEMS microphone. The transmitter performs embedded signal screening, generates mel spectrograms on-device, packetizes the extracted features, and transmits them through LoRa to a gateway node. The gateway forwards the received packets to a Base Station PC, where a Python receiver reconstructs the transmitted mel spectrogram, performs mosquito detection and species classification using two TensorFlow Lite models, records runtime statistics, and forwards the classification results to a Dash-based web dashboard for real-time visualization.

The project integrates embedded signal processing, TinyML, wireless communication, and web-based monitoring into a complete end-to-end mosquito surveillance platform intended to support dengue vector monitoring.

---

# System Architecture

The platform consists of three primary components.

## Field Transmitter Node

### Hardware

- Arduino Nano 33 BLE Sense Rev2
- ICS-43434 digital MEMS microphone
- RFM95W LoRa transceiver (915 MHz)
- Li-ion battery
- AMS1117 3.3 V voltage regulator

### Responsibilities

- Capture mosquito wingbeat audio
- Perform lightweight signal screening
- Generate mel spectrograms
- Packetize spectrogram data
- Transmit packets through LoRa

---

## Gateway Receiver Node

### Hardware

- Arduino with RFM95W LoRa receiver

### Responsibilities

- Receive LoRa packets
- Forward received packets to the Base Station PC through USB serial

---

## Base Station

### Software

- Python receiver
- Dash web dashboard

### Responsibilities

- Reconstruct transmitted mel spectrograms
- Execute TensorFlow Lite inference
- Record runtime logs and statistics
- Display live classification results

---

# Repository Structure

```text
mosquito-bioacoustic-surveillance/
│
├── Models/                         # TensorFlow Lite deployment models
├── training/                       # MATLAB training pipeline and model development
├── mosquito_transmitter_arduino/   # Field transmitter firmware
├── mosquito_receiver_arduino/      # Gateway receiver firmware
├── mosquito_receiver.py            # Python receiver and inference pipeline
├── mosquito_dashboard_simple.py    # Dash web dashboard
├── captured_data_copy/             # Example reconstructed mel spectrogram captures
├── logs/                           # Runtime log directory
├── README.md
└── LICENSE
```

---

# Features

- Real-time mosquito bioacoustic surveillance
- Embedded signal screening
- On-device mel spectrogram generation
- Two-stage TensorFlow Lite inference
- LoRa wireless communication
- Python-based gateway receiver
- Interactive Dash web dashboard
- Runtime logging and transmission statistics
- Example reconstructed mel spectrogram storage

---

# Signal Processing Pipeline

## Audio Acquisition

- Sampling rate: **16 kHz**
- Recording duration: **2 seconds**
- Samples per recording: **32,000**

---

## Signal Screening

Before feature extraction, the embedded transmitter evaluates each recording using lightweight signal processing to reduce unnecessary transmissions.

The screening stage evaluates:

- Signal energy ratio
- Spectral ratio
- Sound pressure level
- Peak-to-average ratio
- Spectral flatness

Only candidate mosquito recordings proceed to mel spectrogram generation.

---

## Feature Extraction

Mel spectrogram parameters:

- 128 mel frequency bins
- FFT size: 1024
- Hop length: 256
- Hamming window
- Log-mel normalization
- Output size: **128 × 122**

The generated mel spectrogram is packetized and transmitted to the Base Station through LoRa.

---

# Machine Learning Models

The system employs a cascaded TensorFlow Lite inference pipeline.

## Model 1

### Mosquito vs. Background Classification

Classes

- Mosquito
- Background

Performance

- Accuracy: **91.83%**
- Precision: **95.72%**
- Recall: **88.00%**

---

## Model 2

### Mosquito Species Classification

Supported species

- *Aedes aegypti*
- *Aedes albopictus*
- *Anopheles arabiensis*
- *Anopheles gambiae*
- *Culex pipiens*
- *Culex quinquefasciatus*

Performance

- Accuracy: **80.40%**

---

# LoRa Communication

Communication parameters

- Frequency: **915 MHz**
- Packetized mel spectrogram transmission
- Gateway architecture
- Packet reception rate: **97.0%** during experimental testing

---

# Software Components

## Field Node Firmware

Location

```text
mosquito_transmitter_arduino/
```

Responsibilities

- Audio acquisition
- Signal screening
- Mel spectrogram computation
- Packetization
- LoRa transmission

---

## Gateway Firmware

Location

```text
mosquito_receiver_arduino/
```

Responsibilities

- Receive LoRa packets
- Forward packets to the Base Station PC over USB serial

---

## Python Receiver

Location

```text
mosquito_receiver.py
```

Responsibilities

- Load the TensorFlow Lite interpreters for Model 1 (Mosquito vs. Background) and Model 2 (Mosquito Species Classification)
- Receive LoRa packets from the gateway node through USB serial
- Reconstruct the transmitted 128 × 122 mel spectrogram
- Execute the two-stage inference pipeline:
  - Model 1 determines whether mosquito activity is present.
  - If mosquito activity is detected, Model 2 classifies the mosquito species.
- Record runtime logs and transmission statistics
- Forward classification results to the web dashboard

---

## Web Dashboard

Location

```text
mosquito_dashboard_simple.py
```

Features

- Live mosquito detection results
- Species classification display
- Detection statistics
- Packet monitoring
- Runtime visualization

---

# Runtime Data

The repository includes example runtime outputs generated during system development and testing.

## captured_data_copy/

Contains reconstructed mel spectrogram captures collected during experimental evaluation.

## logs/

Contains the runtime log directory.

During normal system operation, the following files are generated automatically:

- `live_log.txt` – Live classification and transmission log
- `stats.json` – Runtime transmission and detection statistics

These runtime-generated files are excluded from version control through `.gitignore` to avoid committing frequently changing data while preserving the directory structure.

---

# Getting Started

## Requirements

### Hardware

- Arduino Nano 33 BLE Sense Rev2
- ICS-43434 MEMS microphone
- Two RFM95W LoRa modules (915 MHz)
- Base Station PC

### Software

- Python 3.8 or newer
- Arduino IDE
- MATLAB (for model development only)

Install the required Python packages:

```bash
pip install numpy pandas dash plotly pyserial tflite-runtime
```

---

# Running the System

## 1. Upload the Firmware

Flash

```text
mosquito_transmitter_arduino/mosquito_transmitter/
```

to the field transmitter.

Flash

```text
mosquito_receiver_arduino/mosquito_receiver/
```

to the gateway receiver.

---

## 2. Connect the Gateway

Connect the gateway receiver to the Base Station PC via USB.

---

## 3. Start the Python Receiver

```bash
python mosquito_receiver.py
```

---

## 4. Launch the Dashboard

Open another terminal and run

```bash
python mosquito_dashboard_simple.py
```

Open your browser at

```
http://127.0.0.1:8050
```

---

# Research Notes

- Mosquito recordings used for model development were derived from the **HumBugDB** dataset.
- Background audio used during Model 1 training was additionally sourced from the **ESC-50** environmental sound dataset.
- Model development, training, and evaluation were performed in MATLAB.
- The trained CNN models were converted to TensorFlow Lite for deployment.
- The deployed system uses an ICS-43434 digital MEMS microphone, whereas the HumBugDB recordings were collected using condenser microphones. This domain shift currently limits deployment performance until the models are retrained using MEMS-collected field recordings.

---

# Future Work

Potential improvements include:

- Retraining using MEMS microphone recordings
- Expansion to additional mosquito species
- Continuous model improvement
- Solar-powered autonomous deployment
- Cloud synchronization
- GPS-enabled surveillance
- Mobile application integration

---

# Acknowledgments

This work was conducted as part of the Bachelor of Science in Computer Applications program at **Mindanao State University – Iligan Institute of Technology (MSU-IIT)**.

The authors gratefully acknowledge the following publicly available datasets that made this research possible:

- **HumBugDB**, which served as the primary source of mosquito wingbeat recordings used for model development.
- **ESC-50**, which provided environmental sound recordings used as background audio during Model 1 training and evaluation.

---

# Citation

If you use this repository in academic work, please cite:

> Hudaya, C. H. M., Colanze, D. M. M., & Gapito, N. T. (2026). *IoT-Based Embedded Platform for Bioacoustic Surveillance of Mosquito Species in Support of Dengue Control Programs*. Bachelor of Science in Computer Applications Thesis. Mindanao State University – Iligan Institute of Technology.

### Datasets

> Kiskin, I., et al. (2021). *HumBugDB: A Large-scale Acoustic Mosquito Dataset*. NeurIPS 2021 Datasets and Benchmarks Track.

> Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. Proceedings of the 23rd ACM International Conference on Multimedia.

The ESC-50 dataset is also publicly available through the Harvard Dataverse:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT