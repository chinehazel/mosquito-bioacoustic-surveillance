# IoT-Based Embedded Platform for Bioacoustic Surveillance of Mosquito Species

An IoT-based embedded system for real-time mosquito detection and species classification using bioacoustic sensing, TinyML, and LoRa communication.

**Bachelor of Science in Computer Applications **  
**Mindanao State University – Iligan Institute of Technology (MSU-IIT)**

**Authors**
- Chine Hazel M. Hudaya
- Danissa Mae M. Colanze
- Neilroie T. Gapito

**Adviser**
- Prof. Leah A. Alindayo

---

## Overview

This repository contains the complete implementation of an IoT-based embedded platform designed for bioacoustic surveillance of mosquito species in support of dengue control programs. The system captures mosquito wingbeat sounds using an embedded MEMS microphone, extracts mel spectrograms directly on the field device, performs on-device inference using TensorFlow Lite models, and transmits detection results through LoRa to a gateway node connected to a Python-based monitoring application and web dashboard.

The system is intended for deployment in outdoor environments to provide automated mosquito monitoring while minimizing power consumption and communication bandwidth through edge processing.

---

## System Architecture

The platform consists of three primary components:

### Field Transmitter Node

- Arduino Nano 33 BLE Sense Rev2
- ICS-43434 digital MEMS microphone
- RFM95W LoRa transceiver (915 MHz)
- On-device mel spectrogram computation
- TensorFlow Lite inference
- Battery-powered operation

### Gateway Receiver Node

- Arduino with RFM95W LoRa receiver
- USB serial communication to the base station

### Base Station

- Python receiver application
- Dash-based web dashboard
- Runtime logging
- Spectrogram reconstruction
- Detection visualization

---

## Repository Structure

```text
mosquito-bioacoustic-surveillance/
│
├── Models/                         # TensorFlow Lite deployment models
├── training/                       # MATLAB training pipeline and model development
├── mosquito_transmitter_arduino/   # Field node firmware
├── mosquito_receiver_arduino/      # Gateway node firmware
├── mosquito_receiver.py            # Python gateway receiver
├── mosquito_dashboard_simple.py    # Dash web dashboard
├── captured_data_copy/             # Example captured spectrogram data
├── logs/                           # Example runtime logs
├── README.md
└── LICENSE
```

---

## Features

- Real-time mosquito wingbeat detection
- Embedded mel spectrogram generation
- TensorFlow Lite inference on Arduino Nano 33 BLE Sense Rev2
- Two-stage deep learning classification pipeline
- LoRa-based wireless communication
- Python gateway receiver
- Interactive Dash web dashboard
- Runtime logging and transmission statistics
- Storage of reconstructed mel spectrogram captures
- Low-power edge computing architecture suitable for field deployment

---

## Signal Processing Pipeline

### Audio Acquisition

- Sampling rate: **16 kHz**
- Recording duration: **2 seconds**
- Total samples: **32,000**

### Signal Screening

Audio segments are screened using lightweight signal processing before neural network inference to reduce unnecessary computation.

The screening stage evaluates:

- Signal energy ratio
- Spectral ratio
- Sound pressure level
- Peak-to-average ratio
- Spectral flatness

Only candidate mosquito sounds proceed to feature extraction.

### Feature Extraction

Mel spectrogram parameters:

- 128 mel frequency bins
- FFT size: 1024
- Hop length: 256
- Hamming window
- Normalized log-mel representation
- Output size: **128 × 122**

---

## Machine Learning Models

The system employs a cascaded deep learning architecture.

### Model 1

**Binary Classification**

Classes:

- Mosquito
- Background

Performance:

- Accuracy: **91.83%**
- Precision: **95.72%**
- Recall: **88.00%**

---

### Model 2

**Species Classification**

Supported species:

- *Aedes aegypti*
- *Aedes albopictus*
- *Anopheles arabiensis*
- *Anopheles gambiae*
- *Culex pipiens*
- *Culex quinquefasciatus*

Performance:

- Accuracy: **80.40%**

---

## Communication

### LoRa Parameters

- Frequency: **915 MHz**
- Gateway architecture
- Packetized mel spectrogram transmission
- Packet reception rate: **97.0%** during experimental testing

---

## Software Components

### Embedded Firmware

Located in:

```
mosquito_transmitter_arduino/
```

Responsibilities:

- Audio acquisition
- Signal preprocessing
- Mel spectrogram computation
- TensorFlow Lite inference
- LoRa transmission

---

### Gateway Firmware

Located in:

```
mosquito_receiver_arduino/
```

Responsibilities:

- Receive LoRa packets
- Forward packets over USB serial

---

### Python Receiver

```
mosquito_receiver.py
```

Responsibilities:

- Receive incoming packets
- Reconstruct mel spectrograms
- Execute additional classification
- Save runtime logs
- Forward results to the dashboard

---

### Dashboard

```
mosquito_dashboard_simple.py
```

Features:

- Live classification updates
- Detection statistics
- Packet monitoring
- Runtime visualization

---

## Runtime Data

The repository includes example runtime outputs for demonstration purposes.

### logs/

Contains example:

- transmission statistics
- live detection logs

### captured_data_copy/

Contains example reconstructed mel spectrogram captures generated during system testing.

Additional runtime data are automatically generated while the system is operating.

---

## Getting Started

### Requirements

- Python 3.8 or newer
- Arduino IDE
- Arduino Nano 33 BLE Sense Rev2
- RFM95W LoRa modules

Install Python dependencies:

```bash
pip install numpy pandas dash plotly pyserial tflite-runtime
```

---

## Running the System

### 1. Upload the Firmware

Flash:

```
mosquito_transmitter_arduino/mosquito_transmitter/
```

to the field transmitter.

Flash:

```
mosquito_receiver_arduino/mosquito_receiver/
```

to the gateway receiver.

---

### 2. Connect the Gateway

Connect the gateway Arduino to the base station through USB.

---

### 3. Start the Receiver

```bash
python mosquito_receiver.py
```

---

### 4. Launch the Dashboard

Open another terminal and run:

```bash
python mosquito_dashboard_simple.py
```

Open your browser at:

```
http://127.0.0.1:8050
```

---

## Research Notes

The machine learning models were trained using the **HumBugDB** mosquito acoustic dataset and developed in MATLAB before being converted to TensorFlow Lite for deployment on embedded hardware.

Because the training dataset was collected using condenser microphones while the deployed system uses an ICS-43434 digital MEMS microphone, a domain shift exists between training and deployment data. As a result, live detections are currently displayed as **Unknown/Non-dengue Carrier Mosquito** until the models are retrained using MEMS-collected field recordings.

---

## Future Work

Planned improvements include:

- Retraining using MEMS microphone recordings
- Additional mosquito species support
- Continuous learning pipeline
- Solar-powered autonomous deployment
- Cloud synchronization
- GPS-enabled surveillance
- Mobile application integration

---

## Acknowledgments

This research was conducted as part of the Bachelor of Science in Computer Applications program at **Mindanao State University – Iligan Institute of Technology (MSU-IIT)**.

The authors acknowledge the developers of the **HumBugDB** dataset, which served as the primary training dataset for the machine learning models.

---

## Citation

If you use this repository in academic work, please cite:

> Hudaya, C. H. M., Colanze, D. M. M., & Gapito, N. T. (2026). *IoT-Based Embedded Platform for Bioacoustic Surveillance of Mosquito Species in Support of Dengue Control Programs*. Bachelor of Science in Computer Applications Thesis, Mindanao State University – Iligan Institute of Technology.

Training dataset:

> Kiskin, I., et al. (2021). *HumBugDB: A Large-scale Acoustic Mosquito Dataset*. NeurIPS 2021 Datasets and Benchmarks Track.