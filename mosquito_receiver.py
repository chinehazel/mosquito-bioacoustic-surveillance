#!/usr/bin/env python3
"""
========================================
MOSQUITO RECEIVER PYTHON -- v4.1
MSU-IIT Thesis Project - Chene

CHANGES FROM v4:
  [1] TIME_FRAMES updated from 59 to 122 (2-second window rollback)
  [2] MEL comment updated to reflect 2-second window
  [3] Matches mosquito_vs_background.tflite training pipeline:
      128 x 122 mel spectrogram
      Hamming window
      Fixed normalization (melDB + 80) / 80
      No HPF

CHANGES FROM v3:
  [1] MODEL2_THRESHOLD lowered from 1.1 to 0.80 (now reachable)
  [2] System now focuses on Aedes detection as primary output
  [3] Final result explicitly states DENGUE VECTOR DETECTED
      if Ae. aegypti or Ae. albopictus is classified
  [4] Non-Aedes mosquitoes logged but no dengue alert raised
  [5] Em dashes removed from live_log.txt writes
  [6] Simplified final result output to reflect study purpose
========================================
"""

import serial
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
import os
import time
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================
SERIAL_PORT = '/dev/cu.usbmodem1201'
BAUD_RATE   = 115200

_GDRIVE = Path.home() / 'Library/CloudStorage/GoogleDrive-mosquitodetect@gmail.com/My Drive/Mosquito_detect_mac'

MODEL1_PATH = str(_GDRIVE / 'Models/mosquito_vs_background.tflite')
MODEL2_PATH = str(_GDRIVE / 'Models/mosquito_species_classifier.tflite')

# 2-second window mel dimensions (matches mosquito_vs_background.tflite training)
MEL_BINS    = 128
TIME_FRAMES = 122

# Thresholds
MODEL1_THRESHOLD = 0.50   # mosquito detection confidence
MODEL2_THRESHOLD = 0.80   # species classification confidence

# Dengue vector species -- primary focus of this study
DENGUE_VECTORS = ['Aedes aegypti', 'Aedes albopictus']

# Species labels in MATLAB alphabetical order
SPECIES_LABELS = [
    'Aedes aegypti',
    'Aedes albopictus',
    'Anopheles arabiensis',
    'Anopheles gambiae',
    'Culex pipiens',
    'Culex quinquefasciatus'
]

DETECTOR_LABELS = ['Background', 'Mosquito']

FRAME_START = 0xAA
FRAME_END   = 0x55


# ========================================
# LORA RECEIVER
# ========================================
class LoRaReceiver:
    def __init__(self, port, baudrate):
        print(f"Opening serial port {port}...")
        self.serial = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        self.serial.reset_input_buffer()
        print("Serial port opened\n")

    def read_framed_packet(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            byte = self.serial.read(1)
            if len(byte) > 0 and byte[0] == FRAME_START:
                next_byte = self.serial.read(1)
                if len(next_byte) > 0 and next_byte[0] == FRAME_START:
                    size_bytes = self.serial.read(2)
                    if len(size_bytes) == 2:
                        size = (size_bytes[0] << 8) | size_bytes[1]
                        data = self.serial.read(size)
                        if len(data) == size:
                            end1 = self.serial.read(1)
                            end2 = self.serial.read(1)
                            if len(end1) > 0 and len(end2) > 0:
                                if end1[0] == FRAME_END and end2[0] == FRAME_END:
                                    return data
        return None

    def receive_transmission(self, timeout=120):
        print("=" * 60)
        print("WAITING FOR TRANSMISSION")
        print("=" * 60)

        header_found = False
        start_time   = time.time()
        num_packets  = 0
        mel_bins     = MEL_BINS
        time_frames  = TIME_FRAMES

        while not header_found and time.time() - start_time < 60:
            packet = self.read_framed_packet(timeout=10)
            if packet and len(packet) >= 6:
                if packet[0] == 0xFF and packet[1] == 0xFF:
                    num_packets = (packet[2] << 8) | packet[3]
                    mel_bins    = packet[4]
                    time_frames = packet[5]
                    header_found = True
                    print(f"\nHeader received")
                    print(f"  Expecting {num_packets} packets")
                    print(f"  Mel size: {mel_bins}x{time_frames}\n")
                    break

        if not header_found:
            print("No header found\n")
            return None

        if mel_bins != MEL_BINS or time_frames != TIME_FRAMES:
            print(f"  WARNING: header reports {mel_bins}x{time_frames} but expected {MEL_BINS}x{TIME_FRAMES}\n")

        print("RECEIVING DATA PACKETS")
        print("-" * 60)

        packets       = {}
        start_time    = time.time()
        last_progress = 0

        while len(packets) < num_packets and time.time() - start_time < timeout:
            packet = self.read_framed_packet(timeout=10)
            if packet and len(packet) >= 3:
                packet_id = (packet[0] << 8) | packet[1]
                checksum  = packet[2]
                data      = packet[3:]

                computed_checksum = 0
                for b in data:
                    computed_checksum ^= b

                if computed_checksum != checksum:
                    print(f"  Packet {packet_id}: checksum mismatch, skipping")
                    continue

                if 0 <= packet_id < num_packets:
                    packets[packet_id] = data
                    progress = len(packets)
                    if progress != last_progress and (progress % 10 == 0 or progress == num_packets):
                        print(f"  Received: {progress}/{num_packets} packets")
                        last_progress = progress

        print("-" * 60)
        print(f"Received {len(packets)}/{num_packets} packets\n")

        if len(packets) < num_packets * 0.8:
            print(f"Too few packets: {len(packets)}/{num_packets}\n")
            return None

        print("RECONSTRUCTING MEL SPECTROGRAM")
        print("-" * 60)

        sorted_ids = sorted(packets.keys())
        mel_data   = b''.join([packets[i] for i in sorted_ids])
        mel_uint8  = np.frombuffer(mel_data, dtype=np.uint8)

        expected_size = mel_bins * time_frames
        actual_size   = len(mel_uint8)

        if actual_size < expected_size:
            print(f"  Padding: {actual_size} to {expected_size} bytes")
            mel_uint8 = np.pad(mel_uint8, (0, expected_size - actual_size), 'constant')
        elif actual_size > expected_size:
            print(f"  Truncating: {actual_size} to {expected_size} bytes")
            mel_uint8 = mel_uint8[:expected_size]

        mel_float       = mel_uint8.astype(np.float32) / 255.0
        mel_spectrogram = mel_float.reshape(mel_bins, time_frames)

        print(f"  Shape: {mel_spectrogram.shape}")
        print(f"  Range: [{mel_float.min():.3f}, {mel_float.max():.3f}]")
        print(f"  Mean:  {mel_float.mean():.3f}")
        print("-" * 60)
        print("Reconstruction complete\n")

        return mel_spectrogram

    def close(self):
        self.serial.close()


# ========================================
# ML CLASSIFIER
# ========================================
class MosquitoClassifier:
    def __init__(self, model1_path, model2_path):
        print("=" * 60)
        print("LOADING ML MODELS")
        print("=" * 60)

        print(f"\nModel 1: {model1_path}")
        if not os.path.exists(model1_path):
            raise FileNotFoundError(f"Model 1 not found: {model1_path}")

        self.interpreter1 = tf.lite.Interpreter(model_path=model1_path)
        self.interpreter1.allocate_tensors()
        self.input_details1  = self.interpreter1.get_input_details()
        self.output_details1 = self.interpreter1.get_output_details()
        print(f"  Input shape:  {self.input_details1[0]['shape']}")
        print(f"  Output shape: {self.output_details1[0]['shape']}")
        print("  Loaded")

        print(f"\nModel 2: {model2_path}")
        if not os.path.exists(model2_path):
            raise FileNotFoundError(f"Model 2 not found: {model2_path}")

        self.interpreter2 = tf.lite.Interpreter(model_path=model2_path)
        self.interpreter2.allocate_tensors()
        self.input_details2  = self.interpreter2.get_input_details()
        self.output_details2 = self.interpreter2.get_output_details()
        print(f"  Input shape:  {self.input_details2[0]['shape']}")
        print(f"  Output shape: {self.output_details2[0]['shape']}")
        print("  Loaded")

        print("\n" + "=" * 60)

    def preprocess(self, mel_spec, target_shape):
        h, w = target_shape[1], target_shape[2]
        if mel_spec.shape != (h, w):
            print(f"  Resizing {mel_spec.shape} to ({h}, {w})")
            from PIL import Image
            mel_spec = np.array(Image.fromarray(mel_spec).resize((w, h)))
        mel_spec = mel_spec[:, :, np.newaxis]
        return mel_spec[np.newaxis, :, :, :].astype(np.float32)

    def classify(self, mel_spec):
        result = {
            'timestamp':            datetime.now().isoformat(),
            'is_mosquito':          False,
            'is_dengue_vector':     False,
            'final_classification': 'Background Sound',
            'stage1':               {},
            'stage2':               {}
        }

        # ===== STAGE 1: MOSQUITO DETECTION =====
        print("=" * 60)
        print("STAGE 1: MOSQUITO DETECTION")
        print("=" * 60)

        mel_input1 = self.preprocess(mel_spec, self.input_details1[0]['shape'])
        self.interpreter1.set_tensor(self.input_details1[0]['index'], mel_input1)
        self.interpreter1.invoke()
        output1 = self.interpreter1.get_tensor(self.output_details1[0]['index'])[0]

        background_prob = float(output1[0])
        mosquito_prob   = float(output1[1])

        result['stage1'] = {
            'mosquito_probability':   mosquito_prob,
            'background_probability': background_prob
        }

        print(f"\n  Background: {background_prob:.3f}")
        print(f"  Mosquito:   {mosquito_prob:.3f}")
        print(f"  Threshold:  {MODEL1_THRESHOLD}")

        if mosquito_prob < MODEL1_THRESHOLD:
            result['is_mosquito']          = False
            result['final_classification'] = 'Background Sound'
            print(f"\n  REJECTED: Background sound")
            print("=" * 60 + "\n")
            return result

        result['is_mosquito'] = True
        print(f"\n  PASSED: Mosquito detected")
        print("=" * 60 + "\n")

        # ===== STAGE 2: SPECIES CLASSIFICATION =====
        print("=" * 60)
        print("STAGE 2: SPECIES CLASSIFICATION")
        print("=" * 60)

        mel_input2 = self.preprocess(mel_spec, self.input_details2[0]['shape'])
        self.interpreter2.set_tensor(self.input_details2[0]['index'], mel_input2)
        self.interpreter2.invoke()
        output2 = self.interpreter2.get_tensor(self.output_details2[0]['index'])[0]

        predicted_idx     = int(np.argmax(output2))
        predicted_species = SPECIES_LABELS[predicted_idx]
        confidence        = float(output2[predicted_idx])

        result['stage2'] = {
            'predicted_species': predicted_species,
            'confidence':        confidence,
            'probabilities':     {
                SPECIES_LABELS[i]: float(output2[i])
                for i in range(len(SPECIES_LABELS))
            }
        }

        print(f"\n  Top 3 predictions:")
        top_indices = np.argsort(output2)[-3:][::-1]
        for i in top_indices:
            print(f"    {SPECIES_LABELS[i]}: {output2[i]:.3f}")

        print(f"\n  Confidence threshold: {MODEL2_THRESHOLD}")

        if confidence >= MODEL2_THRESHOLD:
            if predicted_species in DENGUE_VECTORS:
                result['is_dengue_vector']     = True
                result['final_classification'] = f'DENGUE VECTOR DETECTED: {predicted_species}'
                print(f"  DENGUE VECTOR DETECTED: {predicted_species}")
                print(f"  Confidence: {confidence:.3f}")
            else:
                result['is_dengue_vector']     = False
                result['final_classification'] = f'Non-Aedes Mosquito: {predicted_species}'
                print(f"  Non-Aedes mosquito detected: {predicted_species}")
                print(f"  Confidence: {confidence:.3f}")
        else:
            result['is_dengue_vector']     = False
            result['final_classification'] = 'Unknown/Non-dengue Carrier Mosquito'
            print(f"  LOW CONFIDENCE: Unknown/Non-dengue Carrier Mosquito")
            print(f"  Best guess: {predicted_species} ({confidence:.3f})")

        print("=" * 60 + "\n")
        return result


# ========================================
# DATA LOGGER
# ========================================
class DataLogger:
    def __init__(self):
        self.log_dir  = _GDRIVE / 'logs'
        self.data_dir = _GDRIVE / 'captured_data'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        session_id       = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.data_dir / session_id
        self.session_dir.mkdir(exist_ok=True)

        self.log_file      = self.log_dir / f'detections_{session_id}.jsonl'
        self.stats_file    = self.log_dir / 'stats.json'
        self.live_log_file = self.log_dir / 'live_log.txt'

        self.stats = {
            'total':           0,
            'mosquito':        0,
            'background':      0,
            'dengue_vector':   0,
            'non_aedes':       0,
            'unknown':         0,
            'species':         {}
        }

        with open(self.live_log_file, 'w', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  System started (v4.1)\n")
            f.write("=" * 60 + "\n")

        self._write_stats()

        print("=" * 60)
        print("DATA LOGGING")
        print("=" * 60)
        print(f"  Session dir:  {self.session_dir}")
        print(f"  Detections:   {self.log_file}")
        print(f"  Stats:        {self.stats_file}")
        print(f"  Live log:     {self.live_log_file}")
        print("=" * 60 + "\n")

    def log(self, mel_spec, result):
        self.stats['total'] += 1
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        capture_id = self.stats['total']
        mel_file   = self.session_dir / f'capture_{capture_id:04d}.npy'
        np.save(mel_file, mel_spec)

        if result['is_mosquito']:
            self.stats['mosquito'] += 1
            final      = result['final_classification']
            confidence = result.get('stage2', {}).get('confidence', 0.0)

            if result['is_dengue_vector']:
                self.stats['dengue_vector'] += 1
                species = result['stage2']['predicted_species']
                self.stats['species'][species] = self.stats['species'].get(species, 0) + 1

                log_entry = {
                    'capture_id':     capture_id,
                    'timestamp':      result['timestamp'],
                    'mel_file':       str(mel_file),
                    'classification': result
                }
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

                with open(self.live_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}]  DENGUE VECTOR - {species} - {confidence*100:.1f}% confidence\n")

            elif 'Non-Aedes' in final:
                self.stats['non_aedes'] += 1
                species = result['stage2']['predicted_species']
                self.stats['species'][species] = self.stats['species'].get(species, 0) + 1

                with open(self.live_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}]  Non-Aedes - {species} - {confidence*100:.1f}% confidence\n")

            else:
                self.stats['unknown'] += 1
                with open(self.live_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}]  Unknown/Non-dengue Carrier Mosquito - saved as {mel_file.name}")

        else:
            self.stats['background'] += 1
            with open(self.live_log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}]  Background Sound - saved as {mel_file.name}\n")

        self._write_stats()

        if self.stats['total'] % 5 == 0:
            self.print_stats()

    def _write_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump({
                'total':         self.stats['total'],
                'mosquito':      self.stats['mosquito'],
                'background':    self.stats['background'],
                'dengue_vector': self.stats['dengue_vector'],
                'non_aedes':     self.stats['non_aedes'],
                'unknown':       self.stats['unknown'],
                'species':       self.stats['species'],
                'updated_at':    datetime.now().isoformat()
            }, f, indent=2)

    def print_stats(self):
        print("\n" + "=" * 60)
        print(f"STATISTICS ({self.stats['total']} transmissions received)")
        print("=" * 60)

        total      = self.stats['total']
        mosquito   = self.stats['mosquito']
        background = self.stats['background']

        print(f"  Mosquitoes:      {mosquito:3d} ({mosquito/max(1,total)*100:5.1f}%)")
        print(f"  Background:      {background:3d} ({background/max(1,total)*100:5.1f}%)")
        print(f"  Dengue vectors:  {self.stats['dengue_vector']:3d}")
        print(f"  Non-Aedes:       {self.stats['non_aedes']:3d}")
        print(f"  Unknown:         {self.stats['unknown']:3d}")

        if self.stats['species']:
            print("\n  Species breakdown:")
            for species, count in sorted(
                self.stats['species'].items(), key=lambda x: x[1], reverse=True
            ):
                tag = " [DENGUE VECTOR]" if species in DENGUE_VECTORS else ""
                print(f"    {species}: {count}{tag}")

        print("=" * 60 + "\n")


# ========================================
# MAIN SYSTEM
# ========================================
def main():
    print("\n" + "=" * 60)
    print("MOSQUITO DETECTION AND CLASSIFICATION SYSTEM (v4.1)")
    print("MSU-IIT Thesis Project - Chene")
    print("Focused on Aedes dengue vector detection")
    print("=" * 60 + "\n")

    receiver = None
    logger   = None

    try:
        receiver   = LoRaReceiver(SERIAL_PORT, BAUD_RATE)
        classifier = MosquitoClassifier(MODEL1_PATH, MODEL2_PATH)
        logger     = DataLogger()

        print("=" * 60)
        print("SYSTEM INITIALIZED AND READY")
        print("=" * 60)
        print("\nWaiting for transmissions...\n")

        while True:
            mel_spec = receiver.receive_transmission(timeout=120)

            if mel_spec is not None:
                result = classifier.classify(mel_spec)
                logger.log(mel_spec, result)

                print("=" * 60)
                print(f"FINAL RESULT: {result['final_classification']}")
                print("=" * 60 + "\n")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("STOPPING SYSTEM")
        print("=" * 60)
        if logger:
            logger.print_stats()
        print("System stopped gracefully\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if receiver:
            try:
                receiver.close()
            except Exception:
                pass
        print("\nCleanup complete\n")


if __name__ == '__main__':
    main()