/*
 * ========================================
 * MEL SPECTROGRAM GENERATOR — v2 (rollback)
 *
 * Rolled back to match mosquito_vs_background.tflite training:
 *   - 2-second window (was 1 second in v3)
 *   - 122 time frames (was 59)
 *   - Hamming window (was Hann)
 *   - Fixed normalization: (melDB + 80) / 80
 *   - No HPF (old model was trained without it)
 *   - FFT size 1024, hop 256
 *
 * Memory footprint:
 *   - FFT working buffers: 2 x 1024 x 4 bytes = 8 KB
 *   - mel_spec_output: 128 x 122 x 4 bytes = ~62 KB
 *   - Plus the 32000-sample audio buffer in the .ino
 * ========================================
 */

#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <Arduino.h>

// ===== PARAMETERS (must match mosquito_vs_background.tflite training) =====
#define MEL_BINS      128
#define TIME_FRAMES   122      // 2s @ 16kHz with FFT=1024, hop=256
#define FFT_SIZE      1024
#define HOP_LENGTH    256
#define WIN_LENGTH    1024
#define SAMPLE_RATE   16000

int   mel_bin_boundaries[MEL_BINS + 2];
bool  filterbank_initialized = false;

// Pre-computed Hamming window (matches MATLAB hamming() periodic)
float hamming_window[WIN_LENGTH];
bool  hamming_initialized = false;


// ============================================================
// Mel scale conversions
// ============================================================
float hzToMel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}


// ============================================================
// Iterative radix-2 FFT (in-place)
// ============================================================
void simpleFFT(float* real, float* imag, int n) {
  int j = 0;
  for (int i = 1; i < n - 1; i++) {
    int bit = n >> 1;
    while (j >= bit) {
      j -= bit;
      bit >>= 1;
    }
    j += bit;

    if (i < j) {
      float temp = real[i]; real[i] = real[j]; real[j] = temp;
      temp = imag[i]; imag[i] = imag[j]; imag[j] = temp;
    }
  }

  for (int len = 2; len <= n; len <<= 1) {
    float angle   = -2.0f * PI / len;
    float wlen_re = cosf(angle);
    float wlen_im = sinf(angle);

    for (int i = 0; i < n; i += len) {
      float w_re = 1.0f;
      float w_im = 0.0f;

      for (int k = 0; k < len / 2; k++) {
        float u_re = real[i + k];
        float u_im = imag[i + k];
        float v_re = real[i + k + len/2] * w_re - imag[i + k + len/2] * w_im;
        float v_im = real[i + k + len/2] * w_im + imag[i + k + len/2] * w_re;

        real[i + k]         = u_re + v_re;
        imag[i + k]         = u_im + v_im;
        real[i + k + len/2] = u_re - v_re;
        imag[i + k + len/2] = u_im - v_im;

        float w_temp = w_re;
        w_re = w_re * wlen_re - w_im * wlen_im;
        w_im = w_temp * wlen_im + w_im * wlen_re;
      }
    }
  }
}


// ============================================================
// Mel filterbank boundaries
// ============================================================
void initMelFilterbank() {
  if (filterbank_initialized) return;

  Serial.println("  Computing mel boundaries...");

  float min_mel = hzToMel(0);
  float max_mel = hzToMel(SAMPLE_RATE / 2.0f);

  for (int i = 0; i < MEL_BINS + 2; i++) {
    float mel = min_mel + (max_mel - min_mel) * i / (MEL_BINS + 1);
    float hz  = melToHz(mel);
    mel_bin_boundaries[i] = (int)floorf((FFT_SIZE + 1) * hz / SAMPLE_RATE);
  }

  filterbank_initialized = true;
  Serial.println("  Mel boundaries ready");
}


float getMelFilterValue(int mel_bin, int fft_bin) {
  int left   = mel_bin_boundaries[mel_bin];
  int center = mel_bin_boundaries[mel_bin + 1];
  int right  = mel_bin_boundaries[mel_bin + 2];

  if (fft_bin >= left && fft_bin < center) {
    if (center != left) {
      return (float)(fft_bin - left) / (center - left);
    }
  }

  if (fft_bin >= center && fft_bin < right) {
    if (right != center) {
      return (float)(right - fft_bin) / (right - center);
    }
  }

  return 0.0f;
}


// ============================================================
// Hamming window -- matches MATLAB hamming(N, 'periodic')
// w(n) = 0.54 - 0.46 * cos(2*pi*n / N)   for n = 0..N-1
// ============================================================
void initHammingWindow() {
  if (hamming_initialized) return;
  for (int i = 0; i < WIN_LENGTH; i++) {
    hamming_window[i] = 0.54f - 0.46f * cosf(2.0f * PI * i / WIN_LENGTH);
  }
  hamming_initialized = true;
}


// ============================================================
// FFT + power spectrum for one frame
// ============================================================
void computePowerSpectrum(int16_t* audio_frame, float* power_spectrum) {
  static float fft_real[FFT_SIZE];
  static float fft_imag[FFT_SIZE];

  if (!hamming_initialized) initHammingWindow();

  for (int i = 0; i < FFT_SIZE; i++) {
    fft_real[i] = ((float)audio_frame[i] / 32768.0f) * hamming_window[i];
    fft_imag[i] = 0.0f;
  }

  simpleFFT(fft_real, fft_imag, FFT_SIZE);

  for (int i = 0; i < FFT_SIZE / 2 + 1; i++) {
    power_spectrum[i] = fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i];
  }
}


// ============================================================
// Apply mel filterbank
// ============================================================
void applyMelFilterbank(float* power_spectrum, float* mel_spectrum) {
  for (int i = 0; i < MEL_BINS; i++) {
    mel_spectrum[i] = 0.0f;
    for (int j = 0; j < FFT_SIZE / 2 + 1; j++) {
      float fv = getMelFilterValue(i, j);
      if (fv > 0.0f) {
        mel_spectrum[i] += power_spectrum[j] * fv;
      }
    }
  }
}


// ============================================================
// MAIN ENTRY POINT
//
// audio        : int16 PCM, 32000 samples (2 seconds at 16kHz)
// audio_length : actual length of audio buffer
// mel_spec_out : flat float array, size MEL_BINS * TIME_FRAMES
//                Output values normalized by (melDB + 80) / 80
// ============================================================
void audioToMelSpectrogram(int16_t* audio, int audio_length, float* mel_spec_out) {
  initMelFilterbank();
  initHammingWindow();

  // No HPF -- matches old training pipeline

  int num_frames = (audio_length - FFT_SIZE) / HOP_LENGTH + 1;
  if (num_frames > TIME_FRAMES) {
    num_frames = TIME_FRAMES;
  }

  static float power_spectrum[FFT_SIZE / 2 + 1];

  for (int frame = 0; frame < num_frames; frame++) {
    int start_idx = frame * HOP_LENGTH;

    int16_t audio_frame[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; i++) {
      if (start_idx + i < audio_length) {
        audio_frame[i] = audio[start_idx + i];
      } else {
        audio_frame[i] = 0;
      }
    }

    computePowerSpectrum(audio_frame, power_spectrum);

    float mel_spectrum[MEL_BINS];
    applyMelFilterbank(power_spectrum, mel_spectrum);

    for (int bin = 0; bin < MEL_BINS; bin++) {
      mel_spec_out[bin * TIME_FRAMES + frame] = mel_spectrum[bin];
    }
  }

  // Zero out unfilled frames
  for (int frame = num_frames; frame < TIME_FRAMES; frame++) {
    for (int bin = 0; bin < MEL_BINS; bin++) {
      mel_spec_out[bin * TIME_FRAMES + frame] = 0.0f;
    }
  }

  // Convert to dB then apply fixed normalization
  // matches MATLAB: melSpec = (pow2db(mel + eps) + 80) / 80
  const float epsilon = 1e-10f;

  for (int i = 0; i < MEL_BINS * TIME_FRAMES; i++) {
    float db = 10.0f * log10f(mel_spec_out[i] + epsilon);
    float normalized = (db + 80.0f) / 80.0f;
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    mel_spec_out[i] = normalized;
  }
}

#endif  // MEL_SPECTROGRAM_H