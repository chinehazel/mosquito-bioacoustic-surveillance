/*
 * ========================================
 * MOSQUITO TRANSMITTER — v3.5
 * MSU-IIT Thesis Project
 * Arduino Nano 33 BLE Sense Rev2
 *
 * Changes from v3.4:
 *   - Added [ModerateTone] serial flag for spectral flatness >= 0.12
 *     to distinguish moderately tonal captures (0.12 <= Flat <= 0.15)
 *     from clearly tonal captures (Flat < 0.12); frame still passes.
 *   - Setup print block updated to reflect moderate tonality threshold.
 *
 * Changes from v3.3:
 *   - Pre-filter thresholds updated to match thesis specification:
 *       E  : 0.3 <= E <= 20.0
 *       R  : 0.039 <= R <= 0.10  (band: 200-950 Hz)
 *       dB : >= -15.0
 *       P/A: >= 2.0
 *       Flat: 0.03 <= Flat <= 0.15
 *   - Mosquito frequency band updated to 200-950 Hz
 *   - High-R bypass gate removed
 *   - WeakAndDull gate removed
 *   - PDM gain reverted to 30 for normal deployment
 *   - MAX_ENERGY_THRESHOLD added (E <= 20.0)
 *   - MAX_MOSQUITO_RATIO added (R <= 0.10)
 *
 * ICS-43434 Wiring:
 *   +3V3  -> VDD
 *   GND   -> GND & SEL (LR = left channel)
 *   CLK   -> PDM CLK (built-in)
 *   DATA  -> PDM DATA (built-in)
 *
 * Detection-range reality:
 *   Single mosquito wingbeat ~30-40 dB SPL at 10 cm.
 *   ICS-43434 self-noise ~32 dB SPL.
 *   Practical detection range: 5-20 cm in quiet conditions.
 * ========================================
 */

#include <SPI.h>
#include <LoRa.h>
#include <PDM.h>
#include "mel_spectrogram.h"

// ===== TEST MODE =====
#define TEST_MODE 0

// ===== LORA PINS =====
#define LORA_CS    10
#define LORA_RST   9
#define LORA_IRQ   8

// ===== RGB LED PINS =====
#define LED_RED    5
#define LED_GREEN  6
#define LED_BLUE   7

// ===== AUDIO =====
#define DURATION_SEC   2
#define BUFFER_SIZE    (SAMPLE_RATE * DURATION_SEC)   // 32000 samples

// ===== LORA SETTINGS =====
#define LORA_FREQUENCY 915E6
#define LORA_BANDWIDTH 125E3
#define LORA_SPREADING 7
#define LORA_TX_POWER  20

// ===== PRE-FILTERING — matches thesis Section 4.5.1 =====
#define MIN_ENERGY_THRESHOLD    0.3f     // E >= 0.3
#define MAX_ENERGY_THRESHOLD    20.0f    // E <= 20.0
#define MOSQUITO_FREQ_MIN       200      // Hz — band lower bound
#define MOSQUITO_FREQ_MAX       950      // Hz — band upper bound
#define MIN_MOSQUITO_RATIO      0.039f   // R >= 0.039
#define MAX_MOSQUITO_RATIO      0.10f    // R <= 0.10
#define MIN_BAND_DB             -15.0f   // dB >= -15.0
#define MIN_PEAK_TO_AVG         2.0f     // P/A >= 2.0
#define MIN_FLATNESS            0.03f    // Flat >= 0.03
#define MAX_FLATNESS            0.15f    // Flat <= 0.15
#define MODERATE_TONE_THRESHOLD 0.12f    // Flat >= 0.12: moderate tonality flag
#define ANALYSIS_FFT_SIZE       512
#define NUM_SCAN_WINDOWS        5

// ===== BUFFERS =====
int16_t audioBuffer[BUFFER_SIZE];
uint8_t melBuffer[MEL_BINS * TIME_FRAMES];
float   melSpectrogram[MEL_BINS * TIME_FRAMES];

// ===== PDM HELPERS =====
short tempBuffer[512];
volatile int32_t totalSamplesCaptured = 0;

// ===== STATS =====
unsigned long totalCaptures = 0;
unsigned long transmissions = 0;

struct FrequencyAnalysis {
  float peakFrequency;
  float mosquitoEnergyRatio;
  float peakPower;
  float totalPower;
  float peakToAvgRatio;
  float mosquitoBandDb;
  int   bestWindow;
};

// ===== FUNCTION DECLARATIONS =====
float             calculateRMSEnergy(int16_t* audio, int length);
FrequencyAnalysis analyzeFrequencyBand(int16_t* audio, int length);
float             calculateSpectralFlatness(int16_t* audio, int length);
void              captureAudio();
void              onPDMdata();
void              setLED(uint8_t r, uint8_t g, uint8_t b);
bool              sendMelViaLoRa(uint8_t* mel_data, int data_size);

// ===== PDM CALLBACK =====
void onPDMdata() {
  int bytes = PDM.available();
  if (bytes > 0) {
    PDM.read(tempBuffer, bytes);
    int samples = bytes / 2;
    for (int i = 0; i < samples && totalSamplesCaptured < BUFFER_SIZE; i++) {
      audioBuffer[totalSamplesCaptured++] = tempBuffer[i];
    }
  }
}

void captureAudio() {
  totalSamplesCaptured = 0;
  unsigned long start = millis();
  while ((millis() - start) < 2400) {
    if (totalSamplesCaptured >= BUFFER_SIZE) break;
    delay(5);
  }
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  unsigned long serialWait = millis();
  while (!Serial && millis() - serialWait < 3000);

  Serial.println("\n======================================");
  Serial.println("MOSQUITO TRANSMITTER (v3.5)");
  Serial.println("Mic: ICS-43434 (PDM)");
  Serial.println("MSU-IIT Thesis Project");
  Serial.println("======================================");

#if TEST_MODE
  Serial.println("*** TEST_MODE = 1 — pre-filters bypassed ***");
#else
  Serial.println("TEST_MODE = 0 — normal deployment mode.");
#endif

  Serial.println("\nPre-filter thresholds:");
  Serial.print("  E   : "); Serial.print(MIN_ENERGY_THRESHOLD);
  Serial.print(" <= E <= "); Serial.println(MAX_ENERGY_THRESHOLD);
  Serial.print("  R   : "); Serial.print(MIN_MOSQUITO_RATIO);
  Serial.print(" <= R <= "); Serial.println(MAX_MOSQUITO_RATIO);
  Serial.print("  Band: "); Serial.print(MOSQUITO_FREQ_MIN);
  Serial.print(" - "); Serial.print(MOSQUITO_FREQ_MAX); Serial.println(" Hz");
  Serial.print("  dB  : >= "); Serial.println(MIN_BAND_DB);
  Serial.print("  P/A : >= "); Serial.println(MIN_PEAK_TO_AVG);
  Serial.print("  Flat: "); Serial.print(MIN_FLATNESS);
  Serial.print(" <= Flat <= "); Serial.print(MAX_FLATNESS);
  Serial.println("  (>= 0.12: moderate tonality, [ModerateTone] flagged)");

  pinMode(LED_RED,   OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE,  OUTPUT);
  setLED(0, 0, 0);
  setLED(255, 255, 255); delay(500); setLED(0, 0, 0);

  // Init LoRa
  SPI.begin();
  LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);
  Serial.println("\nInitializing LoRa...");
  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("LoRa FAILED");
    while (1) { setLED(255,0,0); delay(300); setLED(0,0,0); delay(300); }
  }
  LoRa.setSpreadingFactor(LORA_SPREADING);
  LoRa.setSignalBandwidth(LORA_BANDWIDTH);
  LoRa.setTxPower(LORA_TX_POWER);
  LoRa.setSyncWord(0x12);
  Serial.println("LoRa OK");

  // Init PDM mic
  Serial.println("Initializing ICS-43434 (PDM)...");
  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(512);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Mic FAILED");
    while (1) { setLED(0,0,255); delay(300); setLED(0,0,0); delay(300); }
  }
  PDM.setGain(30);
  Serial.println("Mic OK (gain=30)");

  Serial.println("Initializing mel pipeline...");
  initMelFilterbank();
  Serial.println("  Mel size: 128 x 122");
  Serial.println("  FFT: 1024, hop: 256");
  Serial.println("  HPF: 300 Hz");
  Serial.println("Mel pipeline OK");

  for (int i = 0; i < 3; i++) {
    setLED(0,255,0); delay(200); setLED(0,0,0); delay(200);
  }
  Serial.println("\nSYSTEM READY\n");
}

// ===== MAIN LOOP =====
void loop() {
  totalCaptures++;
  Serial.print("[");
  Serial.print(totalCaptures);
  Serial.print("] ");

  setLED(0, 0, 255);
  captureAudio();
  setLED(0, 0, 0);

  // ----- Gate 1: RMS Energy -----
  float energy = calculateRMSEnergy(audioBuffer, BUFFER_SIZE);
  Serial.print("E:");
  Serial.print(energy, 2);

  if (energy < MIN_ENERGY_THRESHOLD || energy > MAX_ENERGY_THRESHOLD) {
    Serial.print(energy < MIN_ENERGY_THRESHOLD ? " [Quiet]" : " [TooLoud]");
#if !TEST_MODE
    Serial.println();
    delay(50);
    return;
#endif
  }
  Serial.print(" | ");

  // ----- Gate 2 & 3: Frequency band analysis (R and dB) -----
  FrequencyAnalysis freq = analyzeFrequencyBand(audioBuffer, BUFFER_SIZE);

  Serial.print("F:");
  Serial.print(freq.peakFrequency, 0);
  Serial.print("Hz R:");
  Serial.print(freq.mosquitoEnergyRatio, 3);
  Serial.print(" P/A:");
  Serial.print(freq.peakToAvgRatio, 1);
  Serial.print(" dB:");
  Serial.print(freq.mosquitoBandDb, 1);
  Serial.print(" W:");
  Serial.print(freq.bestWindow);

  // Gate 2: Band energy ratio
  if (freq.mosquitoEnergyRatio < MIN_MOSQUITO_RATIO ||
      freq.mosquitoEnergyRatio > MAX_MOSQUITO_RATIO) {
    Serial.print(freq.mosquitoEnergyRatio < MIN_MOSQUITO_RATIO ?
                 " [LowRatio]" : " [HighRatio]");
#if !TEST_MODE
    Serial.println();
    delay(50);
    return;
#endif
  }

  // Gate 3: In-band dB
  if (freq.mosquitoBandDb < MIN_BAND_DB) {
    Serial.print(" [LowdB]");
#if !TEST_MODE
    Serial.println();
    delay(50);
    return;
#endif
  }

  // Gate 4: Peak-to-average ratio
  if (freq.peakToAvgRatio < MIN_PEAK_TO_AVG) {
    Serial.print(" [LowPA]");
#if !TEST_MODE
    Serial.println();
    delay(50);
    return;
#endif
  }
  Serial.print(" | ");

  // ----- Gate 5: Spectral flatness -----
  float flatness = calculateSpectralFlatness(audioBuffer, BUFFER_SIZE);
  Serial.print("Flat:");
  Serial.print(flatness, 3);

  if (flatness < MIN_FLATNESS) {
    Serial.print(" [PureTone]");
#if !TEST_MODE
    Serial.println();
    delay(50);
    return;
#endif
  }
  if (flatness > MAX_FLATNESS) {
    Serial.print(" [TooNoisy]");
#if !TEST_MODE
    Serial.println();
    delay(50);
    return;
#endif
  }
  if (flatness >= MODERATE_TONE_THRESHOLD) {
    Serial.print(" [ModerateTone]");
  }

#if TEST_MODE
  Serial.println(" | TX [TEST_MODE]");
#else
  Serial.println(" | MOSQUITO CANDIDATE");
#endif

  // ----- Generate mel spectrogram -----
  Serial.print("  Mel...");
  setLED(255, 255, 0);
  audioToMelSpectrogram(audioBuffer, BUFFER_SIZE, melSpectrogram);
  Serial.println("done");

  for (int i = 0; i < MEL_BINS * TIME_FRAMES; i++) {
    melBuffer[i] = (uint8_t)(melSpectrogram[i] * 255.0f);
  }

  int  nonZero = 0, minVal = 255, maxVal = 0;
  long sum = 0;
  for (int i = 0; i < MEL_BINS * TIME_FRAMES; i++) {
    if (melBuffer[i] > 0)      nonZero++;
    if (melBuffer[i] < minVal) minVal = melBuffer[i];
    if (melBuffer[i] > maxVal) maxVal = melBuffer[i];
    sum += melBuffer[i];
  }
  Serial.print("  Mel stats: Min="); Serial.print(minVal);
  Serial.print(" Max=");             Serial.print(maxVal);
  Serial.print(" Avg=");             Serial.print(sum / (MEL_BINS * TIME_FRAMES));
  Serial.print(" NonZero=");         Serial.print(nonZero);
  Serial.print("/");                 Serial.println(MEL_BINS * TIME_FRAMES);

  Serial.println("  Transmitting...");
  setLED(255, 165, 0);

  if (sendMelViaLoRa(melBuffer, MEL_BINS * TIME_FRAMES)) {
    transmissions++;
    Serial.print("  SUCCESS | Total TX: ");
    Serial.println(transmissions);
    setLED(0, 255, 0);
    delay(2000);
  } else {
    Serial.println("  FAILED");
    setLED(255, 0, 0);
    delay(1000);
  }

  setLED(0, 0, 0);
  delay(1000);
}

// ============================================================
// LoRa transmission
// ============================================================
bool sendMelViaLoRa(uint8_t* mel_data, int data_size) {
  const int MAX_PACKET = 240;
  int num_packets = (data_size + MAX_PACKET - 1) / MAX_PACKET;

  Serial.print("    Packets: ");
  Serial.println(num_packets);

  bool headerSent = false;
  for (int retry = 0; retry < 3; retry++) {
    LoRa.beginPacket();
    LoRa.write(0xFF); LoRa.write(0xFF);
    LoRa.write((uint8_t)(num_packets >> 8));
    LoRa.write((uint8_t)(num_packets & 0xFF));
    LoRa.write((uint8_t)MEL_BINS);
    LoRa.write((uint8_t)TIME_FRAMES);
    if (LoRa.endPacket(true)) { headerSent = true; break; }
    delay(100);
  }
  if (!headerSent) { Serial.println("    Header FAILED"); return false; }

  delay(1000);

  int sent = 0;
  for (int i = 0; i < num_packets; i++) {
    int offset = i * MAX_PACKET;
    int size   = min(MAX_PACKET, data_size - offset);

    uint8_t checksum = 0;
    for (int j = 0; j < size; j++) checksum ^= mel_data[offset + j];

    for (int retry = 0; retry < 3; retry++) {
      LoRa.beginPacket();
      LoRa.write((uint8_t)(i >> 8));
      LoRa.write((uint8_t)(i & 0xFF));
      LoRa.write(checksum);
      LoRa.write(mel_data + offset, size);
      if (LoRa.endPacket(true)) { sent++; break; }
      delay(100);
    }

    if ((i + 1) % 10 == 0) {
      Serial.print("    Sent: "); Serial.print(i+1);
      Serial.print("/"); Serial.println(num_packets);
    }
    delay(500);
  }

  Serial.print("    Total sent: "); Serial.print(sent);
  Serial.print("/"); Serial.println(num_packets);
  return (sent >= num_packets * 0.9);
}

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  analogWrite(LED_RED, r); analogWrite(LED_GREEN, g); analogWrite(LED_BLUE, b);
}

float calculateRMSEnergy(int16_t* audio, int length) {
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    float s = (float)audio[i] / 32768.0f;
    sum += s * s;
  }
  return sqrtf(sum / length) * 1000.0f;
}

// ============================================================
// analyzeFrequencyBand — multi-window scanning
// Band: MOSQUITO_FREQ_MIN to MOSQUITO_FREQ_MAX (200-950 Hz)
// ============================================================
FrequencyAnalysis analyzeFrequencyBand(int16_t* audio, int length) {
  FrequencyAnalysis result;
  result.mosquitoEnergyRatio = 0.0f;
  result.peakToAvgRatio      = 0.0f;
  result.peakFrequency       = 0.0f;
  result.peakPower           = 0.0f;
  result.totalPower          = 0.0f;
  result.mosquitoBandDb      = -100.0f;
  result.bestWindow          = 0;

  static float fft_real[ANALYSIS_FFT_SIZE];
  static float fft_imag[ANALYSIS_FFT_SIZE];

  int step = (length - ANALYSIS_FFT_SIZE) / (NUM_SCAN_WINDOWS - 1);

  for (int w = 0; w < NUM_SCAN_WINDOWS; w++) {
    int start = w * step;

    for (int i = 0; i < ANALYSIS_FFT_SIZE; i++) {
      float window = 0.54f - 0.46f * cosf(2.0f * PI * i / ANALYSIS_FFT_SIZE);
      fft_real[i] = (start + i < length) ?
                    ((float)audio[start + i] / 32768.0f) * window : 0.0f;
      fft_imag[i] = 0.0f;
    }

    simpleFFT(fft_real, fft_imag, ANALYSIS_FFT_SIZE);

    float totalEnergy    = 0.0f;
    float mosquitoEnergy = 0.0f;
    float peakPower      = 0.0f;
    int   peakBin        = 0;
    int   bandBinCount   = 0;

    for (int i = 0; i < ANALYSIS_FFT_SIZE / 2 + 1; i++) {
      float freq  = (float)i * SAMPLE_RATE / ANALYSIS_FFT_SIZE;
      float power = fft_real[i]*fft_real[i] + fft_imag[i]*fft_imag[i];
      totalEnergy += power;

      if (freq >= MOSQUITO_FREQ_MIN && freq <= MOSQUITO_FREQ_MAX) {
        mosquitoEnergy += power;
        bandBinCount++;
        if (power > peakPower) { peakPower = power; peakBin = i; }
      }
    }

    float ratio = (totalEnergy > 0) ? (mosquitoEnergy / totalEnergy) : 0.0f;

    if (ratio > result.mosquitoEnergyRatio) {
      result.peakFrequency       = (float)peakBin * SAMPLE_RATE / ANALYSIS_FFT_SIZE;
      result.mosquitoEnergyRatio = ratio;
      result.peakPower           = peakPower;
      result.totalPower          = totalEnergy;
      result.bestWindow          = w + 1;

      if (bandBinCount > 0 && mosquitoEnergy > 0) {
        float avgInBand       = mosquitoEnergy / bandBinCount;
        result.peakToAvgRatio = (avgInBand > 0) ? (peakPower / avgInBand) : 0.0f;
      } else {
        result.peakToAvgRatio = 0.0f;
      }
      result.mosquitoBandDb = (mosquitoEnergy > 1e-10f) ?
                               10.0f * log10f(mosquitoEnergy) : -100.0f;
    }
  }

  return result;
}

float calculateSpectralFlatness(int16_t* audio, int length) {
  static float fft_real[ANALYSIS_FFT_SIZE];
  static float fft_imag[ANALYSIS_FFT_SIZE];

  int start = (length - ANALYSIS_FFT_SIZE) / 2;
  for (int i = 0; i < ANALYSIS_FFT_SIZE; i++) {
    fft_real[i] = (start + i < length) ?
                  (float)audio[start + i] / 32768.0f : 0.0f;
    fft_imag[i] = 0.0f;
  }

  simpleFFT(fft_real, fft_imag, ANALYSIS_FFT_SIZE);

  float logSum = 0.0f, arithmeticSum = 0.0f;
  int   count  = 0;

  for (int i = 10; i < ANALYSIS_FFT_SIZE / 2; i++) {
    float power = fft_real[i]*fft_real[i] + fft_imag[i]*fft_imag[i];
    if (power > 1e-10) {
      logSum        += log(power);
      arithmeticSum += power;
      count++;
    }
  }

  if (count == 0) return 0.0f;
  float geometricMean  = exp(logSum / count);
  float arithmeticMean = arithmeticSum / count;
  return (arithmeticMean > 0) ? (geometricMean / arithmeticMean) : 0.0f;
}
