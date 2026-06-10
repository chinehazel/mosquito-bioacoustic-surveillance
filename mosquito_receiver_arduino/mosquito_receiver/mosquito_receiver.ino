/*
 * ========================================
 * MOSQUITO RECEIVER ARDUINO - FINAL VERSION
 * MSU-IIT Thesis Project
 * Arduino Nano 33 BLE Sense Rev2
 * ========================================
 */

#include <SPI.h>
#include <LoRa.h>

// ===== LORA PINS =====
#define LORA_CS    10   // NSS
#define LORA_RST   9    // RESET
#define LORA_IRQ   8    // DIO0

// ===== LORA SETTINGS =====
#define LORA_FREQUENCY 915E6
#define LORA_BANDWIDTH 125E3
#define LORA_SPREADING 7

// ===== FRAMING =====
#define FRAME_START 0xAA
#define FRAME_END   0x55

// ===== DIAGNOSTICS =====
unsigned long packetsReceived = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10); // Receiver always has USB so this is safe

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("\n======================================");
  Serial.println("MOSQUITO RECEIVER - FINAL VERSION");
  Serial.println("MSU-IIT Thesis Project");
  Serial.println("======================================\n");

  // Init LoRa
  SPI.begin();
  LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);

  if (!LoRa.begin(LORA_FREQUENCY)) {
    Serial.println("ERROR: LoRa init failed!");
    // Fast blink forever = error
    while (1) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(200);
    }
  }

  LoRa.setSpreadingFactor(LORA_SPREADING);
  LoRa.setSignalBandwidth(LORA_BANDWIDTH);
  LoRa.setSyncWord(0x12);
  LoRa.enableCrc();

  Serial.println("✓ LoRa OK");
  Serial.print("  Frequency : "); Serial.print(LORA_FREQUENCY / 1E6); Serial.println(" MHz");
  Serial.print("  Bandwidth : "); Serial.print(LORA_BANDWIDTH / 1E3); Serial.println(" kHz");
  Serial.print("  SF        : "); Serial.println(LORA_SPREADING);
  Serial.println();

  // Ready - blink 3x
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(200);
    digitalWrite(LED_BUILTIN, LOW);
    delay(200);
  }

  Serial.println("✓ READY - Waiting for packets...\n");
}

void loop() {
  int packetSize = LoRa.parsePacket();

  if (packetSize > 0) {

    // LED on while receiving
    digitalWrite(LED_BUILTIN, HIGH);

    // Read packet into buffer
    uint8_t packet[256];
    int bytesRead = 0;

    while (LoRa.available() && bytesRead < 256) {
      packet[bytesRead++] = LoRa.read();
    }

    // Signal quality
    int rssi = LoRa.packetRssi();
    float snr  = LoRa.packetSnr();

    // Diagnostics every 10 packets
    packetsReceived++;
    if (packetsReceived % 10 == 0) {
      Serial.print("["); Serial.print(packetsReceived); Serial.print("] ");
      Serial.print("Size: "); Serial.print(bytesRead);
      Serial.print(" | RSSI: "); Serial.print(rssi);
      Serial.print(" dBm | SNR: "); Serial.print(snr);
      Serial.println(" dB");
    }

    // Send framed packet to Python via USB serial
    // Format: [0xAA][0xAA][size_high][size_low][data...][0x55][0x55]
    Serial.write(FRAME_START);
    Serial.write(FRAME_START);
    Serial.write((uint8_t)(bytesRead >> 8));
    Serial.write((uint8_t)(bytesRead & 0xFF));
    Serial.write(packet, bytesRead);
    Serial.write(FRAME_END);
    Serial.write(FRAME_END);

    digitalWrite(LED_BUILTIN, LOW);
  }

  delay(5);}