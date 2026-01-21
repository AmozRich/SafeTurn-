#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <TinyGPS++.h>
#include <ArduinoJson.h>

// --- Config ---
#define GPS_RX 16
#define GPS_TX 17
#define GSM_RX 4
#define GSM_TX 5
#define SERIAL_BAUD 115200

// --- Objects ---
Adafruit_MPU6050 mpu;
TinyGPSPlus gps;
HardwareSerial gpsSerial(1); // UART 1
HardwareSerial gsmSerial(2); // UART 2

// --- State ---
unsigned long lastTelemetryTime = 0;
const int TELEMETRY_INTERVAL = 50; // 20Hz
bool crashDetected = false;

void setup() {
  Serial.begin(SERIAL_BAUD);
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);
  gsmSerial.begin(9600, SERIAL_8N1, GSM_RX, GSM_TX);

  // MPU Init
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) { delay(10); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void loop() {
  // 1. Read GPS Continuously (Non-blocking)
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // 2. Read IMU
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // 3. Crash Detection (Simple > 3G on X/Y)
  float accel_mag = sqrt(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y);
  if (accel_mag > 30.0 && !crashDetected) { // ~3G (3 * 9.8)
    crashDetected = true;
    triggerSMS();
  }

  // 4. Telemetry Loop (20Hz)
  if (millis() - lastTelemetryTime > TELEMETRY_INTERVAL) {
    lastTelemetryTime = millis();
    sendTelemetry(g.gyro.z, gps.location.lat(), gps.location.lng(), gps.speed.kmph());
  }
}

void sendTelemetry(float yaw_rate, double lat, double lng, double speed) {
  StaticJsonDocument<200> doc;
  doc["yaw"] = yaw_rate; // rad/s
  doc["lat"] = lat;
  doc["lng"] = lng;
  doc["spd"] = speed;
  doc["crash"] = crashDetected;

  serializeJson(doc, Serial);
  Serial.println();
}

void triggerSMS() {
  gsmSerial.println("AT+CMGF=1"); 
  delay(100);
  gsmSerial.println("AT+CMGS=\"+1234567890\""); // Replace with number
  delay(100);
  gsmSerial.print("CRASH DETECTED! Lat: ");
  gsmSerial.print(gps.location.lat(), 6);
  gsmSerial.print(" Lon: ");
  gsmSerial.print(gps.location.lng(), 6);
  delay(100);
  gsmSerial.write(26);
}
