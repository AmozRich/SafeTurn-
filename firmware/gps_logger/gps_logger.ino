// ==========================================
// SafeTurn+ ESP32 GPS Breadcrumb Logger
// ==========================================
// 
// This script runs on the ESP32 and continuously polls a NEO-6M GPS module.
// It outputs raw CSV strings formatted as:
// LATITUDE,LONGITUDE,SPEED_KMH,HEADING,SATELLITES
//
// You can capture this output using a simple Python script on your laptop 
// or PuTTY to save your driven route as a 'breadcrumb' map.
//
// Dependencies:
// - TinyGPSPlus (Install via Arduino Library Manager)
// - HardwareSerial (Built-in)

#include <TinyGPS++.h>
#include <HardwareSerial.h>

// --- CONFIGURATION ---
// Adjust pins based on your ESP32 board
static const int RXPin = 16, TXPin = 17; 
static const uint32_t GPSBaud = 9600;

// Update frequency (ms) between writing a breadcrumb
static const unsigned long LOGGING_INTERVAL_MS = 1000; 

// The TinyGPS++ object
TinyGPSPlus gps;

// The serial connection to the GPS device
HardwareSerial serial_gps(1); // Use UART1

unsigned long last_log_time = 0;

void setup()
{
  // Main serial for CSV output to the Laptop
  Serial.begin(115200);

  // Initialize the GPS module serial port
  serial_gps.begin(GPSBaud, SERIAL_8N1, RXPin, TXPin);

  // Print the CSV Header once at startup
  Serial.println("LATITUDE,LONGITUDE,SPEED_KMH,HEADING,SATELLITES");
}

void loop()
{
  // 1. Feed the GPS parser with raw NMEA sentences
  while (serial_gps.available() > 0) {
    gps.encode(serial_gps.read());
  }

  // 2. Check if it's time to log a breadcrumb
  if (millis() - last_log_time > LOGGING_INTERVAL_MS) {
    last_log_time = millis();

    // 3. Only log if we have a valid fix
    if (gps.location.isValid() && gps.location.isUpdated()) {
      
      double lat = gps.location.lat();
      double lng = gps.location.lng();
      double speed = gps.speed.isValid() ? gps.speed.kmph() : 0.0;
      double heading = gps.course.isValid() ? gps.course.deg() : 0.0;
      int satellites = gps.satellites.isValid() ? gps.satellites.value() : 0;

      // Output strict CSV format
      Serial.print(lat, 6);   Serial.print(",");
      Serial.print(lng, 6);   Serial.print(",");
      Serial.print(speed, 2); Serial.print(",");
      Serial.print(heading, 2); Serial.print(",");
      Serial.println(satellites);
      
    } else {
      // Optional: Wait message if no satellite fix yet.
      // We start it with '#' so a CSV parser can ignore this comment line.
      if (gps.satellites.value() < 4) {
         Serial.println("# WAITING_FOR_FIX");
      }
    }
  }

  // Safety check: if wiring is wrong
  if (millis() > 5000 && gps.charsProcessed() < 10) {
    Serial.println("# ERROR: No GPS data received: check wiring.");
    while(true); // Halt
  }
}
