#include <Wire.h>
#include <MPU6050.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>

// --- ESP32 Pin Configuration ---
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define GPS_RX_PIN 19
#define GPS_TX_PIN 18

MPU6050 mpu;
TinyGPSPlus gps;

void setup()
{
  Serial.begin(115200);

  // Initialize I2C with specified pins
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  // Initialize GPS Hardware Serial 2
  Serial2.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Serial.println("System Starting...");

  mpu.initialize();

  if (mpu.testConnection())
    Serial.println("MPU Connected");
  else
    Serial.println("MPU Failed");
}

void loop()
{

  // Always read GPS first
  while (Serial2.available())
  {
    gps.encode(Serial2.read());
  }

  // Read MPU
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Create JSON output
  Serial.print("{\"yaw\": ");
  // We use Gyro Z for approximate yaw rate (Z-axis rotation)
  // MPU6050 raw range is typically +/- 32768 for +/- 250 deg/s
  // Convert raw to deg/s
  float yaw_rate = gz / 131.0;
  Serial.print(yaw_rate);
  
  Serial.print(", \"accel_z\": ");
  float accel_z = az / 16384.0;
  Serial.print(accel_z);
  
  Serial.print(", \"lat\": ");
  if (gps.location.isValid()) {
    Serial.print(gps.location.lat(), 6);
  } else {
    Serial.print(0.0);
  }
  
  Serial.print(", \"lng\": ");
  if (gps.location.isValid()) {
    Serial.print(gps.location.lng(), 6);
  } else {
    Serial.print(0.0);
  }
  
  Serial.print(", \"spd\": ");
  if (gps.location.isValid()) {
    float speed = gps.speed.kmph();
    if(speed < 2.5) speed = 0;
    Serial.print(speed);
  } else {
    Serial.print(0.0);
  }
  
  Serial.print(", \"sat\": ");
  Serial.print(gps.satellites.value());

  Serial.print(", \"hdop\": ");
  if (gps.hdop.isValid()) {
    Serial.print(gps.hdop.hdop());
  } else {
    Serial.print(99.9);
  }

  Serial.println(", \"crash\": false}");

  delay(100);
}