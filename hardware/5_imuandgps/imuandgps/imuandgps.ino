#include <Wire.h>
#include <MPU6050.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>

MPU6050 mpu;
TinyGPSPlus gps;

SoftwareSerial gpsSerial(D5, D6);

void setup()
{
  Serial.begin(115200);

  Wire.begin(D2, D1);

  gpsSerial.begin(9600);

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
  while (gpsSerial.available())
  {
    gps.encode(gpsSerial.read());
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
  
  Serial.println(", \"crash\": false}");

  delay(100);
}