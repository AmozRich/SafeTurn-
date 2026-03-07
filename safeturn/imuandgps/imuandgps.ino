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

  Serial.println("---- MPU ----");

  Serial.print("Accel: ");
  Serial.print(ax); Serial.print(" ");
  Serial.print(ay); Serial.print(" ");
  Serial.println(az);

  Serial.print("Gyro: ");
  Serial.print(gx); Serial.print(" ");
  Serial.print(gy); Serial.print(" ");
  Serial.println(gz);

  // Print GPS
  if (gps.location.isValid())
  {
    Serial.println("---- GPS ----");

    Serial.print("Lat: ");
    Serial.println(gps.location.lat(), 6);

    Serial.print("Lng: ");
    Serial.println(gps.location.lng(), 6);

    float speed = gps.speed.kmph();
    if(speed < 2.5) speed = 0;

    Serial.print("Speed: ");
    Serial.print(speed);
    Serial.println(" km/h");
  }
  else
  {
    Serial.println("Waiting for GPS...");
  }

  Serial.println("----------------");

  delay(500);
}