#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin(D2, D1);

  Serial.println("Initializing MPU...");

  mpu.initialize();

  if (mpu.testConnection()) {
    Serial.println("MPU connected successfully!");
  } else {
    Serial.println("MPU connection failed!");
  }
}

void loop() {

  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  Serial.println("------ MPU Data ------");

  Serial.print("Accel X: "); Serial.println(ax);
  Serial.print("Accel Y: "); Serial.println(ay);
  Serial.print("Accel Z: "); Serial.println(az);

  Serial.print("Gyro X: "); Serial.println(gx);
  Serial.print("Gyro Y: "); Serial.println(gy);
  Serial.print("Gyro Z: "); Serial.println(gz);

  Serial.println("----------------------");

  delay(1000);
}
