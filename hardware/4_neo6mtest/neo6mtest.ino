#include <TinyGPS++.h>
#include <SoftwareSerial.h>

TinyGPSPlus gps;

// RX, TX
SoftwareSerial gpsSerial(D5, D6);

void setup()
{
  Serial.begin(115200);
  gpsSerial.begin(9600);

  Serial.println("GPS Test Started...");
}

void loop()
{
  while (gpsSerial.available())
  {
    gps.encode(gpsSerial.read());

    if (gps.location.isUpdated())
    {
      Serial.println("GPS Data:");

      Serial.print("Latitude: ");
      Serial.println(gps.location.lat(), 6);

      Serial.print("Longitude: ");
      Serial.println(gps.location.lng(), 6);

      Serial.print("Speed (km/h): ");
      Serial.println(gps.speed.kmph());

      Serial.println("----------------");
    }
  }
}
