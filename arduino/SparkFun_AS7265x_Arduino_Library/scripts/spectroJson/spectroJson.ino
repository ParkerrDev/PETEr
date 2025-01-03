#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
AS7265X sensor;

void setup()
{
  Serial.begin(115200);
  Serial.println("AS7265x Spectral Triad Example");

  unsigned long startMillis = millis();
  while (sensor.begin() == false)
{
  if (millis() - startMillis > 5000) // Timeout after 5 seconds
  {
    Serial.println("Sensor connection timed out. Please check wiring. Continuing without sensor...");
    break;
  }
}


  if (sensor.begin() == false)
  {
    Serial.println("Sensor does not appear to be connected. Please check wiring. Freezing...");
    while (1)
      ;
  }

  sensor.disableIndicator();
  
  Serial.println("A,B,C,D,E,F,G,H,R,I,S,J,T,U,V,W,K,L");
}

void loop()
{
  sensor.takeMeasurementsWithBulb(); //This is a hard wait while all 18 channels are measured

  String jsonData = "{";

  jsonData += "\"A\":" + String(sensor.getCalibratedA()) + ","; //410nm
  jsonData += "\"B\":" + String(sensor.getCalibratedB()) + ","; //435nm
  jsonData += "\"C\":" + String(sensor.getCalibratedC()) + ","; //460nm 
  jsonData += "\"D\":" + String(sensor.getCalibratedD()) + ","; //485nm
  jsonData += "\"E\":" + String(sensor.getCalibratedE()) + ","; //510nm
  jsonData += "\"F\":" + String(sensor.getCalibratedF()) + ","; //535nm

  jsonData += "\"G\":" + String(sensor.getCalibratedG()) + ","; //560nm
  jsonData += "\"H\":" + String(sensor.getCalibratedH()) + ","; //585nm
  jsonData += "\"R\":" + String(sensor.getCalibratedR()) + ","; //610nm
  jsonData += "\"I\":" + String(sensor.getCalibratedI()) + ","; //645nm
  jsonData += "\"S\":" + String(sensor.getCalibratedS()) + ","; //680nm
  jsonData += "\"J\":" + String(sensor.getCalibratedJ()) + ","; //705nm
  
  jsonData += "\"T\":" + String(sensor.getCalibratedT()) + ","; //730nm
  jsonData += "\"U\":" + String(sensor.getCalibratedU()) + ","; //760nm
  jsonData += "\"V\":" + String(sensor.getCalibratedV()) + ","; //810nm
  jsonData += "\"W\":" + String(sensor.getCalibratedW()) + ","; //860nm
  jsonData += "\"K\":" + String(sensor.getCalibratedK()) + ","; //900nm
  jsonData += "\"L\":" + String(sensor.getCalibratedL()); //940nm

  jsonData += "}";

  Serial.println(jsonData);
}

