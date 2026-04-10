#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

#define trig1 D2
#define echo1 D1
#define trig2 D3
#define echo2 D4

const char* ssid = "A52s";
const char* password = "piyush@13";

ESP8266WebServer server(80);

long duration1;
long duration2;
float distance1;
float distance2;

String status1;
String status2;

void readUltrasonic() {

  // Dry Waste Sensor
  digitalWrite(trig1, LOW);
  delayMicroseconds(2);
  digitalWrite(trig1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig1, LOW);

  duration1 = pulseIn(echo1, HIGH);
  distance1 = duration1 * 0.034 / 2;

  // Limit maximum distance
  if(distance1 > 16.5)
    distance1 = 16.5;


  // Wet Waste Sensor
  digitalWrite(trig2, LOW);
  delayMicroseconds(2);
  digitalWrite(trig2, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig2, LOW);

  duration2 = pulseIn(echo2, HIGH);
  distance2 = duration2 * 0.034 / 2;

  // Limit maximum distance
  if(distance2 > 16.5)
    distance2 = 16.5;


  // Overflow Logic
  if(distance1 < 9)
    status1 = "OVERFLOW";
  else
    status1 = "SAFE";

  if(distance2 < 9)
    status2 = "OVERFLOW";
  else
    status2 = "SAFE";
}

void handleRoot() {

  readUltrasonic();

  String page = "<!DOCTYPE html>";
  page += "<html>";
  page += "<head>";
  page += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  page += "<meta http-equiv='refresh' content='2'>";
  page += "<title>Smart Dustbin Dashboard</title>";

  page += "<style>";
  page += "body{font-family:Arial;background:#0f172a;color:white;text-align:center;margin-top:50px;}";
  page += "h1{font-size:40px;margin-bottom:40px;}";
  page += ".container{display:flex;justify-content:center;gap:40px;}";
  page += ".card{background:#1e293b;padding:30px;border-radius:15px;width:260px;box-shadow:0 0 20px rgba(0,0,0,0.5);}";
  page += ".sensor{font-size:26px;margin-bottom:10px;color:#38bdf8;}";
  page += ".distance{font-size:40px;font-weight:bold;color:#22c55e;}";
  page += ".status{font-size:28px;margin-top:10px;font-weight:bold;}";
  page += ".safe{color:#22c55e;}";
  page += ".overflow{color:#ef4444;}";
  page += "</style>";

  page += "</head>";
  page += "<body>";

  page += "<h1>Smart Dustbin Monitoring</h1>";

  page += "<div class='container'>";

  // Dry Waste
  page += "<div class='card'>";
  page += "<div class='sensor'>Dry Waste</div>";
  page += "<div class='distance'>";
  page += distance1;
  page += " cm</div>";

  if(status1 == "SAFE")
    page += "<div class='status safe'>SAFE</div>";
  else
    page += "<div class='status overflow'>OVERFLOW</div>";

  page += "</div>";

  // Wet Waste
  page += "<div class='card'>";
  page += "<div class='sensor'>Wet Waste</div>";
  page += "<div class='distance'>";
  page += distance2;
  page += " cm</div>";

  if(status2 == "SAFE")
    page += "<div class='status safe'>SAFE</div>";
  else
    page += "<div class='status overflow'>OVERFLOW</div>";

  page += "</div>";

  page += "</div>";

  page += "</body>";
  page += "</html>";

  server.send(200, "text/html", page);
}

void setup() {

  Serial.begin(115200);

  pinMode(trig1, OUTPUT);
  pinMode(echo1, INPUT);
  pinMode(trig2, OUTPUT);
  pinMode(echo2, INPUT);

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.begin();
}

void loop() {
  server.handleClient();
}
