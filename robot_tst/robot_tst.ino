#include <Servo.h>

Servo servos[5];
int servoPins[] = {3, 5, 6, 9, 10};

bool readySent = false;
bool sequenceRunning = true;  // keep servo movement only when sequence is running

void setup() {
  Serial.begin(115200);

  // Attach servos and move to robot zero (90°)
  for (int i = 0; i < 5; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(90);
  }

  Serial.println("Arduino ready, moving all servos to 90° (robot zero).");
}

void loop() {
  // Send READY once at startup
  if (!readySent) {
    Serial.println("READY");
    readySent = true;
  }

  // Read incoming Serial line
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    // If ROS 2 sends "END", stop updating but keep last position
    if (line.equalsIgnoreCase("END")) {
      sequenceRunning = false;
      Serial.println("Sequence complete. Robot holds final position.");
      return;
    }

    if (!sequenceRunning) return;  // do not move if sequence finished

    // Parse 5 angles from CSV
    float angles[5];
    int startIndex = 0;
    for (int i = 0; i < 5; i++) {
      int commaIndex = line.indexOf(',', startIndex);
      String angleStr = (i < 4) ? line.substring(startIndex, commaIndex) : line.substring(startIndex);
      angles[i] = angleStr.toFloat();
      startIndex = commaIndex + 1;
    }

    // Map joint angles to servo angles
    for (int i = 0; i < 5; i++) {
      int servoAngle = 0;
      if (i == 0 || i == 4) {           // Joints 1 & 5: 0→180
        servoAngle = constrain(int(angles[i]), 0, 180);
      } else {     
        if (i == 3)
        {
          servoAngle = constrain(int(-angles[i]+90), 0, 180);
        }   
        else{
         servoAngle = constrain(int(angles[i] + 90), 0, 180);  // Joints 2,3,4: -90→90 → 0→180
        }                 
      }
      servos[i].write(servoAngle);
    }

    // Feedback
    Serial.print("Servo angles applied: ");
    for (int i = 0; i < 5; i++) {
      Serial.print(servos[i].read());
      if (i < 4) Serial.print(", ");
    }
    Serial.println();
  }
}
