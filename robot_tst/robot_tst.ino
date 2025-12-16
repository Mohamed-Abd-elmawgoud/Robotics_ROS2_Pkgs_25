#include <Servo.h>

Servo servos[6];
int servoPins[] = {3, 5, 6, 9, 10, 11};

bool sequenceRunning = false;
int totalTrajectories = 0;
int currentTrajectory = 0;
bool initialized = false;

void go_home();
void execute_grip();
void execute_twist();

void setup() {
  Serial.begin(115200);

  // Attach servos and move to robot zero (90°)
  for (int i = 0; i < 6; i++) {
    servos[i].attach(servoPins[i]);
  }

  go_home();

  servos[4].write(0);
  servos[5].write(160);

  Serial.println("Arduino ready, moving all servos to 90° (robot zero).");
  delay(100);
  
  // Send initial READY to receive INIT message
  Serial.println("READY");
}

void loop() {
  // Read incoming Serial line
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    // Check for INIT message with number of trajectories
    if (line.startsWith("INIT ")) {
      totalTrajectories = line.substring(5).toInt();
      currentTrajectory = 0;
      initialized = true;
      
      Serial.print("Initialized: Will execute ");
      Serial.print(totalTrajectories);
      Serial.println(" trajectories");
      
      // Ready to receive first trajectory
      delay(100);
      Serial.println("READY");
      sequenceRunning = true;
      return;
    }

    // Check for END message
    if (line.equalsIgnoreCase("END")) {
      sequenceRunning = false;
      currentTrajectory++;
      
      Serial.print("Trajectory ");
      Serial.print(currentTrajectory);
      Serial.print("/");
      Serial.print(totalTrajectories);
      Serial.println(" complete.");
      if(currentTrajectory ==2){
        execute_grip();
      }
      // Check if all trajectories are done
      if (currentTrajectory >= totalTrajectories) {
        execute_twist();
        execute_twist();
        delay(500);
        servos[4].write(90);
        for(int i=23; i<=50;i++){
          servos[2].write(i);
          delay(600);
        }
        Serial.println("=== ALL TRAJECTORIES COMPLETE ===");
        Serial.println("Robot holding final position.");
        Serial.println("Send READY when you want to start a new cycle.");
        initialized = false;  // Need new INIT for next cycle
      } else {
        // More trajectories to go, send READY for next one
        delay(500);  // Brief pause at final position
        Serial.println("READY");
        sequenceRunning = true;
      }
      return;
    }

    // Only process angle data if sequence is running
    if (!sequenceRunning || !initialized) return;

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
    for (int i = 0; i < 4; i++) {
      int servoAngle = 0;
      if (i == 0 || i == 4) {           // Joints 1 & 5: 0→180
        servoAngle = constrain(int(angles[i]), 0, 180);
      } else {     
        if (i == 3) {
          servoAngle = constrain(int(-angles[i] + 90), 0, 180);
        } else {
          servoAngle = constrain(int(angles[i] + 90), 0, 180);  // Joints 2,3: -90→90 → 0→180
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

void execute_grip(){
  delay(1000);
  servos[4].write(0);
  delay(500);
  servos[5].write(125);
}

void execute_twist(){
  delay(500);
  servos[4].write(180);
  delay(2000);
  servos[5].write(160);
  delay(2000);
  // servos[2].write(30);
  // delay(2000);
  servos[4].write(0);
  // delay(2000);
  // servos[2].write(24);
  delay(2000);
  servos[5].write(125);
  delay(1000);
  servos[4].write(180);
  delay(2000);
  servos[5].write(160);
  delay(2000);
  servos[4].write(0);
  delay(2000);
  servos[5].write(125);
  delay(2000);
  servos[4].write(180);
  delay(2000);
  servos[5].write(160);
}

void go_home()
{
  for(int i=0; i<=3; i++){
    servos[i].write(90);
  }

  
}