#include <AccelStepper.h>
#include <SerialCommand.h>
#include <TimerOne.h>

SerialCommand SCmd;

bool b_move_complete = true;
int lockx = 0;

AccelStepper newStepper(int stepPin, int dirPin, int enablePin, int maxSpeed, int Acceleration) {
  AccelStepper stepper = AccelStepper(stepper.DRIVER, stepPin, dirPin);
  stepper.setEnablePin(enablePin);
  stepper.setPinsInverted(false, false, true);
  stepper.setMaxSpeed(maxSpeed);
  stepper.setAcceleration(Acceleration);
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH); // Ensure motor is disabled initially
  return stepper;
}

const int MotorCount = 1;
AccelStepper steppers[MotorCount];
int enablePins[MotorCount] = {8}; // Store enable pins for each motor

long stepperPos[MotorCount] = {0};
long stepsPerFullTurn[MotorCount] = {400};

void setup() {
  Serial.begin(115200);
  steppers[0] = newStepper(2, 5, 8, 3000, 4000);

  SCmd.addCommand("M", move_stepper);
  SCmd.addCommand("ready", check_move_complete);
  SCmd.addCommand("position", check_position);
  SCmd.addCommand("done", InteralMoveCompCheck);
  SCmd.addCommand("stop", stop_all);
  
  Timer1.initialize(500);
  Timer1.attachInterrupt(runSteppers);
}

void runSteppers(void) {
  for (int i = 0; i < MotorCount; i++) {
    steppers[i].run();
  }
}

void loop() {
  SCmd.readSerial();
}


void stop_spec(int value) {
  steppers[value].move(0);
  digitalWrite(enablePins[value], HIGH); // Disable motor after stopping
}

void stop_all() {
  Serial.println("stopping");
  for (int i = 0; i < MotorCount; i++) {
    stop_spec(i);
  }
}

void InteralMoveCompCheck(){
  for (int i = 0; i < MotorCount; i++) {
    digitalWrite(enablePins[i], HIGH); // Disable motor after move is complete
  }
}

void check_move_complete() {
  if (b_move_complete) {
    Serial.println("Ready for next command");
    return;
  }
  bool b_all_done = true;
  for (int i = 0; i < MotorCount; i++) {
    if (steppers[i].distanceToGo() > 0 || steppers[i].distanceToGo() < 0 ) {
      b_all_done = false;
    }
  }
  if (b_all_done) {
    b_move_complete = true;
  }
  else{
    Serial.println("Busy");
    }
}

void move_stepper() {
  char *arg;
  int step_idx;
  double angle;
  double steps;

  arg = SCmd.next();
  if (arg == NULL) {
    Serial.println("Not recognized: Stepper Number");
    return;
  }
  step_idx = atoi(arg);
  if (step_idx < 0 || step_idx >= MotorCount) {
    Serial.println("Not recognized: Invalid Stepper Index, please restart if unstable");
    Serial.print("Unrecognized index is: "); Serial.println(step_idx);
    return;
  }
  arg = SCmd.next();
  if (arg == NULL) {
    Serial.println("Not recognized: No height parameter given");
    return;
  }
  angle = atof(arg);
  steps = angle / 360 * 15360;
  
  digitalWrite(enablePins[step_idx], LOW); // Enable motor before movement
  
  b_move_complete = false;
  steppers[step_idx].moveTo(steps);
} 

void check_position() {
  for (int i = 0; i < MotorCount; i++) {
    stepperPos[i] = steppers[i].currentPosition();
  }
  Serial.println(String(stepperPos[0]));
}
