"""
Example: Brain-controlled wheelchair using Arduino and motor controllers
"""

import serial
import numpy as np
import time
from src.neural_decoder.robot_control import RobotNeuralDecoder, ControlMode


class WheelchairController:
    """
    Interface between neural decoder and wheelchair hardware
    """
    
    def __init__(self, serial_port="/dev/ttyUSB0", baud_rate=115200):
        # Neural decoder
        self.decoder = RobotNeuralDecoder(
            model_path="artifacts/model.pt",
            control_mode=ControlMode.DISCRETE,
            max_linear_vel=0.5,  # 0.5 m/s max for safety
            max_angular_vel=0.3   # Gentle turns
        )
        
        # Arduino connection for motor control
        self.arduino = serial.Serial(serial_port, baud_rate)
        time.sleep(2)  # Arduino reset time
        
        # Safety sensors
        self.ultrasonic_sensors = {
            'front': 0,
            'left': 1,
            'right': 2
        }
        
    def run(self, neural_source):
        """
        Main control loop
        """
        print("Wheelchair Neural Control Active")
        print("Think 'forward' to move, 'stop' to brake")
        
        try:
            while True:
                # Get neural data (e.g., from OpenBCI)
                neural_signal = neural_source.get_latest_window()
                
                # Read obstacle sensors
                distances = self.read_ultrasonic_sensors()
                obstacle_map = self.create_obstacle_map(distances)
                
                # Decode command
                cmd = self.decoder.decode_command(neural_signal, obstacle_map)
                
                # Send to wheelchair
                self.send_to_motors(cmd)
                
                # Display status
                self.display_status(cmd, distances)
                
                time.sleep(0.05)  # 20 Hz control loop
                
        except KeyboardInterrupt:
            self.emergency_stop()
            print("\nControl stopped")
    
    def send_to_motors(self, cmd):
        """
        Send motor commands via Arduino
        Protocol: "L:<left_speed>,R:<right_speed>\n"
        """
        if cmd.emergency_stop:
            self.arduino.write(b"STOP\n")
            return
        
        # Convert linear/angular to differential drive
        left_speed = cmd.linear_velocity - cmd.angular_velocity
        right_speed = cmd.linear_velocity + cmd.angular_velocity
        
        # Scale to motor values (0-255)
        left_motor = int(np.clip(left_speed * 255, -255, 255))
        right_motor = int(np.clip(right_speed * 255, -255, 255))
        
        # Send command
        command = f"L:{left_motor},R:{right_motor}\n"
        self.arduino.write(command.encode())
    
    def read_ultrasonic_sensors(self):
        """Read distance sensors via Arduino"""
        self.arduino.write(b"READ_SENSORS\n")
        response = self.arduino.readline().decode().strip()
        
        # Parse "F:100,L:80,R:90" format
        distances = {}
        for reading in response.split(','):
            sensor, value = reading.split(':')
            distances[sensor] = float(value) / 100  # cm to meters
            
        return distances
    
    def create_obstacle_map(self, distances):
        """Simple obstacle representation"""
        # In practice, use occupancy grid or point cloud
        return np.array([
            distances.get('F', 999),
            distances.get('L', 999),
            distances.get('R', 999)
        ])
    
    def emergency_stop(self):
        """Hardware emergency stop"""
        self.arduino.write(b"STOP\n")
        # Could also trigger mechanical brakes
    
    def display_status(self, cmd, distances):
        """Show status on LCD or terminal"""
        print(f"\rLinear: {cmd.linear_velocity:+.2f} m/s | "
              f"Angular: {cmd.angular_velocity:+.2f} rad/s | "
              f"Confidence: {cmd.confidence:.0%} | "
              f"Front: {distances.get('F', 0):.1f}m", end='')


# Arduino code (wheelchair_control.ino)
ARDUINO_CODE = """
// Motor pins
#define LEFT_MOTOR_PIN 9
#define RIGHT_MOTOR_PIN 10
#define LEFT_DIR_PIN 7
#define RIGHT_DIR_PIN 8

// Ultrasonic pins
#define TRIG_FRONT 2
#define ECHO_FRONT 3
#define TRIG_LEFT 4
#define ECHO_LEFT 5
#define TRIG_RIGHT 6
#define ECHO_RIGHT 11

void setup() {
  Serial.begin(115200);
  
  // Motor setup
  pinMode(LEFT_MOTOR_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_PIN, OUTPUT);
  pinMode(LEFT_DIR_PIN, OUTPUT);
  pinMode(RIGHT_DIR_PIN, OUTPUT);
  
  // Sensor setup
  pinMode(TRIG_FRONT, OUTPUT);
  pinMode(ECHO_FRONT, INPUT);
  // ... etc
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\\n');
    
    if (command == "STOP") {
      emergencyStop();
    }
    else if (command.startsWith("L:")) {
      parseAndSetMotors(command);
    }
    else if (command == "READ_SENSORS") {
      sendSensorReadings();
    }
  }
}

void parseAndSetMotors(String command) {
  // Parse "L:100,R:-100" format
  int commaIndex = command.indexOf(',');
  int leftSpeed = command.substring(2, commaIndex).toInt();
  int rightSpeed = command.substring(commaIndex + 3).toInt();
  
  // Set motor directions
  digitalWrite(LEFT_DIR_PIN, leftSpeed >= 0 ? HIGH : LOW);
  digitalWrite(RIGHT_DIR_PIN, rightSpeed >= 0 ? HIGH : LOW);
  
  // Set motor speeds
  analogWrite(LEFT_MOTOR_PIN, abs(leftSpeed));
  analogWrite(RIGHT_MOTOR_PIN, abs(rightSpeed));
}

void emergencyStop() {
  analogWrite(LEFT_MOTOR_PIN, 0);
  analogWrite(RIGHT_MOTOR_PIN, 0);
}

float readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH);
  float distance = duration * 0.034 / 2;  // cm
  
  return distance;
}

void sendSensorReadings() {
  float front = readUltrasonic(TRIG_FRONT, ECHO_FRONT);
  float left = readUltrasonic(TRIG_LEFT, ECHO_LEFT);
  float right = readUltrasonic(TRIG_RIGHT, ECHO_RIGHT);
  
  Serial.print("F:");
  Serial.print(front);
  Serial.print(",L:");
  Serial.print(left);
  Serial.print(",R:");
  Serial.println(right);
}
"""


# Example usage with OpenBCI
if __name__ == "__main__":
    # Mock neural source for demo
    class MockNeuralSource:
        def get_latest_window(self):
            # In reality: stream from OpenBCI/EEG device
            return np.random.randn(32, 64).astype(np.float32)
    
    # Run wheelchair
    neural_source = MockNeuralSource()
    controller = WheelchairController(serial_port="COM3")  # Windows
    controller.run(neural_source)
