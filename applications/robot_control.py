"""
Extended neural decoder for wheelchair/robot control
"""

import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

from .models import CNNLSTMDecoder


class ControlMode(Enum):
    VELOCITY = "velocity"  # Direct velocity control
    DISCRETE = "discrete"  # Up/Down/Left/Right/Stop
    GOAL = "goal"  # Point-to-point navigation


@dataclass
class RobotCommand:
    linear_velocity: float  # m/s forward/backward
    angular_velocity: float  # rad/s rotation
    emergency_stop: bool = False
    confidence: float = 1.0


class RobotNeuralDecoder:
    """
    Neural decoder specifically for wheelchair/robot control
    with safety features and multiple control modes
    """
    
    def __init__(
        self,
        model_path: str,
        num_channels: int = 32,
        window_size: int = 64,
        control_mode: ControlMode = ControlMode.VELOCITY,
        max_linear_vel: float = 1.0,  # m/s
        max_angular_vel: float = 1.0,  # rad/s
    ):
        self.control_mode = control_mode
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        
        # Load trained model
        self.model = CNNLSTMDecoder(num_channels, window_size)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        # Safety features
        self.emergency_detector = EmergencyStopDetector()
        self.confidence_estimator = ConfidenceEstimator()
        self.command_smoother = CommandSmoother()
        
    def decode_command(
        self, 
        neural_signal: np.ndarray,
        obstacle_map: Optional[np.ndarray] = None
    ) -> RobotCommand:
        """
        Decode neural signals into safe robot commands
        
        Args:
            neural_signal: [num_channels, window_size] array
            obstacle_map: Optional obstacle/safety information
            
        Returns:
            RobotCommand with velocities and safety flags
        """
        # Check for emergency stop pattern
        if self.emergency_detector.detect(neural_signal):
            return RobotCommand(0, 0, emergency_stop=True, confidence=1.0)
        
        # Get raw velocity from neural decoder
        with torch.no_grad():
            x = torch.from_numpy(neural_signal).unsqueeze(0).float()
            raw_velocity = self.model(x).squeeze().numpy()
        
        # Estimate confidence in the command
        confidence = self.confidence_estimator.estimate(neural_signal, raw_velocity)
        
        # Convert to robot commands based on mode
        if self.control_mode == ControlMode.VELOCITY:
            cmd = self._velocity_mode(raw_velocity, confidence)
        elif self.control_mode == ControlMode.DISCRETE:
            cmd = self._discrete_mode(raw_velocity, confidence)
        else:  # GOAL mode
            cmd = self._goal_mode(raw_velocity, obstacle_map)
        
        # Apply safety limits and smoothing
        cmd = self._apply_safety_limits(cmd, obstacle_map)
        cmd = self.command_smoother.smooth(cmd)
        
        return cmd
    
    def _velocity_mode(self, raw_vel: np.ndarray, confidence: float) -> RobotCommand:
        """Direct velocity control with confidence scaling"""
        # Map 2D cursor velocity to robot linear/angular
        linear_vel = raw_vel[1] * self.max_linear_vel * confidence
        angular_vel = raw_vel[0] * self.max_angular_vel * confidence
        
        return RobotCommand(linear_vel, angular_vel, confidence=confidence)
    
    def _discrete_mode(self, raw_vel: np.ndarray, confidence: float) -> RobotCommand:
        """Discrete command mode for simpler control"""
        # Threshold to get discrete commands
        if np.abs(raw_vel).max() < 0.2:  # Dead zone
            return RobotCommand(0, 0, confidence=confidence)
        
        if np.abs(raw_vel[1]) > np.abs(raw_vel[0]):
            # Forward/Backward
            linear = self.max_linear_vel * 0.5 * np.sign(raw_vel[1])
            return RobotCommand(linear, 0, confidence=confidence)
        else:
            # Left/Right rotation
            angular = self.max_angular_vel * 0.5 * np.sign(raw_vel[0])
            return RobotCommand(0, angular, confidence=confidence)
    
    def _apply_safety_limits(
        self, 
        cmd: RobotCommand, 
        obstacle_map: Optional[np.ndarray]
    ) -> RobotCommand:
        """Apply safety constraints based on environment"""
        if obstacle_map is not None:
            # Reduce speed near obstacles
            min_distance = self._get_min_obstacle_distance(obstacle_map)
            if min_distance < 0.5:  # meters
                scale = min_distance / 0.5
                cmd.linear_velocity *= scale
                cmd.angular_velocity *= scale
        
        return cmd


class EmergencyStopDetector:
    """Detect emergency stop patterns in neural signals"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.baseline = None
        
    def detect(self, signal: np.ndarray) -> bool:
        """
        Detect sudden spike patterns indicating emergency stop
        Real BCIs often use specific thought patterns for this
        """
        # High frequency power as emergency indicator
        fft = np.fft.fft(signal, axis=1)
        high_freq_power = np.abs(fft[:, 30:]).mean()
        
        if self.baseline is None:
            self.baseline = high_freq_power
            return False
        
        # Sudden increase in high frequency = stop
        if high_freq_power > self.baseline * self.threshold:
            return True
            
        # Update baseline with exponential moving average
        self.baseline = 0.95 * self.baseline + 0.05 * high_freq_power
        return False


class ConfidenceEstimator:
    """Estimate confidence in decoded commands"""
    
    def estimate(self, signal: np.ndarray, velocity: np.ndarray) -> float:
        """
        Estimate confidence based on signal quality and consistency
        """
        # Signal-to-noise ratio
        signal_power = np.var(signal)
        noise_estimate = np.var(np.diff(signal, axis=1))
        snr = signal_power / (noise_estimate + 1e-6)
        
        # Velocity magnitude (very high velocities are suspicious)
        vel_magnitude = np.linalg.norm(velocity)
        vel_confidence = 1.0 / (1.0 + vel_magnitude ** 2)
        
        # Combined confidence
        confidence = np.tanh(snr * 0.1) * vel_confidence
        return float(np.clip(confidence, 0.1, 1.0))


class CommandSmoother:
    """Smooth commands for safer robot control"""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.prev_cmd = None
        
    def smooth(self, cmd: RobotCommand) -> RobotCommand:
        """Exponential smoothing to prevent jerky movements"""
        if self.prev_cmd is None:
            self.prev_cmd = cmd
            return cmd
        
        # Smooth velocities
        cmd.linear_velocity = (
            self.alpha * cmd.linear_velocity + 
            (1 - self.alpha) * self.prev_cmd.linear_velocity
        )
        cmd.angular_velocity = (
            self.alpha * cmd.angular_velocity + 
            (1 - self.alpha) * self.prev_cmd.angular_velocity
        )
        
        self.prev_cmd = cmd
        return cmd


# Example integration with ROS for real robot control
class ROSRobotInterface:
    """Interface to send commands to ROS-based robots"""
    
    def __init__(self):
        import rospy
        from geometry_msgs.msg import Twist
        
        rospy.init_node('neural_decoder_node')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()
        
    def send_command(self, cmd: RobotCommand):
        """Send command to robot via ROS"""
        if cmd.emergency_stop:
            self.twist.linear.x = 0
            self.twist.angular.z = 0
        else:
            self.twist.linear.x = cmd.linear_velocity
            self.twist.angular.z = cmd.angular_velocity
        
        self.cmd_pub.publish(self.twist)


# Training data collection for real BCI
class BCIDataCollector:
    """Collect labeled training data from users"""
    
    def __init__(self, neural_device, paradigm="motor_imagery"):
        self.device = neural_device
        self.paradigm = paradigm
        
    def collect_training_data(self, num_trials: int = 100):
        """
        Collect data using different paradigms:
        - Motor Imagery: Imagine moving left/right/forward
        - P300: Look at flashing arrows
        - SSVEP: Look at flickering targets
        """
        data = []
        labels = []
        
        for trial in range(num_trials):
            # Show target direction
            target = self._show_random_target()
            
            # Record neural signals while user imagines/looks
            signal = self.device.record(duration=2.0)
            
            data.append(signal)
            labels.append(target)
            
        return np.array(data), np.array(labels)
