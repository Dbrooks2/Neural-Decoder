"""
Face-Controlled Mouse with Adaptive Learning
Control your actual mouse cursor with facial expressions!
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import threading

# For mouse control
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move to corner to stop
    pyautogui.PAUSE = 0.01  # Reduce lag
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    print("Install pyautogui for mouse control: pip install pyautogui")

import mediapipe as mp


@dataclass 
class FaceMouseConfig:
    """Configuration for face-controlled mouse"""
    # Control mappings
    eyebrow_raise_action: str = "move_up"
    mouth_open_action: str = "left_click"
    wink_left_action: str = "right_click"
    wink_right_action: str = "double_click"
    tongue_out_action: str = "middle_click"
    
    # Sensitivity settings (learned per user)
    vertical_sensitivity: float = 2.0
    horizontal_sensitivity: float = 2.0
    gaze_sensitivity: float = 1.5
    
    # Dead zones
    movement_threshold: float = 0.02
    click_threshold: float = 0.5
    
    # Smoothing
    movement_smoothing: float = 0.7
    
    # Click timing
    click_duration: float = 0.1
    double_click_interval: float = 0.3
    
    # Comfort settings
    auto_recenter: bool = True
    recenter_interval: float = 5.0
    
    # Accessibility
    dwell_click_enabled: bool = False
    dwell_time: float = 2.0
    tremor_reduction: float = 0.0


class FaceControlledMouse:
    """
    Complete face-controlled mouse with adaptive learning
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.config_dir = Path("face_mouse_configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # Load user config
        self.config = self.load_config()
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State tracking
        self.neutral_position = None
        self.is_calibrated = False
        self.last_recenter = time.time()
        
        # Movement smoothing
        self.smooth_x = 0
        self.smooth_y = 0
        
        # Click state
        self.last_click_time = 0
        self.is_clicking = False
        self.dwell_start_time = None
        self.dwell_position = None
        
        # Performance tracking
        self.movement_history = deque(maxlen=1000)
        self.click_history = deque(maxlen=100)
        self.error_corrections = deque(maxlen=50)
        
        # Learning parameters
        self.learning_enabled = True
        self.adaptation_rate = 0.02
        
        # Get screen size
        if HAS_PYAUTOGUI:
            self.screen_width, self.screen_height = pyautogui.size()
        else:
            self.screen_width, self.screen_height = 1920, 1080
    
    def load_config(self) -> FaceMouseConfig:
        """Load user-specific configuration"""
        config_path = self.config_dir / f"{self.user_id}_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                return FaceMouseConfig(**data)
        return FaceMouseConfig()
    
    def save_config(self):
        """Save user configuration"""
        config_path = self.config_dir / f"{self.user_id}_config.json"
        
        config_dict = {
            'eyebrow_raise_action': self.config.eyebrow_raise_action,
            'mouth_open_action': self.config.mouth_open_action,
            'wink_left_action': self.config.wink_left_action,
            'wink_right_action': self.config.wink_right_action,
            'tongue_out_action': self.config.tongue_out_action,
            'vertical_sensitivity': self.config.vertical_sensitivity,
            'horizontal_sensitivity': self.config.horizontal_sensitivity,
            'gaze_sensitivity': self.config.gaze_sensitivity,
            'movement_threshold': self.config.movement_threshold,
            'click_threshold': self.config.click_threshold,
            'movement_smoothing': self.config.movement_smoothing,
            'click_duration': self.config.click_duration,
            'double_click_interval': self.config.double_click_interval,
            'auto_recenter': self.config.auto_recenter,
            'recenter_interval': self.config.recenter_interval,
            'dwell_click_enabled': self.config.dwell_click_enabled,
            'dwell_time': self.config.dwell_time,
            'tremor_reduction': self.config.tremor_reduction,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def calibrate(self, frame: np.ndarray, landmarks):
        """Calibrate neutral face position"""
        print("\nCalibrating... Keep a neutral expression")
        
        # Extract key features
        features = self.extract_features(landmarks)
        self.neutral_position = features.copy()
        self.is_calibrated = True
        
        print("Calibration complete!")
        print("\nControls:")
        print("- Look around: Move mouse")
        print("- Raise eyebrows: Move up faster")
        print("- Open mouth: Left click")
        print("- Wink left eye: Right click")
        print("- Wink right eye: Double click")
        print("- Press 'r' to recenter")
        print("- Press 'c' to recalibrate")
        print("- Press 'q' to quit")
    
    def extract_features(self, landmarks) -> dict:
        """Extract facial features for control"""
        features = {}
        
        # Head position (use nose tip)
        features['nose_x'] = landmarks[1].x
        features['nose_y'] = landmarks[1].y
        
        # Eye positions (for gaze)
        features['left_eye_x'] = landmarks[33].x
        features['left_eye_y'] = landmarks[33].y
        features['right_eye_x'] = landmarks[263].x
        features['right_eye_y'] = landmarks[263].y
        
        # Eye openness (for winks/blinks)
        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y
        right_eye_top = landmarks[386].y
        right_eye_bottom = landmarks[374].y
        
        features['left_eye_open'] = abs(left_eye_top - left_eye_bottom)
        features['right_eye_open'] = abs(right_eye_top - right_eye_bottom)
        
        # Eyebrow height
        left_eyebrow = landmarks[70].y
        right_eyebrow = landmarks[300].y
        features['eyebrow_height'] = (left_eyebrow + right_eyebrow) / 2
        
        # Mouth openness
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        features['mouth_open'] = abs(mouth_top - mouth_bottom)
        
        # Iris positions (if available)
        if len(landmarks) > 468:
            features['left_iris_x'] = landmarks[468].x
            features['left_iris_y'] = landmarks[468].y
            features['right_iris_x'] = landmarks[473].x
            features['right_iris_y'] = landmarks[473].y
        
        return features
    
    def calculate_mouse_movement(self, features: dict) -> tuple:
        """Calculate mouse movement from facial features"""
        if not self.is_calibrated or self.neutral_position is None:
            return 0, 0
        
        # Head-based movement (primary control)
        head_dx = features['nose_x'] - self.neutral_position['nose_x']
        head_dy = features['nose_y'] - self.neutral_position['nose_y']
        
        # Gaze-based fine control (if iris tracking available)
        gaze_dx = 0
        gaze_dy = 0
        if 'left_iris_x' in features:
            # Calculate gaze direction relative to eye center
            left_gaze_x = features['left_iris_x'] - features['left_eye_x']
            right_gaze_x = features['right_iris_x'] - features['right_eye_x']
            gaze_dx = (left_gaze_x + right_gaze_x) / 2
            
            left_gaze_y = features['left_iris_y'] - features['left_eye_y']
            right_gaze_y = features['right_iris_y'] - features['right_eye_y']
            gaze_dy = (left_gaze_y + right_gaze_y) / 2
        
        # Combine head and gaze movement
        total_dx = (head_dx * self.config.horizontal_sensitivity + 
                   gaze_dx * self.config.gaze_sensitivity)
        total_dy = (head_dy * self.config.vertical_sensitivity + 
                   gaze_dy * self.config.gaze_sensitivity)
        
        # Eyebrow raise boost (for faster upward movement)
        eyebrow_diff = self.neutral_position['eyebrow_height'] - features['eyebrow_height']
        if eyebrow_diff > 0.01:  # Raised eyebrows
            total_dy -= eyebrow_diff * self.config.vertical_sensitivity * 2
        
        # Apply dead zone
        if abs(total_dx) < self.config.movement_threshold:
            total_dx = 0
        if abs(total_dy) < self.config.movement_threshold:
            total_dy = 0
        
        # Apply tremor reduction if needed
        if self.config.tremor_reduction > 0:
            # Simple low-pass filter
            alpha = 1 - self.config.tremor_reduction
            total_dx = alpha * total_dx + (1 - alpha) * self.smooth_x
            total_dy = alpha * total_dy + (1 - alpha) * self.smooth_y
        
        # Convert to screen coordinates
        mouse_dx = total_dx * self.screen_width
        mouse_dy = total_dy * self.screen_height
        
        # Apply smoothing
        smooth = self.config.movement_smoothing
        self.smooth_x = smooth * self.smooth_x + (1 - smooth) * mouse_dx
        self.smooth_y = smooth * self.smooth_y + (1 - smooth) * mouse_dy
        
        # Track movement for learning
        self.movement_history.append({
            'dx': self.smooth_x,
            'dy': self.smooth_y,
            'raw_dx': head_dx,
            'raw_dy': head_dy,
            'time': time.time()
        })
        
        return self.smooth_x, self.smooth_y
    
    def detect_gestures(self, features: dict) -> list:
        """Detect facial gestures for clicks"""
        gestures = []
        
        # Mouth open -> click
        mouth_open_ratio = features['mouth_open'] / 0.05  # Normalize
        if mouth_open_ratio > self.config.click_threshold:
            gestures.append('mouth_open')
        
        # Eye winks (one eye closed, other open)
        left_closed = features['left_eye_open'] < 0.01
        right_closed = features['right_eye_open'] < 0.01
        
        if left_closed and not right_closed:
            gestures.append('wink_left')
        elif right_closed and not left_closed:
            gestures.append('wink_right')
        elif left_closed and right_closed:
            gestures.append('both_eyes_closed')
        
        # Eyebrow gestures
        eyebrow_raise = self.neutral_position['eyebrow_height'] - features['eyebrow_height']
        if eyebrow_raise > 0.02:
            gestures.append('eyebrow_raise')
        
        return gestures
    
    def execute_action(self, action: str):
        """Execute mouse action"""
        if not HAS_PYAUTOGUI:
            return
        
        current_time = time.time()
        
        if action == 'left_click':
            if current_time - self.last_click_time > 0.3:  # Debounce
                pyautogui.click()
                self.last_click_time = current_time
                self.click_history.append({
                    'type': 'left',
                    'time': current_time
                })
        
        elif action == 'right_click':
            if current_time - self.last_click_time > 0.3:
                pyautogui.rightClick()
                self.last_click_time = current_time
                self.click_history.append({
                    'type': 'right',
                    'time': current_time
                })
        
        elif action == 'double_click':
            if current_time - self.last_click_time > 0.5:
                pyautogui.doubleClick()
                self.last_click_time = current_time
                self.click_history.append({
                    'type': 'double',
                    'time': current_time
                })
        
        elif action == 'middle_click':
            if current_time - self.last_click_time > 0.3:
                pyautogui.middleClick()
                self.last_click_time = current_time
    
    def check_dwell_click(self, dx: float, dy: float):
        """Check for dwell clicking (hover to click)"""
        if not self.config.dwell_click_enabled or not HAS_PYAUTOGUI:
            return
        
        current_pos = pyautogui.position()
        movement = abs(dx) + abs(dy)
        
        if movement < 5:  # Mouse is still
            if self.dwell_position is None:
                self.dwell_position = current_pos
                self.dwell_start_time = time.time()
            else:
                # Check if still in same position
                dist = np.sqrt((current_pos[0] - self.dwell_position[0])**2 + 
                             (current_pos[1] - self.dwell_position[1])**2)
                
                if dist < 20:  # Still hovering
                    dwell_duration = time.time() - self.dwell_start_time
                    if dwell_duration > self.config.dwell_time:
                        pyautogui.click()
                        self.dwell_position = None
                        self.dwell_start_time = None
                else:
                    # Moved away, reset
                    self.dwell_position = current_pos
                    self.dwell_start_time = time.time()
        else:
            # Moving, reset dwell
            self.dwell_position = None
            self.dwell_start_time = None
    
    def learn_from_usage(self):
        """Adapt settings based on usage patterns"""
        if not self.learning_enabled or len(self.movement_history) < 100:
            return
        
        recent_movements = list(self.movement_history)[-100:]
        
        # 1. Learn sensitivity preferences
        # If user makes many small movements, increase sensitivity
        avg_movement = np.mean([abs(m['dx']) + abs(m['dy']) for m in recent_movements])
        
        if avg_movement < 5:  # Very small movements
            self.config.horizontal_sensitivity *= (1 + self.adaptation_rate)
            self.config.vertical_sensitivity *= (1 + self.adaptation_rate)
        elif avg_movement > 50:  # Large movements
            self.config.horizontal_sensitivity *= (1 - self.adaptation_rate)
            self.config.vertical_sensitivity *= (1 - self.adaptation_rate)
        
        # 2. Learn directional bias
        h_movements = [abs(m['dx']) for m in recent_movements]
        v_movements = [abs(m['dy']) for m in recent_movements]
        
        if np.mean(h_movements) > np.mean(v_movements) * 1.5:
            # User moves more horizontally
            self.config.horizontal_sensitivity *= (1 + self.adaptation_rate * 0.5)
        elif np.mean(v_movements) > np.mean(h_movements) * 1.5:
            # User moves more vertically
            self.config.vertical_sensitivity *= (1 + self.adaptation_rate * 0.5)
        
        # 3. Learn smoothing preference from movement patterns
        # Check for jittery movements
        if len(recent_movements) > 10:
            jitter = np.std([m['dx'] for m in recent_movements[-10:]])
            
            if jitter > 10:  # High jitter
                self.config.movement_smoothing = min(0.9, 
                    self.config.movement_smoothing + self.adaptation_rate)
            elif jitter < 2:  # Very smooth
                self.config.movement_smoothing = max(0.3, 
                    self.config.movement_smoothing - self.adaptation_rate)
        
        # 4. Learn click patterns
        if len(self.click_history) > 10:
            recent_clicks = list(self.click_history)[-10:]
            
            # Check for frequent double-clicks
            double_clicks = [c for c in recent_clicks if c['type'] == 'double']
            if len(double_clicks) > len(recent_clicks) * 0.3:
                # User double-clicks often, might need to adjust timing
                self.config.double_click_interval *= 0.95
    
    def draw_overlay(self, frame: np.ndarray, features: dict, gestures: list):
        """Draw helpful overlay on video"""
        # Status bar
        status = "CALIBRATED" if self.is_calibrated else "NOT CALIBRATED"
        color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Current gestures
        if gestures:
            gesture_text = "Gestures: " + ", ".join(gestures)
            cv2.putText(frame, gesture_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Sensitivity indicators
        cv2.putText(frame, f"H-Sens: {self.config.horizontal_sensitivity:.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"V-Sens: {self.config.vertical_sensitivity:.1f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Dwell click indicator
        if self.config.dwell_click_enabled and self.dwell_start_time:
            dwell_progress = (time.time() - self.dwell_start_time) / self.config.dwell_time
            dwell_progress = min(1.0, dwell_progress)
            
            # Draw progress bar
            bar_width = int(200 * dwell_progress)
            cv2.rectangle(frame, (220, 20), (220 + bar_width, 40), (0, 255, 0), -1)
            cv2.rectangle(frame, (220, 20), (420, 40), (255, 255, 255), 2)
            cv2.putText(frame, "Dwell Click", (220, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Learning status
        learn_color = (0, 255, 0) if self.learning_enabled else (128, 128, 128)
        cv2.putText(frame, f"Learning: {'ON' if self.learning_enabled else 'OFF'}", 
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, learn_color, 1)
    
    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(0)
        
        print("\n=== Face-Controlled Mouse ===")
        print(f"User: {self.user_id}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract features
                features = self.extract_features(landmarks)
                
                # Calibrate if needed
                if not self.is_calibrated:
                    self.calibrate(frame, landmarks)
                else:
                    # Calculate movement
                    dx, dy = self.calculate_mouse_movement(features)
                    
                    # Move mouse
                    if HAS_PYAUTOGUI and (abs(dx) > 1 or abs(dy) > 1):
                        current_x, current_y = pyautogui.position()
                        new_x = current_x + dx
                        new_y = current_y + dy
                        
                        # Keep on screen
                        new_x = max(0, min(self.screen_width - 1, new_x))
                        new_y = max(0, min(self.screen_height - 1, new_y))
                        
                        pyautogui.moveTo(new_x, new_y, duration=0)
                    
                    # Detect gestures
                    gestures = self.detect_gestures(features)
                    
                    # Execute actions based on gestures
                    if 'mouth_open' in gestures:
                        self.execute_action(self.config.mouth_open_action)
                    if 'wink_left' in gestures:
                        self.execute_action(self.config.wink_left_action)
                    if 'wink_right' in gestures:
                        self.execute_action(self.config.wink_right_action)
                    
                    # Check dwell click
                    self.check_dwell_click(dx, dy)
                    
                    # Auto-recenter
                    if (self.config.auto_recenter and 
                        time.time() - self.last_recenter > self.config.recenter_interval):
                        self.neutral_position = features.copy()
                        self.last_recenter = time.time()
                    
                    # Learn from usage
                    if len(self.movement_history) % 50 == 0:
                        self.learn_from_usage()
                
                # Draw overlay
                self.draw_overlay(frame, features, gestures if self.is_calibrated else [])
                
                # Draw face mesh
                for idx in [1, 33, 263, 13, 14, 70, 300]:  # Key points
                    x = int(landmarks[idx].x * frame.shape[1])
                    y = int(landmarks[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            cv2.imshow('Face Mouse Control', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.is_calibrated = False
            elif key == ord('r'):
                if self.is_calibrated:
                    self.neutral_position = features.copy()
                    print("Recentered!")
            elif key == ord('l'):
                self.learning_enabled = not self.learning_enabled
                print(f"Learning: {'ON' if self.learning_enabled else 'OFF'}")
            elif key == ord('d'):
                self.config.dwell_click_enabled = not self.config.dwell_click_enabled
                print(f"Dwell click: {'ON' if self.config.dwell_click_enabled else 'OFF'}")
            elif key == ord('s'):
                self.save_config()
                print("Configuration saved!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.save_config()
        
        # Print usage statistics
        print("\n=== Session Statistics ===")
        print(f"Total movements: {len(self.movement_history)}")
        print(f"Total clicks: {len(self.click_history)}")
        print(f"Final sensitivity - H: {self.config.horizontal_sensitivity:.2f}, "
              f"V: {self.config.vertical_sensitivity:.2f}")
        print(f"Learned smoothing: {self.config.movement_smoothing:.2f}")


def main():
    print("\n=== Face-Controlled Mouse Setup ===")
    
    if not HAS_PYAUTOGUI:
        print("\nWARNING: pyautogui not installed!")
        print("Install with: pip install pyautogui")
        print("The demo will run but won't control your actual mouse.\n")
    
    user_id = input("Enter your name (or press Enter for 'default'): ").strip()
    if not user_id:
        user_id = "default"
    
    controller = FaceControlledMouse(user_id=user_id)
    
    print("\nStarting face-controlled mouse...")
    print("Make sure your face is well-lit and clearly visible.")
    print("\nThe system will learn your preferences over time!")
    
    controller.run()


if __name__ == "__main__":
    main()
