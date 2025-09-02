"""
Advanced Webcam Brain-Computer Interface
Uses facial expressions and head movements as neural signals
"""

import cv2
import numpy as np
import time
import sys
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# You'll need: pip install opencv-python mediapipe
import mediapipe as mp

# Import our neural decoder
from src.neural_decoder.models import build_model
import torch


@dataclass
class FacialSignal:
    """Represents a facial expression signal"""
    eyebrow_raise: float = 0.0      # 0-1, forehead muscle
    eye_squint: float = 0.0         # 0-1, concentration
    mouth_open: float = 0.0         # 0-1, jaw drop
    head_tilt_x: float = 0.0        # -1 to 1, left/right
    head_tilt_y: float = 0.0        # -1 to 1, up/down
    gaze_x: float = 0.0             # -1 to 1, eye direction
    gaze_y: float = 0.0             # -1 to 1
    blink_left: bool = False
    blink_right: bool = False
    tongue_out: bool = False        # Fun control!


class AdvancedWebcamBCI:
    """
    Full-featured facial BCI with multiple control modes
    """
    
    def __init__(self, use_neural_decoder: bool = True):
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Signal processing
        self.signal_buffer = deque(maxlen=64)  # Match neural decoder window
        self.calibration_data = {}
        self.is_calibrated = False
        
        # Neural decoder integration
        self.use_neural_decoder = use_neural_decoder
        if use_neural_decoder:
            self.setup_neural_decoder()
        
        # Control modes
        self.control_modes = {
            'cursor': self.cursor_control_mode,
            'game': self.game_control_mode,
            'paint': self.paint_mode,
            'keyboard': self.keyboard_mode
        }
        self.current_mode = 'cursor'
        
        # Visual elements
        self.cursor_x = 320
        self.cursor_y = 240
        self.paint_points = []
        self.game_objects = []
        
    def setup_neural_decoder(self):
        """Load the trained neural decoder model"""
        try:
            self.decoder = build_model(32, 64)
            model_path = "artifacts/model.pt"
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location="cpu")
                self.decoder.load_state_dict(state["state_dict"])
                self.decoder.eval()
                print("✓ Loaded neural decoder model")
            else:
                print("! Using random neural decoder (train for better results)")
        except Exception as e:
            print(f"Neural decoder setup failed: {e}")
            self.use_neural_decoder = False
    
    def extract_facial_signals(self, landmarks) -> FacialSignal:
        """Extract comprehensive facial signals from landmarks"""
        signal = FacialSignal()
        
        # Helper to get landmark position
        def get_point(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        
        # 1. Eyebrow raise (forehead muscle activation)
        eyebrow_points = [70, 63, 105, 66, 107]  # Eyebrow landmarks
        eyebrow_avg_y = np.mean([landmarks[i].y for i in eyebrow_points])
        nose_bridge_y = landmarks[6].y
        signal.eyebrow_raise = max(0, (nose_bridge_y - eyebrow_avg_y - 0.1) * 5)
        
        # 2. Eye squint (concentration)
        # Vertical eye aperture
        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y
        right_eye_top = landmarks[386].y
        right_eye_bottom = landmarks[374].y
        
        left_aperture = abs(left_eye_top - left_eye_bottom)
        right_aperture = abs(right_eye_top - right_eye_bottom)
        avg_aperture = (left_aperture + right_aperture) / 2
        
        signal.eye_squint = max(0, 1 - (avg_aperture / 0.02))  # Normalized
        
        # 3. Blink detection (separate for each eye)
        signal.blink_left = left_aperture < 0.005
        signal.blink_right = right_aperture < 0.005
        
        # 4. Mouth open (jaw control)
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        mouth_aperture = abs(mouth_top - mouth_bottom)
        signal.mouth_open = min(1, mouth_aperture / 0.05)
        
        # 5. Head pose estimation
        # Using nose tip and face center
        nose_tip = get_point(1)
        face_center = np.mean([get_point(i) for i in [1, 6, 10, 151, 234]], axis=0)
        
        # Estimate rotation
        signal.head_tilt_x = (nose_tip[0] - 0.5) * 2  # Left/right
        signal.head_tilt_y = (nose_tip[1] - face_center[1]) * 10  # Up/down
        
        # 6. Gaze direction (using iris landmarks)
        if len(landmarks) > 468:  # Has iris landmarks
            left_iris = get_point(468)
            right_iris = get_point(473)
            
            # Eye centers
            left_eye_center = np.mean([get_point(i) for i in [33, 133, 157, 158, 159, 160]], axis=0)
            right_eye_center = np.mean([get_point(i) for i in [362, 263, 387, 388, 389, 390]], axis=0)
            
            # Gaze relative to eye center
            left_gaze = left_iris - left_eye_center
            right_gaze = right_iris - right_eye_center
            avg_gaze = (left_gaze + right_gaze) / 2
            
            signal.gaze_x = avg_gaze[0] * 20
            signal.gaze_y = avg_gaze[1] * 20
        
        # 7. Tongue detection (fun control!)
        # Check if mouth is open and look for tongue landmarks
        if signal.mouth_open > 0.5:
            # Simplified: use lip distance as proxy
            upper_lip = landmarks[12].y
            lower_lip = landmarks[15].y
            lip_distance = abs(upper_lip - lower_lip)
            signal.tongue_out = lip_distance > 0.03
        
        return signal
    
    def calibrate(self):
        """Calibrate neutral facial position"""
        print("\n=== CALIBRATION ===")
        print("Keep a neutral expression for 3 seconds...")
        
        calibration_signals = []
        start_time = time.time()
        
        while time.time() - start_time < 3:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                signal = self.extract_facial_signals(landmarks)
                calibration_signals.append(signal)
            
            # Show calibration progress
            progress = (time.time() - start_time) / 3
            cv2.putText(frame, f"Calibrating... {progress*100:.0f}%", 
                       (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam BCI', frame)
            cv2.waitKey(1)
        
        # Calculate baselines
        if calibration_signals:
            self.calibration_data = {
                'eyebrow': np.mean([s.eyebrow_raise for s in calibration_signals]),
                'squint': np.mean([s.eye_squint for s in calibration_signals]),
                'mouth': np.mean([s.mouth_open for s in calibration_signals]),
                'tilt_x': np.mean([s.head_tilt_x for s in calibration_signals]),
                'tilt_y': np.mean([s.head_tilt_y for s in calibration_signals]),
            }
            self.is_calibrated = True
            print("✓ Calibration complete!")
            print("\nFacial Controls:")
            print("- Raise eyebrows: Forward/Up")
            print("- Squint: Stop/Select")
            print("- Open mouth: Boost/Action")
            print("- Look left/right: Turn")
            print("- Tilt head: Fine control")
            print("- Wink left: Previous mode")
            print("- Wink right: Next mode")
            print("- Stick tongue out: Special action!")
        
    def facial_signals_to_neural(self, signal: FacialSignal) -> np.ndarray:
        """Convert facial signals to simulated neural signals"""
        # Create 32 channels from facial features
        channels = []
        
        # Channel groups for different features
        # 1-8: Eyebrow/forehead (motor cortex simulation)
        for i in range(8):
            base = signal.eyebrow_raise
            noise = np.random.normal(0, 0.1)
            channels.append(base + noise + 0.1 * np.sin(i * 0.5))
        
        # 9-16: Eye movements (frontal eye fields)
        for i in range(8):
            base = signal.gaze_x * 0.5 + signal.gaze_y * 0.3
            noise = np.random.normal(0, 0.15)
            channels.append(base + noise + signal.eye_squint * 0.2)
        
        # 17-24: Head position (vestibular simulation)
        for i in range(8):
            base = signal.head_tilt_x * 0.4 + signal.head_tilt_y * 0.6
            noise = np.random.normal(0, 0.1)
            channels.append(base + noise)
        
        # 25-32: Mouth/jaw (motor areas)
        for i in range(8):
            base = signal.mouth_open * 0.7
            if signal.tongue_out:
                base += 0.5
            noise = np.random.normal(0, 0.12)
            channels.append(base + noise)
        
        return np.array(channels, dtype=np.float32)
    
    def cursor_control_mode(self, signal: FacialSignal, frame: np.ndarray):
        """Control a cursor with facial expressions"""
        # Calculate velocity from signals
        if self.use_neural_decoder and len(self.signal_buffer) == 64:
            # Use neural decoder
            neural_data = np.array(self.signal_buffer).T  # [32, 64]
            with torch.no_grad():
                x = torch.from_numpy(neural_data).unsqueeze(0)
                velocity = self.decoder(x).squeeze().numpy()
            vx, vy = velocity[0] * 10, velocity[1] * 10
        else:
            # Direct mapping
            vx = signal.gaze_x * 5 + signal.head_tilt_x * 3
            vy = -signal.eyebrow_raise * 5 + signal.head_tilt_y * 3
        
        # Stop on squint
        if signal.eye_squint > 0.7:
            vx, vy = 0, 0
        
        # Boost on mouth open
        if signal.mouth_open > 0.5:
            vx *= 2
            vy *= 2
        
        # Update cursor
        self.cursor_x += vx
        self.cursor_y += vy
        self.cursor_x = np.clip(self.cursor_x, 20, 620)
        self.cursor_y = np.clip(self.cursor_y, 20, 460)
        
        # Draw cursor and trail
        cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 
                  15, (255, 0, 255), -1)
        
        # Draw target zones
        targets = [(100, 100), (540, 100), (100, 380), (540, 380)]
        for tx, ty in targets:
            cv2.circle(frame, (tx, ty), 30, (0, 255, 0), 2)
            # Check if cursor is in target
            if np.sqrt((self.cursor_x - tx)**2 + (self.cursor_y - ty)**2) < 30:
                cv2.circle(frame, (tx, ty), 30, (0, 255, 0), -1)
    
    def game_control_mode(self, signal: FacialSignal, frame: np.ndarray):
        """Simple game: catch falling objects"""
        # Player paddle
        paddle_x = int(320 + signal.head_tilt_x * 200 + signal.gaze_x * 100)
        paddle_x = np.clip(paddle_x, 50, 590)
        paddle_y = 400
        paddle_width = 80 + int(signal.mouth_open * 40)  # Mouth open = bigger paddle
        
        cv2.rectangle(frame, (paddle_x - paddle_width//2, paddle_y), 
                     (paddle_x + paddle_width//2, paddle_y + 20), 
                     (255, 255, 0), -1)
        
        # Spawn objects
        if np.random.random() < 0.02:
            self.game_objects.append({
                'x': np.random.randint(50, 590),
                'y': 0,
                'speed': np.random.uniform(2, 5)
            })
        
        # Update and draw objects
        caught = 0
        remaining_objects = []
        for obj in self.game_objects:
            obj['y'] += obj['speed']
            
            # Check collision
            if (obj['y'] > paddle_y - 10 and obj['y'] < paddle_y + 20 and
                abs(obj['x'] - paddle_x) < paddle_width//2):
                caught += 1
                cv2.putText(frame, "CAUGHT!", (obj['x']-30, obj['y']), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif obj['y'] < 480:
                remaining_objects.append(obj)
                cv2.circle(frame, (obj['x'], int(obj['y'])), 10, (0, 0, 255), -1)
        
        self.game_objects = remaining_objects
    
    def paint_mode(self, signal: FacialSignal, frame: np.ndarray):
        """Draw by moving your head"""
        # Brush position from head movement
        brush_x = int(320 + signal.head_tilt_x * 300 + signal.gaze_x * 50)
        brush_y = int(240 + signal.head_tilt_y * 200 - signal.eyebrow_raise * 100)
        
        # Drawing control
        if signal.mouth_open > 0.3:  # Mouth open = draw
            self.paint_points.append((brush_x, brush_y))
            
        # Clear on tongue out
        if signal.tongue_out:
            self.paint_points = []
        
        # Draw paint trail
        for i in range(1, len(self.paint_points)):
            cv2.line(frame, self.paint_points[i-1], self.paint_points[i], 
                    (0, 255, 255), 3)
        
        # Draw brush
        brush_size = int(5 + signal.eye_squint * 10)
        cv2.circle(frame, (brush_x, brush_y), brush_size, (255, 255, 255), -1)
    
    def draw_signal_visualization(self, frame: np.ndarray, signal: FacialSignal):
        """Draw signal strength indicators"""
        # Signal bars
        bar_x = 10
        bar_width = 150
        bar_height = 15
        spacing = 20
        
        signals = [
            ("Eyebrow", signal.eyebrow_raise, (0, 255, 0)),
            ("Squint", signal.eye_squint, (255, 0, 0)),
            ("Mouth", signal.mouth_open, (0, 0, 255)),
            ("Gaze X", (signal.gaze_x + 1) / 2, (255, 255, 0)),
            ("Gaze Y", (signal.gaze_y + 1) / 2, (255, 0, 255)),
            ("Tilt X", (signal.head_tilt_x + 1) / 2, (0, 255, 255)),
        ]
        
        for i, (name, value, color) in enumerate(signals):
            y = 10 + i * spacing
            
            # Background
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), 
                         (50, 50, 50), -1)
            
            # Value bar
            fill_width = int(value * bar_width)
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill_width, y + bar_height), 
                         color, -1)
            
            # Label
            cv2.putText(frame, name, (bar_x + bar_width + 5, y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mode indicator
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}", 
                   (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Special indicators
        if signal.blink_left:
            cv2.circle(frame, (580, 20), 10, (0, 255, 0), -1)
        if signal.blink_right:
            cv2.circle(frame, (610, 20), 10, (0, 255, 0), -1)
        if signal.tongue_out:
            cv2.putText(frame, "TONGUE!", (270, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    def run(self):
        """Main BCI loop"""
        print("Starting Advanced Webcam BCI...")
        print("Press 'c' to calibrate, 'q' to quit")
        
        cv2.namedWindow('Webcam BCI', cv2.WINDOW_NORMAL)
        mode_index = 0
        mode_names = list(self.control_modes.keys())
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract signals
                signal = self.extract_facial_signals(landmarks)
                
                # Update signal buffer for neural decoder
                neural_signal = self.facial_signals_to_neural(signal)
                self.signal_buffer.append(neural_signal)
                
                # Mode switching with winks
                if signal.blink_left and not signal.blink_right:
                    mode_index = (mode_index - 1) % len(mode_names)
                    self.current_mode = mode_names[mode_index]
                    time.sleep(0.5)  # Debounce
                elif signal.blink_right and not signal.blink_left:
                    mode_index = (mode_index + 1) % len(mode_names)
                    self.current_mode = mode_names[mode_index]
                    time.sleep(0.5)
                
                # Run current mode
                self.control_modes[self.current_mode](signal, frame)
                
                # Draw visualization
                self.draw_signal_visualization(frame, signal)
                
                # Draw face mesh (optional)
                if cv2.waitKey(1) & 0xFF == ord('m'):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.multi_face_landmarks[0],
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
            else:
                cv2.putText(frame, "No face detected", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Webcam BCI', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate()
            elif key == ord(' '):
                # Reset current mode
                self.cursor_x, self.cursor_y = 320, 240
                self.paint_points = []
                self.game_objects = []
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n=== ADVANCED WEBCAM BCI ===\n")
    print("This demo shows how facial expressions can control a computer")
    print("like a real brain-computer interface!\n")
    
    # Check if neural decoder model exists
    use_decoder = os.path.exists("artifacts/model.pt")
    if not use_decoder:
        print("! No trained model found. Using direct control mapping.")
        print("  Run training first for neural decoder integration.\n")
    
    bci = AdvancedWebcamBCI(use_neural_decoder=use_decoder)
    bci.run()
