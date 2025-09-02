"""
Adaptive Control Demo - Learns your preferences over time
Works with basic webcam, no complex dependencies
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass
class UserPreferences:
    """Stores learned user preferences"""
    sensitivity_x: float = 1.0      # Horizontal movement sensitivity
    sensitivity_y: float = 1.0      # Vertical movement sensitivity
    dead_zone: float = 0.1          # Minimum movement threshold
    smoothing: float = 0.3          # Movement smoothing factor
    preferred_speed: float = 1.0    # Overall speed preference
    gesture_timings: Dict[str, float] = None  # How long user holds gestures
    
    def __post_init__(self):
        if self.gesture_timings is None:
            self.gesture_timings = {}


class AdaptiveController:
    """
    Learns and adapts to individual user control preferences
    No ML required - just smart heuristics!
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.save_dir = Path("user_preferences")
        self.save_dir.mkdir(exist_ok=True)
        
        # Load or create preferences
        self.prefs = self.load_preferences()
        
        # Performance tracking
        self.movement_history = deque(maxlen=1000)
        self.gesture_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
        # Adaptation parameters
        self.learning_rate = 0.01
        self.adaptation_enabled = True
        
        # Current state
        self.cursor_x = 320
        self.cursor_y = 240
        self.target_x = 320
        self.target_y = 240
        
        # Timing
        self.last_update = time.time()
        self.session_start = time.time()
        
    def load_preferences(self) -> UserPreferences:
        """Load user preferences from file"""
        pref_file = self.save_dir / f"{self.user_id}_preferences.json"
        
        if pref_file.exists():
            with open(pref_file, 'r') as f:
                data = json.load(f)
                return UserPreferences(**data)
        else:
            return UserPreferences()
    
    def save_preferences(self):
        """Save user preferences to file"""
        pref_file = self.save_dir / f"{self.user_id}_preferences.json"
        with open(pref_file, 'w') as f:
            json.dump(asdict(self.prefs), f, indent=2)
    
    def process_movement(self, raw_dx: float, raw_dy: float) -> Tuple[float, float]:
        """Process raw movement with learned preferences"""
        # Apply dead zone
        if abs(raw_dx) < self.prefs.dead_zone:
            raw_dx = 0
        if abs(raw_dy) < self.prefs.dead_zone:
            raw_dy = 0
        
        # Apply sensitivity
        dx = raw_dx * self.prefs.sensitivity_x * self.prefs.preferred_speed
        dy = raw_dy * self.prefs.sensitivity_y * self.prefs.preferred_speed
        
        # Track raw movement for learning
        self.movement_history.append({
            'raw_dx': raw_dx,
            'raw_dy': raw_dy,
            'processed_dx': dx,
            'processed_dy': dy,
            'timestamp': time.time()
        })
        
        return dx, dy
    
    def update_cursor(self, dx: float, dy: float):
        """Update cursor with smoothing"""
        # Smooth movement
        self.cursor_x += dx * (1 - self.prefs.smoothing)
        self.cursor_y += dy * (1 - self.prefs.smoothing)
        
        # Boundary checking
        self.cursor_x = np.clip(self.cursor_x, 0, 640)
        self.cursor_y = np.clip(self.cursor_y, 0, 480)
    
    def learn_from_behavior(self):
        """Adapt to user behavior patterns"""
        if not self.adaptation_enabled or len(self.movement_history) < 100:
            return
        
        recent_movements = list(self.movement_history)[-100:]
        
        # 1. Learn preferred speed
        avg_movement = np.mean([
            np.sqrt(m['raw_dx']**2 + m['raw_dy']**2) 
            for m in recent_movements
        ])
        
        if avg_movement > 0:
            # User making larger movements = wants higher sensitivity
            if avg_movement > 0.5:
                self.prefs.preferred_speed *= (1 + self.learning_rate)
            elif avg_movement < 0.2:
                self.prefs.preferred_speed *= (1 - self.learning_rate)
            
            self.prefs.preferred_speed = np.clip(self.prefs.preferred_speed, 0.5, 2.0)
        
        # 2. Learn directional preferences
        x_movements = [abs(m['raw_dx']) for m in recent_movements if m['raw_dx'] != 0]
        y_movements = [abs(m['raw_dy']) for m in recent_movements if m['raw_dy'] != 0]
        
        if x_movements and y_movements:
            x_avg = np.mean(x_movements)
            y_avg = np.mean(y_movements)
            
            # Adjust sensitivity based on usage patterns
            if x_avg > y_avg * 1.5:
                # User moves more horizontally
                self.prefs.sensitivity_x *= (1 + self.learning_rate * 0.5)
                self.prefs.sensitivity_y *= (1 - self.learning_rate * 0.5)
            elif y_avg > x_avg * 1.5:
                # User moves more vertically
                self.prefs.sensitivity_y *= (1 + self.learning_rate * 0.5)
                self.prefs.sensitivity_x *= (1 - self.learning_rate * 0.5)
        
        # 3. Learn dead zone preference
        small_movements = [
            m for m in recent_movements 
            if 0 < np.sqrt(m['raw_dx']**2 + m['raw_dy']**2) < 0.1
        ]
        
        if len(small_movements) > len(recent_movements) * 0.3:
            # Many small movements = reduce dead zone
            self.prefs.dead_zone *= (1 - self.learning_rate)
        elif len(small_movements) < len(recent_movements) * 0.1:
            # Few small movements = increase dead zone
            self.prefs.dead_zone *= (1 + self.learning_rate)
        
        self.prefs.dead_zone = np.clip(self.prefs.dead_zone, 0.05, 0.3)
        
        # 4. Learn smoothing preference from jitter
        if len(recent_movements) > 10:
            # Calculate jitter (variance in movement direction)
            directions = []
            for i in range(1, len(recent_movements)):
                if recent_movements[i]['raw_dx'] != 0 or recent_movements[i]['raw_dy'] != 0:
                    angle = np.arctan2(recent_movements[i]['raw_dy'], 
                                     recent_movements[i]['raw_dx'])
                    directions.append(angle)
            
            if len(directions) > 5:
                direction_variance = np.var(np.diff(directions))
                
                if direction_variance > 0.5:
                    # High jitter = increase smoothing
                    self.prefs.smoothing = min(0.8, self.prefs.smoothing + self.learning_rate)
                elif direction_variance < 0.1:
                    # Low jitter = decrease smoothing
                    self.prefs.smoothing = max(0.1, self.prefs.smoothing - self.learning_rate)
    
    def detect_user_intent(self, movements: List[Dict]) -> str:
        """Detect what the user is trying to do"""
        if len(movements) < 5:
            return "idle"
        
        recent = movements[-5:]
        
        # Check for circular motion
        angles = []
        for m in recent:
            if m['raw_dx'] != 0 or m['raw_dy'] != 0:
                angle = np.arctan2(m['raw_dy'], m['raw_dx'])
                angles.append(angle)
        
        if len(angles) >= 4:
            # Check if angles are increasing/decreasing consistently
            angle_diffs = np.diff(angles)
            if np.all(angle_diffs > 0) or np.all(angle_diffs < 0):
                return "circular_motion"
        
        # Check for precise positioning
        speeds = [np.sqrt(m['raw_dx']**2 + m['raw_dy']**2) for m in recent]
        if np.mean(speeds) < 0.1 and np.std(speeds) < 0.05:
            return "precise_positioning"
        
        # Check for rapid movement
        if np.mean(speeds) > 0.5:
            return "rapid_movement"
        
        return "normal_movement"
    
    def provide_feedback(self, frame: np.ndarray):
        """Visual feedback about adaptation"""
        # Adaptation status
        status_color = (0, 255, 0) if self.adaptation_enabled else (0, 0, 255)
        cv2.putText(frame, f"Adaptation: {'ON' if self.adaptation_enabled else 'OFF'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Current preferences
        cv2.putText(frame, f"Speed: {self.prefs.preferred_speed:.1f}x", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Smoothing: {self.prefs.smoothing:.0%}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Dead zone: {self.prefs.dead_zone:.2f}", 
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # User intent
        if len(self.movement_history) > 5:
            intent = self.detect_user_intent(list(self.movement_history))
            intent_color = {
                'circular_motion': (255, 0, 255),
                'precise_positioning': (0, 255, 255),
                'rapid_movement': (255, 255, 0),
                'normal_movement': (200, 200, 200)
            }.get(intent, (200, 200, 200))
            
            cv2.putText(frame, f"Mode: {intent.replace('_', ' ').title()}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, intent_color, 2)
        
        # Session time
        session_time = int(time.time() - self.session_start)
        cv2.putText(frame, f"Session: {session_time//60}:{session_time%60:02d}", 
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def draw_performance_graph(self, frame: np.ndarray):
        """Draw a simple performance graph"""
        if len(self.movement_history) < 50:
            return
        
        # Graph area
        graph_x, graph_y = 450, 350
        graph_w, graph_h = 180, 100
        
        # Background
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h), 
                     (50, 50, 50), -1)
        
        # Plot recent movement magnitudes
        recent = list(self.movement_history)[-50:]
        magnitudes = [
            np.sqrt(m['raw_dx']**2 + m['raw_dy']**2) 
            for m in recent
        ]
        
        # Normalize and plot
        if magnitudes:
            max_mag = max(magnitudes) or 1
            for i, mag in enumerate(magnitudes):
                x = graph_x + int(i * graph_w / len(magnitudes))
                y = graph_y + graph_h - int(mag / max_mag * graph_h)
                
                if i > 0:
                    prev_x = graph_x + int((i-1) * graph_w / len(magnitudes))
                    prev_y = graph_y + graph_h - int(magnitudes[i-1] / max_mag * graph_h)
                    cv2.line(frame, (prev_x, prev_y), (x, y), (0, 255, 0), 1)
        
        cv2.putText(frame, "Movement Activity", (graph_x, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_demo(self):
        """Run the adaptive control demo"""
        print("\n=== Adaptive Control Demo ===")
        print("The system learns your preferences over time!")
        print("Controls:")
        print("- Move mouse to control cursor")
        print("- Press 'a' to toggle adaptation")
        print("- Press 'r' to reset preferences")
        print("- Press 's' to save preferences")
        print("- Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        last_face_x = 320
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple face tracking
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_center_x = x + w // 2
                
                # Calculate movement
                raw_dx = (face_center_x - last_face_x) / 640.0
                last_face_x = face_center_x
                
                # Simple up/down from face size
                expected_size = 150
                raw_dy = (expected_size - h) / 480.0
                
                # Process with preferences
                dx, dy = self.process_movement(raw_dx, raw_dy)
                self.update_cursor(dx, dy)
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw cursor
            cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 
                      15, (255, 0, 255), -1)
            
            # Draw UI
            self.provide_feedback(frame)
            self.draw_performance_graph(frame)
            
            # Periodic learning
            if time.time() - self.last_update > 2.0:
                self.learn_from_behavior()
                self.last_update = time.time()
            
            cv2.imshow('Adaptive Control', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.adaptation_enabled = not self.adaptation_enabled
                print(f"Adaptation: {'ON' if self.adaptation_enabled else 'OFF'}")
            elif key == ord('r'):
                self.prefs = UserPreferences()
                print("Preferences reset!")
            elif key == ord('s'):
                self.save_preferences()
                print("Preferences saved!")
        
        # Auto-save on exit
        self.save_preferences()
        print(f"\nSession complete! Preferences saved for {self.user_id}")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n=== Adaptive Control System ===")
    print("This system learns your control preferences over time!")
    
    user_id = input("Enter your name (or press Enter for 'default'): ").strip()
    if not user_id:
        user_id = "default"
    
    controller = AdaptiveController(user_id=user_id)
    controller.run_demo()
