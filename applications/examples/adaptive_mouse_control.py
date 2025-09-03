"""
Adaptive Mouse Control System
Learns your mouse movement patterns and adapts to your style
"""

import numpy as np
import time
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# For actual mouse control
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    print("Install pyautogui for actual mouse control: pip install pyautogui")

# For mouse tracking
try:
    from pynput import mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("Install pynput for mouse tracking: pip install pynput")


@dataclass
class MouseProfile:
    """Learned mouse movement profile"""
    # Movement characteristics
    avg_speed: float = 1.0
    acceleration_curve: List[float] = None
    preferred_smoothing: float = 0.3
    
    # Precision vs Speed preference
    precision_zones: List[Dict] = None  # Areas where user moves slowly
    speed_zones: List[Dict] = None      # Areas where user moves quickly
    
    # Click patterns
    double_click_speed: float = 0.3
    drag_threshold: float = 5.0
    preferred_button: str = "left"
    
    # Gesture patterns
    common_paths: List[List[Tuple[float, float]]] = None
    gesture_shortcuts: Dict[str, str] = None
    
    # Ergonomics
    comfort_zone: Dict[str, float] = None  # Most used screen area
    fatigue_indicators: Dict[str, float] = None
    
    # Statistics
    total_distance: float = 0.0
    total_clicks: int = 0
    session_count: int = 0
    total_time: float = 0.0
    
    def __post_init__(self):
        if self.acceleration_curve is None:
            self.acceleration_curve = [1.0] * 10
        if self.precision_zones is None:
            self.precision_zones = []
        if self.speed_zones is None:
            self.speed_zones = []
        if self.common_paths is None:
            self.common_paths = []
        if self.gesture_shortcuts is None:
            self.gesture_shortcuts = {}
        if self.comfort_zone is None:
            self.comfort_zone = {"x": 0.5, "y": 0.5, "radius": 0.3}
        if self.fatigue_indicators is None:
            self.fatigue_indicators = {"jitter": 0.0, "speed_decline": 0.0}


class AdaptiveMouseController:
    """
    Learns and adapts to individual mouse usage patterns
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.save_dir = Path("mouse_profiles")
        self.save_dir.mkdir(exist_ok=True)
        
        # Load or create profile
        self.profile = self.load_profile()
        
        # Movement tracking
        self.movement_buffer = deque(maxlen=1000)
        self.click_buffer = deque(maxlen=100)
        self.gesture_buffer = deque(maxlen=50)
        
        # Real-time state
        self.last_x = 0
        self.last_y = 0
        self.last_time = time.time()
        self.is_dragging = False
        self.session_start = time.time()
        
        # Learning parameters
        self.learning_enabled = True
        self.adaptation_rate = 0.01
        
        # Mouse listener
        self.mouse_listener = None
        if HAS_PYNPUT:
            self.start_tracking()
    
    def load_profile(self) -> MouseProfile:
        """Load user mouse profile"""
        profile_path = self.save_dir / f"{self.user_id}_mouse_profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                data = json.load(f)
                return MouseProfile(**data)
        return MouseProfile()
    
    def save_profile(self):
        """Save user mouse profile"""
        profile_path = self.save_dir / f"{self.user_id}_mouse_profile.json"
        
        # Update session stats
        self.profile.session_count += 1
        self.profile.total_time += time.time() - self.session_start
        
        with open(profile_path, 'w') as f:
            json.dump(asdict(self.profile), f, indent=2)
    
    def start_tracking(self):
        """Start tracking mouse movements"""
        def on_move(x, y):
            self.track_movement(x, y)
        
        def on_click(x, y, button, pressed):
            if pressed:
                self.track_click(x, y, button)
        
        def on_scroll(x, y, dx, dy):
            self.track_scroll(x, y, dx, dy)
        
        self.mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll
        )
        self.mouse_listener.start()
    
    def track_movement(self, x: int, y: int):
        """Track mouse movement and learn patterns"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt > 0 and self.last_x != 0:
            # Calculate movement metrics
            dx = x - self.last_x
            dy = y - self.last_y
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / dt if dt > 0 else 0
            
            # Track movement
            movement = {
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy,
                'speed': speed,
                'time': current_time,
                'dt': dt
            }
            self.movement_buffer.append(movement)
            
            # Update total distance
            self.profile.total_distance += distance
            
            # Learn from movement
            if self.learning_enabled:
                self.learn_movement_patterns(movement)
        
        self.last_x = x
        self.last_y = y
        self.last_time = current_time
    
    def track_click(self, x: int, y: int, button):
        """Track mouse clicks and learn patterns"""
        click_data = {
            'x': x,
            'y': y,
            'button': str(button),
            'time': time.time()
        }
        self.click_buffer.append(click_data)
        self.profile.total_clicks += 1
        
        # Detect double-clicks
        if len(self.click_buffer) >= 2:
            last_click = self.click_buffer[-2]
            time_diff = click_data['time'] - last_click['time']
            
            if time_diff < 0.5 and last_click['button'] == click_data['button']:
                # Update double-click speed preference
                self.profile.double_click_speed = (
                    0.9 * self.profile.double_click_speed + 
                    0.1 * time_diff
                )
    
    def track_scroll(self, x: int, y: int, dx: int, dy: int):
        """Track scrolling patterns"""
        # Could extend to learn scroll speed preferences
        pass
    
    def learn_movement_patterns(self, movement: Dict):
        """Learn from movement patterns"""
        if len(self.movement_buffer) < 10:
            return
        
        recent_movements = list(self.movement_buffer)[-50:]
        
        # 1. Learn speed preferences
        speeds = [m['speed'] for m in recent_movements]
        avg_speed = np.mean(speeds)
        
        # Update average speed with exponential moving average
        self.profile.avg_speed = (
            0.95 * self.profile.avg_speed + 
            0.05 * avg_speed
        )
        
        # 2. Learn acceleration curve
        # Group movements by initial speed and see how they accelerate
        if len(recent_movements) > 20:
            for i in range(1, len(recent_movements)):
                prev_speed = recent_movements[i-1]['speed']
                curr_speed = recent_movements[i]['speed']
                
                if prev_speed > 0:
                    acceleration = curr_speed / prev_speed
                    # Bin by speed range
                    speed_bin = min(int(prev_speed / 100), 9)
                    
                    # Update acceleration curve
                    self.profile.acceleration_curve[speed_bin] = (
                        0.95 * self.profile.acceleration_curve[speed_bin] +
                        0.05 * acceleration
                    )
        
        # 3. Detect precision zones (areas with slow, careful movement)
        slow_movements = [m for m in recent_movements if m['speed'] < 50]
        if len(slow_movements) > 5:
            # Find common slow areas
            x_coords = [m['x'] for m in slow_movements]
            y_coords = [m['y'] for m in slow_movements]
            
            # Simple clustering - could use k-means for better results
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # Check if this is a new precision zone
            is_new_zone = True
            for zone in self.profile.precision_zones:
                dist = np.sqrt((zone['x'] - center_x)**2 + (zone['y'] - center_y)**2)
                if dist < 100:
                    # Update existing zone
                    zone['x'] = 0.9 * zone['x'] + 0.1 * center_x
                    zone['y'] = 0.9 * zone['y'] + 0.1 * center_y
                    zone['count'] += 1
                    is_new_zone = False
                    break
            
            if is_new_zone and len(self.profile.precision_zones) < 10:
                self.profile.precision_zones.append({
                    'x': center_x,
                    'y': center_y,
                    'radius': 50,
                    'count': 1
                })
        
        # 4. Learn gesture patterns
        self.detect_gesture_patterns(recent_movements)
        
        # 5. Update comfort zone
        x_positions = [m['x'] for m in recent_movements]
        y_positions = [m['y'] for m in recent_movements]
        
        self.profile.comfort_zone['x'] = (
            0.99 * self.profile.comfort_zone['x'] + 
            0.01 * np.mean(x_positions) / 1920  # Normalize to screen
        )
        self.profile.comfort_zone['y'] = (
            0.99 * self.profile.comfort_zone['y'] + 
            0.01 * np.mean(y_positions) / 1080
        )
        
        # 6. Detect fatigue (increasing jitter over time)
        if len(self.movement_buffer) > 100:
            early_movements = list(self.movement_buffer)[:50]
            late_movements = list(self.movement_buffer)[-50:]
            
            early_jitter = np.std([m['speed'] for m in early_movements])
            late_jitter = np.std([m['speed'] for m in late_movements])
            
            if late_jitter > early_jitter * 1.5:
                self.profile.fatigue_indicators['jitter'] += 0.1
    
    def detect_gesture_patterns(self, movements: List[Dict]):
        """Detect common movement patterns (gestures)"""
        if len(movements) < 10:
            return
        
        # Simple gesture detection - look for specific patterns
        # Example: Quick back-and-forth = "shake"
        x_positions = [m['x'] for m in movements[-10:]]
        x_changes = np.diff(x_positions)
        
        # Detect horizontal shake
        if len(x_changes) > 4:
            direction_changes = np.sum(np.diff(np.sign(x_changes)) != 0)
            if direction_changes >= 3:
                self.gesture_buffer.append({
                    'type': 'horizontal_shake',
                    'time': time.time()
                })
    
    def apply_profile_to_movement(self, dx: float, dy: float, 
                                 current_speed: float) -> Tuple[float, float]:
        """Apply learned profile to enhance movement"""
        # Get screen position for zone detection
        if HAS_PYAUTOGUI:
            x, y = pyautogui.position()
        else:
            x, y = self.last_x, self.last_y
        
        # Check if in precision zone
        in_precision_zone = False
        for zone in self.profile.precision_zones:
            dist = np.sqrt((x - zone['x'])**2 + (y - zone['y'])**2)
            if dist < zone['radius']:
                in_precision_zone = True
                break
        
        # Apply speed scaling
        if in_precision_zone:
            # Slow down in precision zones
            speed_scale = 0.5
        else:
            # Use learned average speed
            speed_scale = self.profile.avg_speed / 100.0
        
        # Apply acceleration curve
        speed_bin = min(int(current_speed / 100), 9)
        acceleration = self.profile.acceleration_curve[speed_bin]
        
        # Apply smoothing
        smooth_factor = self.profile.preferred_smoothing
        
        # Calculate enhanced movement
        enhanced_dx = dx * speed_scale * acceleration
        enhanced_dy = dy * speed_scale * acceleration
        
        # Apply smoothing
        if hasattr(self, 'smooth_dx'):
            self.smooth_dx = smooth_factor * self.smooth_dx + (1 - smooth_factor) * enhanced_dx
            self.smooth_dy = smooth_factor * self.smooth_dy + (1 - smooth_factor) * enhanced_dy
        else:
            self.smooth_dx = enhanced_dx
            self.smooth_dy = enhanced_dy
        
        return self.smooth_dx, self.smooth_dy
    
    def create_visualization_window(self):
        """Create a window to visualize mouse patterns"""
        root = tk.Tk()
        root.title(f"Mouse Profile - {self.user_id}")
        root.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)
        
        # Tab 1: Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        stats_text = tk.Text(stats_frame, height=20, width=50)
        stats_text.pack(padx=10, pady=10)
        
        def update_stats():
            stats_text.delete(1.0, tk.END)
            stats = f"""
Mouse Usage Statistics for {self.user_id}
{'='*40}

Movement Patterns:
- Average Speed: {self.profile.avg_speed:.1f} pixels/sec
- Total Distance: {self.profile.total_distance/1000:.1f} meters
- Preferred Smoothing: {self.profile.preferred_smoothing:.2f}

Click Patterns:
- Total Clicks: {self.profile.total_clicks}
- Double-click Speed: {self.profile.double_click_speed*1000:.0f} ms
- Preferred Button: {self.profile.preferred_button}

Zones:
- Precision Zones: {len(self.profile.precision_zones)}
- Speed Zones: {len(self.profile.speed_zones)}

Comfort Zone:
- Center X: {self.profile.comfort_zone['x']*100:.1f}%
- Center Y: {self.profile.comfort_zone['y']*100:.1f}%

Session Info:
- Current Session: {(time.time() - self.session_start)/60:.1f} min
- Total Sessions: {self.profile.session_count}
- Total Time: {self.profile.total_time/3600:.1f} hours

Fatigue Indicators:
- Jitter Level: {self.profile.fatigue_indicators['jitter']:.2f}
"""
            stats_text.insert(1.0, stats)
            root.after(1000, update_stats)
        
        update_stats()
        
        # Tab 2: Movement Heatmap
        heatmap_frame = ttk.Frame(notebook)
        notebook.add(heatmap_frame, text="Movement Heatmap")
        
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, heatmap_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        def update_heatmap():
            ax.clear()
            
            if len(self.movement_buffer) > 50:
                # Create heatmap from recent movements
                x_coords = [m['x'] for m in self.movement_buffer]
                y_coords = [m['y'] for m in self.movement_buffer]
                
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(
                    x_coords, y_coords, bins=50
                )
                
                ax.imshow(hist.T, origin='lower', cmap='hot', 
                         aspect='auto', interpolation='gaussian')
                ax.set_title('Mouse Movement Heatmap')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                
                # Mark precision zones
                for zone in self.profile.precision_zones:
                    circle = plt.Circle(
                        (zone['x'], zone['y']), 
                        zone['radius'], 
                        fill=False, 
                        color='cyan', 
                        linewidth=2
                    )
                    ax.add_patch(circle)
            
            canvas.draw()
            root.after(2000, update_heatmap)
        
        update_heatmap()
        
        # Tab 3: Speed Profile
        speed_frame = ttk.Frame(notebook)
        notebook.add(speed_frame, text="Speed Profile")
        
        fig2 = plt.Figure(figsize=(6, 4), dpi=100)
        ax2 = fig2.add_subplot(111)
        canvas2 = FigureCanvasTkAgg(fig2, speed_frame)
        canvas2.get_tk_widget().pack(fill='both', expand=True)
        
        def update_speed_profile():
            ax2.clear()
            
            # Plot acceleration curve
            speeds = np.arange(0, 1000, 100)
            accelerations = self.profile.acceleration_curve
            
            ax2.plot(speeds, accelerations, 'b-', linewidth=2, label='Acceleration Curve')
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Neutral')
            
            ax2.set_xlabel('Initial Speed (pixels/sec)')
            ax2.set_ylabel('Acceleration Factor')
            ax2.set_title('Learned Acceleration Profile')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            canvas2.draw()
            root.after(5000, update_speed_profile)
        
        update_speed_profile()
        
        # Control buttons
        control_frame = ttk.Frame(root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        def toggle_learning():
            self.learning_enabled = not self.learning_enabled
            learn_btn.config(
                text=f"Learning: {'ON' if self.learning_enabled else 'OFF'}"
            )
        
        learn_btn = ttk.Button(
            control_frame, 
            text=f"Learning: {'ON' if self.learning_enabled else 'OFF'}",
            command=toggle_learning
        )
        learn_btn.pack(side='left', padx=5)
        
        def save_and_quit():
            self.save_profile()
            if self.mouse_listener:
                self.mouse_listener.stop()
            root.destroy()
        
        ttk.Button(
            control_frame, 
            text="Save Profile", 
            command=self.save_profile
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame, 
            text="Quit", 
            command=save_and_quit
        ).pack(side='left', padx=5)
        
        root.protocol("WM_DELETE_WINDOW", save_and_quit)
        root.mainloop()


class MouseGestureTrainer:
    """Train custom mouse gestures"""
    
    def __init__(self, profile: MouseProfile):
        self.profile = profile
        self.recording = False
        self.current_gesture = []
        
    def record_gesture(self, name: str):
        """Record a new mouse gesture"""
        print(f"\nRecording gesture: {name}")
        print("Move the mouse to create the gesture, then press Enter")
        
        self.recording = True
        self.current_gesture = []
        
        # Record mouse movements
        # ... (implementation depends on specific needs)
        
        # Save gesture
        self.profile.gesture_shortcuts[name] = self.current_gesture
        print(f"Gesture '{name}' recorded!")


def demo_adaptive_mouse():
    """Run the adaptive mouse demo"""
    print("\n=== Adaptive Mouse Control System ===")
    print("This system learns your mouse movement patterns!")
    
    if not HAS_PYNPUT:
        print("\nERROR: pynput not installed")
        print("Install with: pip install pynput")
        return
    
    user_id = input("Enter your name (or press Enter for 'default'): ").strip()
    if not user_id:
        user_id = "default"
    
    controller = AdaptiveMouseController(user_id=user_id)
    
    print(f"\nTracking mouse movements for {user_id}...")
    print("The system will learn:")
    print("- Your preferred movement speed")
    print("- Areas where you need precision")
    print("- Your acceleration patterns")
    print("- Common gestures you make")
    print("\nA visualization window will open...")
    
    try:
        controller.create_visualization_window()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        controller.save_profile()
        if controller.mouse_listener:
            controller.mouse_listener.stop()


if __name__ == "__main__":
    demo_adaptive_mouse()
