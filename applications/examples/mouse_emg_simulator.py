"""
Turn your mouse into an EMG sensor!
Tiny movements = muscle signals
"""

import numpy as np
import pygame
import time
from collections import deque


class MouseEMGController:
    """
    Detect micro-movements as 'muscle signals'
    Keep mouse still and make tiny twitches in different directions
    """
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Mouse EMG Brain Control")
        
        # Signal processing
        self.movement_buffer = deque(maxlen=30)
        self.baseline_x = 0
        self.baseline_y = 0
        self.calibrated = False
        
        # Visual feedback
        self.cursor_x = 400
        self.cursor_y = 300
        
    def calibrate(self):
        """Calibrate baseline mouse position"""
        print("CALIBRATION: Keep mouse perfectly still for 3 seconds...")
        
        positions = []
        for _ in range(90):  # 3 seconds at 30fps
            x, y = pygame.mouse.get_pos()
            positions.append((x, y))
            time.sleep(0.033)
        
        # Calculate average position
        self.baseline_x = np.mean([p[0] for p in positions])
        self.baseline_y = np.mean([p[1] for p in positions])
        self.calibrated = True
        
        print(f"Calibrated! Baseline: ({self.baseline_x:.1f}, {self.baseline_y:.1f})")
        print("\nNow make tiny movements:")
        print("- Twitch UP = Forward")
        print("- Twitch DOWN = Backward")  
        print("- Twitch LEFT/RIGHT = Turn")
        print("- Hold STILL = Stop")
    
    def detect_micro_movement(self):
        """Detect tiny intentional movements"""
        x, y = pygame.mouse.get_pos()
        
        # Calculate micro-displacement from baseline
        dx = x - self.baseline_x
        dy = y - self.baseline_y
        
        # Reset baseline if moved too far (repositioning)
        if abs(dx) > 50 or abs(dy) > 50:
            self.baseline_x = x
            self.baseline_y = y
            return 0, 0
        
        # Threshold for intentional micro-movements
        threshold = 2  # pixels
        
        signal_x = 0
        signal_y = 0
        
        if abs(dx) > threshold:
            signal_x = np.tanh(dx / 10)  # Normalize to -1 to 1
        if abs(dy) > threshold:
            signal_y = np.tanh(dy / 10)
        
        return signal_x, signal_y
    
    def run(self):
        """Main control loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Calibrate first
        self.calibrate()
        
        font = pygame.font.Font(None, 36)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear screen
            self.screen.fill((20, 20, 20))
            
            # Detect muscle signals
            signal_x, signal_y = self.detect_micro_movement()
            
            # Update cursor based on signals
            self.cursor_x += signal_x * 5
            self.cursor_y -= signal_y * 5  # Invert Y
            
            # Keep in bounds
            self.cursor_x = np.clip(self.cursor_x, 50, 750)
            self.cursor_y = np.clip(self.cursor_y, 50, 550)
            
            # Draw EMG visualization
            # Signal strength bars
            bar_width = 200
            bar_height = 20
            
            # X signal (green/red)
            x_color = (0, 255, 0) if signal_x > 0 else (255, 0, 0)
            x_width = int(abs(signal_x) * bar_width)
            pygame.draw.rect(self.screen, x_color, 
                           (400 - x_width//2, 520, x_width, bar_height))
            
            # Y signal (blue/yellow)
            y_color = (0, 0, 255) if signal_y > 0 else (255, 255, 0)
            y_width = int(abs(signal_y) * bar_width)
            pygame.draw.rect(self.screen, y_color,
                           (400 - y_width//2, 550, y_width, bar_height))
            
            # Draw cursor
            pygame.draw.circle(self.screen, (255, 0, 255), 
                             (int(self.cursor_x), int(self.cursor_y)), 20)
            
            # Instructions
            text = font.render("Make tiny mouse movements!", True, (255, 255, 255))
            self.screen.blit(text, (200, 20))
            
            # Signal values
            signal_text = font.render(f"X: {signal_x:+.2f}  Y: {signal_y:+.2f}", 
                                    True, (255, 255, 255))
            self.screen.blit(signal_text, (250, 480))
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()


# Even simpler: Meditation detector
class BreathingBrainwaves:
    """
    Use spacebar timing as 'brainwaves'
    Slow, regular pressing = meditation/calm = forward
    Fast, irregular = stressed = stop
    """
    
    def __init__(self):
        self.press_times = deque(maxlen=10)
        self.last_press = time.time()
    
    def detect_mental_state(self):
        """Analyze spacebar rhythm"""
        current_time = time.time()
        
        # Add press interval
        interval = current_time - self.last_press
        self.press_times.append(interval)
        self.last_press = current_time
        
        if len(self.press_times) < 3:
            return 'neutral', 0
        
        # Calculate rhythm regularity
        intervals = list(self.press_times)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Regular slow breathing (4-6 seconds) = calm
        if 3 < mean_interval < 7 and std_interval < 1:
            return 'calm', 0.8
        
        # Fast irregular = stressed
        elif mean_interval < 2 and std_interval > 0.5:
            return 'stressed', 0.2
        
        else:
            return 'neutral', 0.5


if __name__ == "__main__":
    print("\nPOOR MAN'S BRAIN CONTROL OPTIONS:\n")
    print("1. Mouse Micro-Movements (EMG simulator)")
    print("2. Webcam Face Control")
    print("3. Phone Tilt Control")
    print("4. Breathing Rhythm (spacebar)")
    
    choice = input("\nChoose 1-4: ")
    
    if choice == '1':
        controller = MouseEMGController()
        controller.run()
    else:
        print("Run the appropriate example file!")
