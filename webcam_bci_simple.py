"""
Simple Webcam BCI - Get started in 2 minutes!
Control a cursor with just your eyebrows and eyes
"""

import cv2
import numpy as np
import time

# Simplified - no mediapipe needed, just OpenCV!

class SimpleWebcamBCI:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Cursor position
        self.x = 320
        self.y = 240
        
        # Calibration
        self.baseline_face_height = None
        self.baseline_eye_count = 2
        
    def detect_signals(self, frame):
        """Simple signal detection using face size and eye visibility"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detect eyes within face
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
        # Calculate signals
        signals = {
            'face_height': h,
            'face_center_x': x + w//2,
            'face_center_y': y + h//2,
            'eye_count': len(eyes),
            'face_width': w
        }
        
        return signals
    
    def calibrate(self, frame):
        """Simple calibration - just get baseline face size"""
        print("Keep your face neutral for calibration...")
        
        signals = self.detect_signals(frame)
        if signals:
            self.baseline_face_height = signals['face_height']
            print(f"Calibrated! Baseline face height: {self.baseline_face_height}")
            return True
        return False
    
    def run(self):
        print("\n=== SIMPLE WEBCAM BCI ===")
        print("Controls:")
        print("- Move head UP (chin up) = Cursor UP")
        print("- Move head DOWN (chin down) = Cursor DOWN")
        print("- Turn head LEFT/RIGHT = Cursor LEFT/RIGHT")
        print("- Close one eye = Slow down")
        print("- Close both eyes = Stop")
        print("\nPress 'c' to calibrate, 'q' to quit\n")
        
        calibrated = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            signals = self.detect_signals(frame)
            
            if signals and calibrated:
                # Calculate movement from signals
                
                # Vertical: Face size change (moving closer/farther)
                if self.baseline_face_height:
                    size_ratio = signals['face_height'] / self.baseline_face_height
                    # Bigger face (closer) = move up, smaller = move down
                    dy = (size_ratio - 1.0) * 20
                else:
                    dy = 0
                
                # Horizontal: Face position
                dx = (signals['face_center_x'] - 320) * 0.05
                
                # Eye-based speed control
                if signals['eye_count'] == 0:
                    # Both eyes closed = stop
                    dx, dy = 0, 0
                elif signals['eye_count'] == 1:
                    # One eye closed = half speed
                    dx *= 0.5
                    dy *= 0.5
                
                # Update cursor
                self.x += dx
                self.y -= dy  # Invert Y
                self.x = np.clip(self.x, 20, 620)
                self.y = np.clip(self.y, 20, 460)
                
                # Draw info
                cv2.putText(frame, f"Size ratio: {size_ratio:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Eyes detected: {signals['eye_count']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw cursor
            cv2.circle(frame, (int(self.x), int(self.y)), 20, (255, 0, 255), -1)
            
            # Draw crosshair
            cv2.line(frame, (int(self.x)-30, int(self.y)), 
                    (int(self.x)+30, int(self.y)), (255, 0, 255), 2)
            cv2.line(frame, (int(self.x), int(self.y)-30), 
                    (int(self.x), int(self.y)+30), (255, 0, 255), 2)
            
            if not calibrated:
                cv2.putText(frame, "Press 'c' to calibrate", 
                           (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Simple Webcam BCI', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibrated = self.calibrate(frame)
            elif key == ord(' '):
                # Reset cursor
                self.x, self.y = 320, 240
        
        self.cap.release()
        cv2.destroyAllWindows()


# Even simpler: Nose tracking!
class NoseTrackingBCI:
    """Ultra simple - just track nose position"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Using face detection to approximate nose position
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Trail effect
        self.trail = []
        self.max_trail = 20
        
    def run(self):
        print("\n=== NOSE TRACKING BCI ===")
        print("Your nose is the cursor!")
        print("Move your head to move the cursor")
        print("Make a game: try to draw shapes!\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Approximate nose position (center of face, slightly down)
                nose_x = x + w // 2
                nose_y = y + int(h * 0.6)
                
                # Add to trail
                self.trail.append((nose_x, nose_y))
                if len(self.trail) > self.max_trail:
                    self.trail.pop(0)
                
                # Draw trail
                for i in range(1, len(self.trail)):
                    # Fade effect
                    thickness = int(i / len(self.trail) * 5) + 1
                    color_intensity = int(i / len(self.trail) * 255)
                    cv2.line(frame, self.trail[i-1], self.trail[i], 
                            (color_intensity, 0, 255-color_intensity), thickness)
                
                # Draw nose point
                cv2.circle(frame, (nose_x, nose_y), 10, (0, 255, 0), -1)
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
                # Fun: Draw targets to hit
                targets = [(100, 100), (540, 100), (320, 240), (100, 380), (540, 380)]
                for tx, ty in targets:
                    cv2.circle(frame, (tx, ty), 30, (0, 255, 255), 2)
                    # Check if nose hit target
                    if np.sqrt((nose_x - tx)**2 + (nose_y - ty)**2) < 30:
                        cv2.circle(frame, (tx, ty), 30, (0, 255, 0), -1)
                        cv2.putText(frame, "HIT!", (tx-20, ty), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, "NOSE TRACKING GAME", (200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Hit all the targets!", (220, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Nose Tracking BCI', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\nWEBCAM BCI DEMOS")
    print("1. Simple Face Control (head movement + eyes)")
    print("2. Nose Tracking Game")
    print("3. Advanced BCI (requires mediapipe)")
    
    choice = input("\nChoose 1-3: ")
    
    if choice == '1':
        bci = SimpleWebcamBCI()
        bci.run()
    elif choice == '2':
        bci = NoseTrackingBCI()
        bci.run()
    else:
        print("Run 'python examples/advanced_webcam_bci.py' for the full version!")
