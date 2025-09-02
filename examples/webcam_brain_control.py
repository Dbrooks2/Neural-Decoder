`"""
Poor man's BCI: Use webcam + facial expressions as "neural" signals
No hardware needed - just your laptop camera!
"""

import cv2
import numpy as np
import time
from collections import deque

# You'll need: pip install opencv-python mediapipe

import mediapipe as mp


class WebcamBrainController:
    """
    Simulate BCI using facial expressions and eye movements
    - Raise eyebrows = Forward
    - Squint = Stop
    - Look left/right = Turn
    - Blink twice = Emergency stop
    """
    
    def __init__(self):
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        
        # MediaPipe face detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Control state
        self.cursor_x = 320
        self.cursor_y = 240
        self.blink_buffer = deque(maxlen=30)  # 1 second at 30fps
        
    def detect_control_signals(self, landmarks):
        """Extract 'neural' signals from face"""
        signals = {}
        
        # Eyebrow height (forehead wrinkles = forward)
        left_eyebrow = landmarks[70].y
        nose_tip = landmarks[1].y
        eyebrow_raise = nose_tip - left_eyebrow
        signals['forward'] = max(0, (eyebrow_raise - 0.1) * 10)
        
        # Eye squint (concentration = stop)
        left_eye_height = abs(landmarks[159].y - landmarks[145].y)
        signals['stop'] = left_eye_height < 0.01
        
        # Gaze direction (look left/right to turn)
        left_iris = landmarks[468].x
        right_iris = landmarks[473].x
        eye_center = (left_iris + right_iris) / 2
        signals['turn'] = (eye_center - 0.5) * 2
        
        # Blink detection for emergency stop
        signals['blink'] = left_eye_height < 0.005
        
        return signals
    
    def run_cursor_demo(self):
        """Control a cursor with your face!"""
        print("WEBCAM BRAIN CONTROL")
        print("- Raise eyebrows: Move forward")
        print("- Look left/right: Turn")
        print("- Squint: Slow down")
        print("- Blink twice: Reset")
        
        cv2.namedWindow('Brain Control', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                signals = self.detect_control_signals(landmarks)
                
                # Update cursor based on "brain" signals
                if not signals['stop']:
                    self.cursor_x += signals['turn'] * 5
                    self.cursor_y -= signals['forward'] * 3
                
                # Keep in bounds
                self.cursor_x = np.clip(self.cursor_x, 0, 640)
                self.cursor_y = np.clip(self.cursor_y, 0, 480)
                
                # Check for double blink (reset)
                self.blink_buffer.append(signals['blink'])
                if self.detect_double_blink():
                    self.cursor_x, self.cursor_y = 320, 240
                
                # Draw face mesh
                for idx in [70, 1, 159, 145, 468, 473]:  # Key points
                    x = int(landmarks[idx].x * frame.shape[1])
                    y = int(landmarks[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw control indicators
                cv2.putText(frame, f"Forward: {signals['forward']:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Turn: {signals['turn']:+.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if signals['stop']:
                    cv2.putText(frame, "STOP", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw cursor
            cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), 
                      15, (255, 0, 255), -1)
            
            # Show frame
            cv2.imshow('Brain Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def detect_double_blink(self):
        """Detect two blinks in quick succession"""
        if len(self.blink_buffer) < 30:
            return False
        
        # Look for pattern: open -> closed -> open -> closed -> open
        blinks = list(self.blink_buffer)
        blink_count = 0
        for i in range(1, len(blinks)):
            if not blinks[i-1] and blinks[i]:  # Rising edge
                blink_count += 1
        
        return blink_count >= 2


class EMGFromKeyboard:
    """
    Simulate EMG (muscle) signals using keyboard pressure
    Harder you press = stronger signal!
    """
    
    def __init__(self):
        self.muscle_signals = {
            'left': 0,
            'right': 0,
            'forward': 0,
            'back': 0
        }
        self.press_times = {}
    
    def get_muscle_signal(self, key):
        """Simulate EMG based on how long key is held"""
        import time
        
        if key not in self.press_times:
            self.press_times[key] = time.time()
            return 0.1
        
        # Longer press = stronger muscle signal
        hold_time = time.time() - self.press_times[key]
        signal = min(1.0, hold_time * 0.5)
        
        # Add realistic EMG noise
        noise = np.random.normal(0, 0.05)
        return np.clip(signal + noise, 0, 1)


class AudioBrainwaves:
    """
    Use microphone to detect 'brainwaves' through humming/sounds
    Different pitches = different commands
    """
    
    def __init__(self):
        import pyaudio
        self.CHUNK = 1024
        self.RATE = 44100
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
    
    def get_brain_frequency(self):
        """Detect pitch of humming as 'brainwave'"""
        data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.int16)
        
        # Simple FFT to find dominant frequency
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(fft), 1/self.RATE)
        
        # Find peak frequency in human voice range
        idx = np.where((freqs > 80) & (freqs < 400))[0]
        if len(idx) > 0:
            peak_idx = idx[np.argmax(np.abs(fft[idx]))]
            peak_freq = freqs[peak_idx]
            
            # Map frequency to control
            # Low hum (100-150 Hz) = left
            # Medium (150-200 Hz) = forward  
            # High (200-250 Hz) = right
            
            if 100 < peak_freq < 150:
                return 'left', np.abs(fft[peak_idx]) / 10000
            elif 150 < peak_freq < 200:
                return 'forward', np.abs(fft[peak_idx]) / 10000
            elif 200 < peak_freq < 250:
                return 'right', np.abs(fft[peak_idx]) / 10000
        
        return 'none', 0


# Combine everything into a fun demo
if __name__ == "__main__":
    print("Choose your 'Brain' Interface:")
    print("1. Webcam face control")
    print("2. Keyboard muscle signals")
    print("3. Microphone brainwaves")
    
    choice = input("Enter 1-3: ")
    
    if choice == '1':
        controller = WebcamBrainController()
        controller.run_cursor_demo()
    
    elif choice == '2':
        print("Hold WASD keys - longer press = stronger signal!")
        print("This simulates EMG muscle sensors")
        # Add pygame keyboard demo here
    
    elif choice == '3':
        print("Hum at different pitches:")
        print("Low hum = left, Medium = forward, High = right")
        # Add audio control demo here
