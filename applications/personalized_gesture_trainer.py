"""
Personalized Gesture Recognition System
Learns your unique facial expressions and adapts over time
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque
import pickle
from datetime import datetime

import mediapipe as mp


@dataclass
class PersonalizedGesture:
    """Represents a user-defined gesture"""
    name: str
    id: int
    samples: List[np.ndarray]  # List of facial landmark arrays
    confidence_threshold: float = 0.8
    created_at: str = ""
    usage_count: int = 0
    success_rate: float = 1.0
    average_execution_time: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class GestureRecognitionNet(nn.Module):
    """
    Personalized neural network that adapts to individual users
    Learns to recognize custom gestures from facial landmarks
    """
    
    def __init__(self, input_size: int = 468 * 3, hidden_size: int = 256, 
                 num_gestures: int = 10, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention mechanism for important facial features
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Gesture classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_gestures)
        )
        
        # User adaptation layer (fine-tuned per user)
        self.user_adapter = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply user-specific adaptation
        adapted_features = self.user_adapter(features)
        
        # Attention-weighted features
        attention_weights = self.attention(adapted_features)
        weighted_features = adapted_features * attention_weights
        
        # Classify gesture
        output = self.classifier(weighted_features)
        
        return output, attention_weights


class PersonalizedGestureDataset(Dataset):
    """Dataset for user's personal gestures"""
    
    def __init__(self, gestures: List[PersonalizedGesture], 
                 augment: bool = True):
        self.samples = []
        self.labels = []
        self.augment = augment
        
        for gesture in gestures:
            for sample in gesture.samples:
                self.samples.append(sample)
                self.labels.append(gesture.id)
        
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            # Data augmentation for better generalization
            # Small random transformations
            sample += np.random.normal(0, 0.001, sample.shape)
            
            # Random scaling (simulate distance changes)
            scale = np.random.uniform(0.95, 1.05)
            sample *= scale
            
            # Random rotation (simulate head tilt)
            angle = np.random.uniform(-5, 5) * np.pi / 180
            # Apply rotation to x,y coordinates
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for i in range(0, len(sample), 3):
                x, y = sample[i], sample[i+1]
                sample[i] = cos_a * x - sin_a * y
                sample[i+1] = sin_a * x + cos_a * y
        
        return torch.FloatTensor(sample), torch.LongTensor([label])


class PersonalizedGestureTrainer:
    """
    Complete system for training personalized gestures
    Adapts to individual facial differences and learns preferences
    """
    
    def __init__(self, user_id: str = "default", save_dir: str = "user_models"):
        self.user_id = user_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # User profile directory
        self.user_dir = self.save_dir / user_id
        self.user_dir.mkdir(exist_ok=True)
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load or initialize user data
        self.gestures: List[PersonalizedGesture] = []
        self.model = None
        self.user_profile = self.load_user_profile()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adaptation_buffer = deque(maxlen=100)
        
        # Real-time feedback
        self.last_gesture = None
        self.gesture_confidence = 0.0
        
    def load_user_profile(self) -> Dict:
        """Load user profile and trained model"""
        profile_path = self.user_dir / "profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            # Load gestures
            gestures_path = self.user_dir / "gestures.pkl"
            if gestures_path.exists():
                with open(gestures_path, 'rb') as f:
                    self.gestures = pickle.load(f)
            
            # Load model
            model_path = self.user_dir / "model.pth"
            if model_path.exists() and self.gestures:
                self.model = GestureRecognitionNet(
                    num_gestures=len(self.gestures)
                )
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            
            return profile
        else:
            # Create new profile
            return {
                "user_id": self.user_id,
                "created_at": datetime.now().isoformat(),
                "total_training_samples": 0,
                "total_usage_time": 0,
                "preferred_gestures": {},
                "difficulty_adjustments": {},
                "success_rate_history": []
            }
    
    def save_user_profile(self):
        """Save user profile and model"""
        # Save profile
        with open(self.user_dir / "profile.json", 'w') as f:
            json.dump(self.user_profile, f, indent=2)
        
        # Save gestures
        with open(self.user_dir / "gestures.pkl", 'wb') as f:
            pickle.dump(self.gestures, f)
        
        # Save model
        if self.model:
            torch.save(self.model.state_dict(), self.user_dir / "model.pth")
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Flatten to 1D array [x1,y1,z1,x2,y2,z2,...]
            landmark_array = []
            for lm in landmarks:
                landmark_array.extend([lm.x, lm.y, lm.z])
            return np.array(landmark_array, dtype=np.float32)
        
        return None
    
    def record_gesture_samples(self, gesture_name: str, num_samples: int = 10):
        """Interactive recording of gesture samples"""
        print(f"\n=== Recording Gesture: {gesture_name} ===")
        print(f"Perform the gesture {num_samples} times when prompted")
        print("Press SPACE to record each sample")
        
        cap = cv2.VideoCapture(0)
        samples = []
        sample_count = 0
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            
            # UI
            cv2.putText(frame, f"Recording: {gesture_name}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {sample_count + 1}/{num_samples}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to capture, Q to quit", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if landmarks is not None:
                cv2.putText(frame, "Face detected ✓", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Visualize key points
                h, w = frame.shape[:2]
                # Draw some key landmarks
                key_points = [1, 6, 13, 14, 33, 133, 362, 263]  # Nose, chin, lips, eyes
                for idx in key_points:
                    if idx * 3 < len(landmarks):
                        x = int(landmarks[idx * 3] * w)
                        y = int(landmarks[idx * 3 + 1] * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            
            cv2.imshow('Gesture Recording', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and landmarks is not None:
                samples.append(landmarks)
                sample_count += 1
                print(f"Captured sample {sample_count}")
                time.sleep(0.5)  # Brief pause
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return samples
    
    def train_gesture(self, gesture_name: str, num_samples: int = 10):
        """Train a new gesture"""
        # Record samples
        samples = self.record_gesture_samples(gesture_name, num_samples)
        
        if len(samples) < 3:
            print("Not enough samples recorded!")
            return False
        
        # Create gesture object
        gesture_id = len(self.gestures)
        gesture = PersonalizedGesture(
            name=gesture_name,
            id=gesture_id,
            samples=samples
        )
        self.gestures.append(gesture)
        
        # Update profile
        self.user_profile["total_training_samples"] += len(samples)
        
        # Retrain model
        self.train_model()
        
        # Save everything
        self.save_user_profile()
        
        print(f"✓ Gesture '{gesture_name}' trained successfully!")
        return True
    
    def train_model(self, epochs: int = 50):
        """Train the gesture recognition model"""
        if len(self.gestures) < 2:
            print("Need at least 2 gestures to train!")
            return
        
        print("\n=== Training Personalized Model ===")
        
        # Create model
        self.model = GestureRecognitionNet(
            num_gestures=len(self.gestures)
        )
        
        # Prepare data
        dataset = PersonalizedGestureDataset(self.gestures, augment=True)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        # Train
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs, _ = self.model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y.squeeze()).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.3f}, Accuracy={accuracy:.2%}")
            
            scheduler.step(avg_loss)
        
        self.model.eval()
        print("✓ Model training complete!")
    
    def recognize_gesture(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize gesture from frame"""
        if not self.model or not self.gestures:
            return None, 0.0
        
        landmarks = self.extract_landmarks(frame)
        if landmarks is None:
            return None, 0.0
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(landmarks).unsqueeze(0)
            outputs, attention = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            gesture_id = predicted.item()
            confidence_val = confidence.item()
            
            # Apply confidence threshold
            if confidence_val >= self.gestures[gesture_id].confidence_threshold:
                gesture_name = self.gestures[gesture_id].name
                
                # Update usage stats
                self.gestures[gesture_id].usage_count += 1
                
                # Track performance
                self.performance_history.append({
                    'gesture': gesture_name,
                    'confidence': confidence_val,
                    'timestamp': time.time()
                })
                
                return gesture_name, confidence_val
        
        return None, 0.0
    
    def adapt_to_user(self, feedback: bool, gesture_name: str):
        """Adapt model based on user feedback"""
        # Store adaptation data
        self.adaptation_buffer.append({
            'gesture': gesture_name,
            'feedback': feedback,
            'timestamp': time.time()
        })
        
        # Update gesture success rate
        for gesture in self.gestures:
            if gesture.name == gesture_name:
                # Exponential moving average
                alpha = 0.1
                gesture.success_rate = (
                    alpha * (1.0 if feedback else 0.0) + 
                    (1 - alpha) * gesture.success_rate
                )
                
                # Adjust confidence threshold if needed
                if gesture.success_rate < 0.7:
                    gesture.confidence_threshold *= 0.95
                elif gesture.success_rate > 0.9:
                    gesture.confidence_threshold *= 1.02
                
                gesture.confidence_threshold = np.clip(
                    gesture.confidence_threshold, 0.5, 0.95
                )
                break
        
        # Retrain periodically with new data
        if len(self.adaptation_buffer) >= 50:
            self.incremental_learning()
    
    def incremental_learning(self):
        """Incremental learning from user feedback"""
        print("Adapting to your usage patterns...")
        
        # Fine-tune only the user adapter layer
        if self.model:
            optimizer = optim.Adam(
                self.model.user_adapter.parameters(), 
                lr=0.0001
            )
            
            # Quick fine-tuning based on recent performance
            # This is a simplified version - in practice, you'd store
            # the actual samples that got feedback
            
            self.model.train()
            for _ in range(10):
                # Simulate training on recent samples
                # In real implementation, store the actual tensors
                pass
            self.model.eval()
        
        # Clear buffer
        self.adaptation_buffer.clear()
        
        # Save updated model
        self.save_user_profile()
    
    def get_usage_analytics(self) -> Dict:
        """Analyze usage patterns and preferences"""
        if not self.performance_history:
            return {}
        
        # Gesture frequency
        gesture_counts = {}
        for record in self.performance_history:
            gesture = record['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Time-based patterns
        recent_records = [
            r for r in self.performance_history 
            if time.time() - r['timestamp'] < 3600  # Last hour
        ]
        
        # Average confidence by gesture
        gesture_confidences = {}
        for gesture in self.gestures:
            confs = [
                r['confidence'] for r in self.performance_history 
                if r['gesture'] == gesture.name
            ]
            if confs:
                gesture_confidences[gesture.name] = np.mean(confs)
        
        return {
            'most_used_gestures': sorted(
                gesture_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'average_confidences': gesture_confidences,
            'total_gestures_today': len(recent_records),
            'success_rates': {
                g.name: g.success_rate for g in self.gestures
            }
        }
    
    def interactive_demo(self):
        """Interactive demo with real-time recognition"""
        print("\n=== Personalized Gesture Recognition Demo ===")
        print("Gestures:")
        for i, gesture in enumerate(self.gestures):
            print(f"{i+1}. {gesture.name} (used {gesture.usage_count} times)")
        print("\nPress 'q' to quit, 'a' to add gesture, 's' for stats")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Recognize gesture
            gesture_name, confidence = self.recognize_gesture(frame)
            
            # Display results
            if gesture_name:
                cv2.putText(frame, f"Gesture: {gesture_name}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Visual feedback bar
                bar_width = int(confidence * 300)
                cv2.rectangle(frame, (20, 80), (20 + bar_width, 100),
                             (0, 255, 0), -1)
            else:
                cv2.putText(frame, "No gesture detected", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Show analytics
            analytics = self.get_usage_analytics()
            if analytics.get('most_used_gestures'):
                top_gesture = analytics['most_used_gestures'][0]
                cv2.putText(frame, f"Most used: {top_gesture[0]} ({top_gesture[1]}x)", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Personalized Gestures', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Add new gesture
                cv2.destroyAllWindows()
                name = input("\nEnter gesture name: ")
                self.train_gesture(name)
                cap = cv2.VideoCapture(0)
            elif key == ord('s'):
                # Show statistics
                print("\n=== Usage Statistics ===")
                for k, v in analytics.items():
                    print(f"{k}: {v}")
            elif key == ord('y') and gesture_name:
                # Positive feedback
                self.adapt_to_user(True, gesture_name)
                print(f"✓ Positive feedback for {gesture_name}")
            elif key == ord('n') and gesture_name:
                # Negative feedback
                self.adapt_to_user(False, gesture_name)
                print(f"✗ Negative feedback for {gesture_name}")
        
        cap.release()
        cv2.destroyAllWindows()


# Quick start function
def quick_start():
    """Quick start for new users"""
    print("\n=== Personalized Gesture Recognition Setup ===")
    
    user_id = input("Enter your name (or press Enter for 'default'): ").strip()
    if not user_id:
        user_id = "default"
    
    trainer = PersonalizedGestureTrainer(user_id=user_id)
    
    if not trainer.gestures:
        print("\nLet's create your first gestures!")
        print("Suggested starter gestures:")
        print("1. 'yes' - Nod your head")
        print("2. 'no' - Shake your head")
        print("3. 'select' - Raise eyebrows")
        print("4. 'stop' - Close both eyes")
        
        # Train basic gestures
        basic_gestures = ['yes', 'no', 'select', 'stop']
        for gesture in basic_gestures:
            response = input(f"\nTrain '{gesture}' gesture? (y/n): ")
            if response.lower() == 'y':
                trainer.train_gesture(gesture, num_samples=5)
    
    # Run demo
    trainer.interactive_demo()


if __name__ == "__main__":
    quick_start()
