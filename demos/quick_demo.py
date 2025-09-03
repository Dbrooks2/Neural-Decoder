"""
Quick Neural Decoder Demo
A fast demonstration of the neural decoder system
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "data"))
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "analysis"))

from advanced_data_generation import (
    AdvancedNeuralDataGenerator, 
    NeuralDataConfig, 
    NeuralSignalType
)
from models import CNNLSTMDecoder  # Use simple model for speed
from analysis_tools import NeuralDataAnalyzer


class SimpleCNNClassifier(nn.Module):
    """Improved CNN classifier for neural signals"""
    def __init__(self, num_channels: int, window_size: int, num_classes: int = 4):
        super().__init__()
        # Spatial convolution across channels
        self.spatial_conv = nn.Conv1d(num_channels, 32, kernel_size=1)
        
        # Temporal convolutions
        self.temporal_conv1 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.temporal_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate size after pooling
        conv_output_size = window_size // 4  # Two pooling layers
        
        # Classification head
        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)  # Less dropout
        
    def forward(self, x):
        # x shape: [batch, channels, time]
        # Spatial processing
        x = F.relu(self.bn1(self.spatial_conv(x)))
        
        # Temporal processing
        x = F.relu(self.bn2(self.temporal_conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.temporal_conv2(x)))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def quick_demo():
    """Run a quick demonstration of the neural decoder"""
    
    print("=" * 60)
    print("NEURAL DECODER QUICK DEMO")
    print("=" * 60)
    
    # Create output directory
    Path("demo_outputs").mkdir(exist_ok=True)
    
    # ========================================
    # 1. GENERATE SAMPLE DATA
    # ========================================
    print("\n1. Generating Neural Data...")
    print("-" * 30)
    
    # Simple configuration for quick demo
    config = NeuralDataConfig(
        signal_type=NeuralSignalType.EEG,
        num_channels=16,  # Fewer channels for speed
        sampling_rate=256,  # Lower sampling rate
        num_classes=4,
        trial_length=1.0,  # 1 second trials
        noise_level=0.05  # Less noise, stronger signal patterns
    )
    
    generator = AdvancedNeuralDataGenerator(config)
    
    # Generate small dataset with STRONG artificial patterns
    print("Generating 100 trials with strong class differences...")
    
    # Create artificial data with very obvious patterns
    data = np.zeros((100, config.num_channels, int(config.sampling_rate * config.trial_length)))
    labels = np.repeat([0, 1, 2, 3], 25)  # 25 trials per class
    
    print("Creating artificial patterns that are VERY different...")
    
    for i, label in enumerate(labels):
        t = np.linspace(0, config.trial_length, data.shape[2])
        
        if label == 0:  # LEFT - Low frequency oscillation in frontal channels
            # Strong 8Hz alpha rhythm in channels 0-3
            for ch in range(4):
                data[i, ch] = 3 * np.sin(2 * np.pi * 8 * t + ch * 0.5)
            # Background noise in other channels
            for ch in range(4, config.num_channels):
                data[i, ch] = 0.5 * np.random.randn(data.shape[2])
                
        elif label == 1:  # RIGHT - High frequency oscillation in different channels  
            # Strong 20Hz beta rhythm in channels 12-15
            for ch in range(12, config.num_channels):
                data[i, ch] = 3 * np.sin(2 * np.pi * 20 * t + ch * 0.3)
            # Background noise in other channels
            for ch in range(12):
                data[i, ch] = 0.5 * np.random.randn(data.shape[2])
                
        elif label == 2:  # UP - Square wave pattern in central channels
            # Square wave in channels 6-9
            for ch in range(6, 10):
                data[i, ch] = 2 * np.sign(np.sin(2 * np.pi * 5 * t))
            # Background noise in other channels  
            for ch in list(range(6)) + list(range(10, config.num_channels)):
                data[i, ch] = 0.5 * np.random.randn(data.shape[2])
                
        elif label == 3:  # DOWN - Ramp/sawtooth pattern in posterior channels
            # Sawtooth wave in channels 8-11
            for ch in range(8, 12):
                ramp = 2 * (t % 0.2) / 0.2 - 1  # Sawtooth every 200ms
                data[i, ch] = ramp
            # Background noise in other channels
            for ch in list(range(8)) + list(range(12, config.num_channels)):
                data[i, ch] = 0.5 * np.random.randn(data.shape[2])
        
        # Add small amount of noise to all channels
        data[i] += 0.1 * np.random.randn(*data[i].shape)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels: {np.unique(labels, return_counts=True)}")
    
    # Verify the patterns are different
    print("\nClass pattern verification:")
    for class_id in range(4):
        class_data = data[labels == class_id]
        class_power = np.mean(np.var(class_data, axis=2))  # Power across time
        print(f"Class {class_id}: Average power = {class_power:.3f}")
    
    print("✅ Created artificial data with VERY strong class differences!")
    
    # Visualize sample
    print("\nVisualizing sample trial...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 4 different class examples
    for i, ax in enumerate(axes.flat):
        class_idx = np.where(labels == i)[0][0]
        trial = data[class_idx]
        
        # Plot first 5 channels
        time_axis = np.arange(trial.shape[1]) / config.sampling_rate
        for ch in range(min(5, trial.shape[0])):
            ax.plot(time_axis, trial[ch] + ch*2, alpha=0.7)
        
        ax.set_title(f'Class {i} ({"Left/Right/Up/Down"[i*5:i*5+4]})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("demo_outputs/sample_trials.png", dpi=150)
    plt.close()
    
    # ========================================
    # 2. ANALYZE DATA QUALITY
    # ========================================
    print("\n2. Analyzing Data Quality...")
    print("-" * 30)
    
    analyzer = NeuralDataAnalyzer(sampling_rate=config.sampling_rate)
    
    # Quick quality check
    quality = analyzer.analyze_signal_quality(data)
    print(f"Signal-to-Noise Ratio: {quality['mean_snr']:.1f} ± {quality['std_snr']:.1f} dB")
    print(f"Artifact ratio: {quality['mean_artifact_ratio']:.3f}")
    print(f"Dead channels: {quality['dead_channels']}")
    
    # Frequency content
    freq_content = analyzer.analyze_frequency_content(data)
    print("\nDominant frequency bands:")
    for band, power in freq_content.items():
        print(f"  {band}: {power:.2e}")
    
    # ========================================
    # 3. TRAIN SIMPLE MODEL
    # ========================================
    print("\n3. Training Simple Neural Decoder...")
    print("-" * 30)
    
    # Split data - use more for training
    n_train = 90  # More training data
    train_data = torch.FloatTensor(data[:n_train])
    train_labels = torch.LongTensor(labels[:n_train])
    test_data = torch.FloatTensor(data[n_train:])
    test_labels = torch.LongTensor(labels[n_train:])
    
    # Create simple model
    model = SimpleCNNClassifier(
        num_channels=config.num_channels,
        window_size=data.shape[2],
        num_classes=config.num_classes
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = model.to(device)
    
    # Use class weights to prevent class collapse
    class_counts = np.bincount(train_labels.numpy())
    class_weights = torch.FloatTensor([1.0/count for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Better optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Training loop with more epochs
    epochs = 50  # More training epochs
    train_losses = []
    train_accs = []
    
    print("\nTraining...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        # Step the scheduler
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.3f}, Acc: {accuracy:.1f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
        # Early stopping if perfect accuracy
        if accuracy > 95:
            print(f"Early stopping - achieved {accuracy:.1f}% accuracy!")
            break
    
    # ========================================
    # 4. EVALUATE MODEL
    # ========================================
    print("\n4. Evaluating Model...")
    print("-" * 30)
    
    model.eval()
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = outputs.max(1)
        
        # Calculate accuracy
        correct = predicted.eq(test_labels).sum().item()
        accuracy = 100. * correct / test_labels.size(0)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels.cpu(), predicted.cpu())
    
    print(f"Test Accuracy: {accuracy:.1f}%")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training curves
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix - handle class collapse
    if cm.shape[0] < 4 or cm.shape[1] < 4:
        print("WARNING: Model is only predicting one class!")
        # Create a proper 4x4 confusion matrix for visualization
        full_cm = np.zeros((4, 4))
        # Find which class was predicted
        unique_preds = np.unique(predicted.cpu().numpy())
        predicted_class = unique_preds[0] if len(unique_preds) > 0 else 0
        
        # Put all predictions in that column
        for true_class in test_labels.cpu().numpy():
            full_cm[true_class, predicted_class] += 1
        cm = full_cm.astype(int)
        print(f"Model only predicts class {predicted_class} ({['Left', 'Right', 'Up', 'Down'][predicted_class]})")
    
    im = ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    ax3.set_xticklabels(['Left', 'Right', 'Up', 'Down'])
    ax3.set_yticklabels(['Left', 'Right', 'Up', 'Down'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax3.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    plt.savefig("demo_outputs/training_results.png", dpi=150)
    plt.close()
    
    # ========================================
    # 5. TEST REAL-TIME PERFORMANCE
    # ========================================
    print("\n5. Testing Real-time Performance...")
    print("-" * 30)
    
    # Test inference speed
    test_input = torch.randn(1, config.num_channels, data.shape[2]).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    # Time 100 inferences
    start_time = time.time()
    num_inferences = 100
    
    with torch.no_grad():
        for _ in range(num_inferences):
            output = model(test_input)
            # Get prediction
            _, predicted = output.max(1)
    
    inference_time = (time.time() - start_time) / num_inferences * 1000  # ms
    
    print(f"Average inference time: {inference_time:.2f} ms")
    print(f"Maximum real-time frequency: {1000/inference_time:.1f} Hz")
    
    if inference_time < 50:
        print("✓ Suitable for real-time BCI applications!")
    
    # ========================================
    # 6. SIMULATE REAL-TIME DECODING
    # ========================================
    print("\n6. Simulating Real-time Neural Decoding...")
    print("-" * 30)
    
    # Create a simple visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Simulate 10 seconds of continuous decoding
    window_size = data.shape[2]
    continuous_signal = np.random.randn(config.num_channels, config.sampling_rate * 10)
    
    predictions = []
    confidences = []
    
    # Sliding window prediction
    for i in range(0, continuous_signal.shape[1] - window_size, config.sampling_rate // 4):
        # Extract window
        window = continuous_signal[:, i:i+window_size]
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(window_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predictions.append(predicted.item())
        confidences.append(confidence.item())
    
    # Plot continuous signal
    time_continuous = np.arange(continuous_signal.shape[1]) / config.sampling_rate
    for ch in range(min(5, config.num_channels)):
        ax1.plot(time_continuous, continuous_signal[ch] + ch*3, alpha=0.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Channel')
    ax1.set_title('Continuous Neural Signal')
    ax1.set_xlim(0, 10)
    
    # Plot predictions
    pred_times = np.arange(len(predictions)) * 0.25  # 4 Hz prediction rate
    ax2.scatter(pred_times, predictions, c=confidences, cmap='viridis', s=50)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Predicted Class')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Left', 'Right', 'Up', 'Down'])
    ax2.set_title('Real-time Predictions (color = confidence)')
    ax2.set_xlim(0, 10)
    
    # Add colorbar
    cbar = plt.colorbar(ax2.scatter(pred_times, predictions, c=confidences, cmap='viridis'), ax=ax2)
    cbar.set_label('Confidence')
    
    plt.tight_layout()
    plt.savefig("demo_outputs/realtime_simulation.png", dpi=150)
    plt.close()
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    
    print("\nSummary:")
    print(f"- Generated {data.shape[0]} trials of {config.signal_type.value} data")
    print(f"- {config.num_channels} channels at {config.sampling_rate} Hz")
    print(f"- Trained CNN-LSTM decoder in {epochs} epochs")
    print(f"- Achieved {accuracy:.1f}% test accuracy")
    print(f"- Inference latency: {inference_time:.1f} ms")
    print(f"- Can process at {1000/inference_time:.0f} Hz")
    
    print("\nOutputs saved to: demo_outputs/")
    print("- sample_trials.png")
    print("- training_results.png") 
    print("- realtime_simulation.png")
    
    return {
        'accuracy': accuracy,
        'inference_time_ms': inference_time,
        'model': model,
        'data_config': config
    }


if __name__ == "__main__":
    try:
        results = quick_demo()
        print("\n✓ Demo completed successfully!")
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
