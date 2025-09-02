"""
Neural Decoder Demo with Realistic EEG Data
Uses real neuroscience-based motor imagery patterns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from realistic_eeg_data import generate_realistic_motor_imagery_eeg
from sklearn.metrics import confusion_matrix
import time


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG classification
    Based on Lawhern et al. 2018 - specifically designed for EEG
    """
    def __init__(self, n_channels=22, n_classes=4, samples=1000):
        super().__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temporal_bn = nn.BatchNorm2d(16)
        
        # Spatial convolution (depthwise)
        self.spatial_conv = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.spatial_bn = nn.BatchNorm2d(32)
        
        # Separable convolution
        self.separable_conv = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.separable_pointwise = nn.Conv2d(32, 32, (1, 1), bias=False)
        self.separable_bn = nn.BatchNorm2d(32)
        
        # Pooling
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
        
        # Calculate final size
        self.final_size = self._get_final_size(n_channels, samples)
        
        # Classifier
        self.classifier = nn.Linear(self.final_size, n_classes)
        
    def _get_final_size(self, n_channels, samples):
        """Calculate the size after all convolutions and pooling"""
        x = torch.zeros(1, 1, n_channels, samples)
        x = self.pool1(self.temporal_conv(x))
        x = self.pool2(self.spatial_conv(x))
        x = self.separable_pointwise(self.separable_conv(x))
        return x.numel()
    
    def forward(self, x):
        # Input: [batch, channels, samples]
        # Reshape for 2D conv: [batch, 1, channels, samples]
        x = x.unsqueeze(1)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Separable convolution
        x = self.separable_conv(x)
        x = self.separable_pointwise(x)
        x = self.separable_bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


def realistic_demo():
    """Run demo with realistic EEG data"""
    
    print("=" * 60)
    print("REALISTIC NEURAL DECODER DEMO")
    print("Using neuroscience-based motor imagery EEG")
    print("=" * 60)
    
    Path("realistic_outputs").mkdir(exist_ok=True)
    
    # ========================================
    # 1. GENERATE REALISTIC EEG DATA
    # ========================================
    print("\n1. Generating Realistic Motor Imagery EEG...")
    print("-" * 40)
    
    # Generate realistic motor imagery data
    print("Creating EEG with realistic:")
    print("- 1/f noise characteristics")
    print("- ERD patterns in mu/beta bands")
    print("- Spatial correlation")
    print("- Occasional artifacts")
    
    data, labels = generate_realistic_motor_imagery_eeg(
        n_trials=400,  # More data for better training
        n_channels=22,  # Standard EEG montage
        fs=250,         # Standard sampling rate
        trial_length=4.0  # 4 second trials
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Classes: 0=Rest, 1=Left Hand, 2=Right Hand, 3=Both Hands")
    print(f"Labels: {np.unique(labels, return_counts=True)}")
    
    # ========================================
    # 2. DATA ANALYSIS
    # ========================================
    print("\n2. Analyzing EEG Data Quality...")
    print("-" * 40)
    
    from scipy import signal
    
    # Analyze frequency content
    fs = 250
    print("\nFrequency analysis:")
    
    for class_id in range(4):
        class_data = data[labels == class_id]
        motor_channels = [8, 9]  # Motor cortex channels
        
        # Calculate average PSD for motor channels
        all_psds = []
        for trial in class_data[:5]:
            for ch in motor_channels:
                freqs, psd = signal.welch(trial[ch], fs=fs, nperseg=fs)
                all_psds.append(psd)
        
        avg_psd = np.mean(all_psds, axis=0)
        
        # Power in different bands
        mu_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 18) & (freqs <= 26)
        
        mu_power = np.mean(avg_psd[mu_mask])
        beta_power = np.mean(avg_psd[beta_mask])
        
        class_names = ['Rest', 'Left', 'Right', 'Both']
        print(f"{class_names[class_id]:5}: Mu={mu_power:.1f}, Beta={beta_power:.1f}")
    
    # ========================================
    # 3. TRAIN EEGNET MODEL
    # ========================================
    print("\n3. Training EEGNet (State-of-Art EEG Classifier)...")
    print("-" * 40)
    
    # Split data
    n_train = 320  # 80% for training
    
    # Convert to torch tensors
    train_data = torch.FloatTensor(data[:n_train])
    train_labels = torch.LongTensor(labels[:n_train])
    test_data = torch.FloatTensor(data[n_train:])
    test_labels = torch.LongTensor(labels[n_train:])
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create model
    model = EEGNet(
        n_channels=data.shape[1],
        n_classes=4,
        samples=data.shape[2]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Data loaders
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    epochs = 80
    train_losses = []
    train_accs = []
    
    print("\nTraining EEGNet...")
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
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:5.1f}%, LR: {lr:.2e}")
        
        # Early stopping
        if accuracy > 85:
            print(f"Early stopping - achieved {accuracy:.1f}% accuracy!")
            break
    
    # ========================================
    # 4. EVALUATE MODEL
    # ========================================
    print("\n4. Evaluating on Test Set...")
    print("-" * 40)
    
    model.eval()
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = outputs.max(1)
        
        correct = predicted.eq(test_labels).sum().item()
        accuracy = 100. * correct / test_labels.size(0)
        
        # Get probabilities for confidence analysis
        probs = F.softmax(outputs, dim=1)
        confidences = torch.max(probs, 1)[0]
        
        cm = confusion_matrix(test_labels.cpu(), predicted.cpu())
    
    print(f"Test Accuracy: {accuracy:.1f}%")
    print(f"Average Confidence: {torch.mean(confidences):.3f}")
    print(f"Confidence Range: {torch.min(confidences):.3f} - {torch.max(confidences):.3f}")
    
    print("\nConfusion Matrix:")
    print("     Rest Left Right Both")
    for i, row in enumerate(cm):
        class_names = ['Rest', 'Left', 'Right', 'Both']
        print(f"{class_names[i]:4} {row}")
    
    # ========================================
    # 5. VISUALIZATIONS
    # ========================================
    print("\n5. Creating Visualizations...")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Sample EEG trials
    ax = axes[0, 0]
    class_names = ['Rest', 'Left Hand', 'Right Hand', 'Both Hands']
    colors = ['blue', 'red', 'green', 'orange']
    
    for class_id in range(4):
        class_trials = data[labels == class_id]
        if len(class_trials) > 0:
            # Average across trials for motor channel
            motor_ch = 8  # Left motor cortex
            avg_signal = np.mean(class_trials[:5, motor_ch, :], axis=0)
            t = np.arange(len(avg_signal)) / fs
            ax.plot(t, avg_signal + class_id*30, color=colors[class_id], 
                   label=class_names[class_id], linewidth=1.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV) + Offset')
    ax.set_title('Average EEG Patterns\n(Motor Cortex Channel)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training curves
    ax = axes[0, 1]
    ax.plot(train_losses, color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(train_accs, color='blue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Confusion matrix
    ax = axes[1, 0]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['Rest', 'Left', 'Right', 'Both'])
    ax.set_yticklabels(['Rest', 'Left', 'Right', 'Both'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.colorbar(im, ax=ax)
    
    # Frequency analysis
    ax = axes[1, 1]
    for class_id in range(4):
        class_data = data[labels == class_id]
        # Average PSD for motor channel
        all_psds = []
        for trial in class_data[:5]:
            freqs, psd = signal.welch(trial[8], fs=fs, nperseg=fs//2)
            all_psds.append(psd)
        avg_psd = np.mean(all_psds, axis=0)
        
        ax.semilogy(freqs[:50], avg_psd[:50], color=colors[class_id], 
                   label=class_names[class_id], linewidth=2)
    
    ax.axvspan(8, 12, alpha=0.2, color='gray', label='Mu band')
    ax.axvspan(18, 26, alpha=0.2, color='yellow', label='Beta band')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Frequency Content\n(Motor Channel)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confidence distribution
    ax = axes[1, 2]
    correct_mask = predicted.cpu() == test_labels.cpu()
    correct_conf = confidences[correct_mask].cpu().numpy()
    incorrect_conf = confidences[~correct_mask].cpu().numpy()
    
    ax.hist(correct_conf, bins=20, alpha=0.7, color='green', label='Correct')
    ax.hist(incorrect_conf, bins=20, alpha=0.7, color='red', label='Incorrect')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("realistic_outputs/realistic_eeg_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # 6. INFERENCE SPEED TEST
    # ========================================
    print("\n6. Testing Real-time Performance...")
    print("-" * 40)
    
    # Test inference speed
    test_input = torch.randn(1, data.shape[1], data.shape[2]).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    # Time inference
    num_inferences = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_inferences):
            output = model(test_input)
    
    inference_time = (time.time() - start_time) / num_inferences * 1000
    
    print(f"Average inference time: {inference_time:.2f} ms")
    print(f"Maximum real-time frequency: {1000/inference_time:.1f} Hz")
    
    if inference_time < 100:  # 100ms is reasonable for BCI
        print("✅ Suitable for real-time BCI applications!")
    else:
        print("⚠️  May be too slow for some real-time applications")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("REALISTIC DEMO COMPLETE!")
    print("=" * 60)
    
    print(f"\nResults Summary:")
    print(f"- Dataset: 400 realistic motor imagery EEG trials")
    print(f"- Model: EEGNet (state-of-art for EEG)")
    print(f"- Test Accuracy: {accuracy:.1f}%")
    print(f"- Inference Time: {inference_time:.1f} ms")
    print(f"- Data Quality: Realistic 1/f noise + ERD patterns")
    
    # Performance interpretation
    if accuracy >= 75:
        print("🎉 EXCELLENT performance! Research-grade results!")
    elif accuracy >= 60:
        print("😊 GOOD performance! Suitable for BCI applications!")
    elif accuracy >= 45:
        print("😐 FAIR performance! Better than random, needs tuning!")
    else:
        print("😞 POOR performance! Check data or model!")
    
    print(f"\nOutputs saved to: realistic_outputs/")
    
    return {
        'accuracy': accuracy,
        'inference_time_ms': inference_time,
        'model': model,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    try:
        results = realistic_demo()
        print("\n✅ Realistic demo completed successfully!")
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
