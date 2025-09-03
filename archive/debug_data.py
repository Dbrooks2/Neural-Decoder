"""
Debug script to check if our synthetic data has learnable patterns
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.neural_decoder.advanced_data_generation import AdvancedNeuralDataGenerator, NeuralDataConfig, NeuralSignalType

# Test with VERY strong artificial patterns
config = NeuralDataConfig(
    signal_type=NeuralSignalType.EEG,
    num_channels=16,
    sampling_rate=256,
    num_classes=4,
    trial_length=1.0,
    noise_level=0.01,  # Almost no noise
    spatial_correlation=0.8,  # Strong spatial patterns
    temporal_correlation=0.9   # Strong temporal patterns
)

generator = AdvancedNeuralDataGenerator(config)
data, labels = generator.generate_dataset(num_trials=100)

print(f"Data shape: {data.shape}")
print(f"Label distribution: {np.unique(labels, return_counts=True)}")

# Check if classes are actually different
print("\nClass differences:")
for class_id in range(4):
    class_data = data[labels == class_id]
    class_mean = np.mean(class_data)
    class_std = np.std(class_data)
    print(f"Class {class_id}: mean={class_mean:.4f}, std={class_std:.4f}")

# Plot class averages
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    class_data = data[labels == i]
    if len(class_data) > 0:
        # Average across trials and channels
        avg_signal = np.mean(class_data, axis=(0, 1))
        time = np.arange(len(avg_signal)) / config.sampling_rate
        ax.plot(time, avg_signal, linewidth=2)
        ax.set_title(f'Class {i} Average Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('class_differences.png', dpi=150)
plt.show()

# Check signal-to-noise ratio
signal_power = np.var(data, axis=2)  # Variance across time
noise_estimate = np.var(np.diff(data, axis=2), axis=2)  # High-freq noise
snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
print(f"\nSNR statistics:")
print(f"Mean SNR: {np.mean(snr):.2f} dB")
print(f"SNR range: {np.min(snr):.2f} to {np.max(snr):.2f} dB")

# Test with completely artificial patterns
print("\n" + "="*50)
print("CREATING OBVIOUSLY DIFFERENT PATTERNS")
print("="*50)

artificial_data = np.zeros((100, 16, 256))
artificial_labels = np.repeat([0, 1, 2, 3], 25)

for i, label in enumerate(artificial_labels):
    if label == 0:  # Left - low frequency sine wave
        t = np.linspace(0, 1, 256)
        artificial_data[i] = np.sin(2 * np.pi * 5 * t)[None, :] + 0.1 * np.random.randn(16, 256)
    elif label == 1:  # Right - high frequency sine wave  
        t = np.linspace(0, 1, 256)
        artificial_data[i] = np.sin(2 * np.pi * 20 * t)[None, :] + 0.1 * np.random.randn(16, 256)
    elif label == 2:  # Up - square wave
        t = np.linspace(0, 1, 256)
        artificial_data[i] = np.sign(np.sin(2 * np.pi * 3 * t))[None, :] + 0.1 * np.random.randn(16, 256)
    elif label == 3:  # Down - ramp wave
        t = np.linspace(0, 1, 256)
        artificial_data[i] = (t % 0.2) * 10 - 1 + 0.1 * np.random.randn(16, 256)

print("Artificial data created with VERY obvious patterns")
print("If the model can't learn this, there's a bug in the training code")

# Save for testing
np.save('artificial_test_data.npy', artificial_data)
np.save('artificial_test_labels.npy', artificial_labels)
print("Saved artificial_test_data.npy and artificial_test_labels.npy")
