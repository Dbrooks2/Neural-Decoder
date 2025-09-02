"""
Generate realistic EEG data based on actual neuroscience principles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def generate_realistic_motor_imagery_eeg(n_trials=100, n_channels=22, fs=250, trial_length=4.0):
    """
    Generate realistic motor imagery EEG based on actual BCI research
    
    Motor imagery creates:
    - ERD (Event-Related Desynchronization) in mu rhythm (8-12 Hz)
    - ERD in beta rhythm (18-26 Hz) 
    - Different spatial patterns for left vs right hand
    """
    
    n_samples = int(trial_length * fs)
    data = np.zeros((n_trials, n_channels, n_samples))
    labels = np.zeros(n_trials, dtype=int)
    
    # Standard 10-20 electrode positions (simplified)
    # Channels 7-10: Motor cortex area (C3, Cz, C4 region)
    motor_channels_left = [7, 8]    # Left motor cortex (right hand imagery)
    motor_channels_right = [9, 10]  # Right motor cortex (left hand imagery) 
    
    for trial in range(n_trials):
        # Assign class (0=rest, 1=left hand, 2=right hand, 3=both hands)
        label = trial % 4
        labels[trial] = label
        
        # Generate base EEG with realistic 1/f noise
        for ch in range(n_channels):
            # Base EEG: 1/f noise + alpha rhythm
            base_eeg = generate_realistic_background_eeg(n_samples, fs)
            data[trial, ch] = base_eeg
        
        # Add task-specific patterns
        if label == 1:  # Left hand imagery
            # ERD in right motor cortex (channels 9-10)
            for ch in motor_channels_right:
                # Reduce mu (8-12 Hz) and beta (18-26 Hz) power
                data[trial, ch] = add_erd_pattern(data[trial, ch], fs, 
                                                 mu_suppression=0.6, beta_suppression=0.4)
                
        elif label == 2:  # Right hand imagery  
            # ERD in left motor cortex (channels 7-8)
            for ch in motor_channels_left:
                data[trial, ch] = add_erd_pattern(data[trial, ch], fs,
                                                 mu_suppression=0.6, beta_suppression=0.4)
                
        elif label == 3:  # Both hands
            # ERD in both motor areas
            for ch in motor_channels_left + motor_channels_right:
                data[trial, ch] = add_erd_pattern(data[trial, ch], fs,
                                                 mu_suppression=0.4, beta_suppression=0.3)
        
        # Add some cross-talk between nearby channels
        data[trial] = add_spatial_correlation(data[trial])
        
        # Add realistic artifacts occasionally
        if np.random.random() < 0.1:  # 10% of trials have artifacts
            data[trial] = add_eeg_artifacts(data[trial], fs)
    
    return data, labels

def generate_realistic_background_eeg(n_samples, fs):
    """Generate realistic background EEG with proper spectral characteristics"""
    
    # Generate 1/f noise (pink noise)
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    freqs[0] = 1e-10  # Avoid division by zero
    
    # 1/f spectrum
    spectrum = 1 / np.abs(freqs)
    spectrum[0] = spectrum[1]  # Fix DC component
    
    # Add random phases
    phases = np.random.uniform(0, 2*np.pi, len(freqs))
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # Convert back to time domain
    eeg = np.real(np.fft.ifft(complex_spectrum))
    
    # Add dominant alpha rhythm (8-12 Hz) - typical in resting EEG
    t = np.arange(n_samples) / fs
    alpha_freq = np.random.uniform(8, 12)
    alpha_amplitude = np.random.uniform(10, 30)  # microvolts
    alpha_phase = np.random.uniform(0, 2*np.pi)
    
    alpha_wave = alpha_amplitude * np.sin(2*np.pi*alpha_freq*t + alpha_phase)
    
    # Amplitude modulation (alpha blocking/enhancement)
    alpha_envelope = 1 + 0.3 * np.sin(2*np.pi*0.1*t)  # Slow modulation
    alpha_wave *= alpha_envelope
    
    eeg += alpha_wave
    
    # Scale to realistic EEG amplitude range
    eeg = eeg * 20 / np.std(eeg)  # ~20 microvolts std
    
    return eeg

def add_erd_pattern(eeg_signal, fs, mu_suppression=0.5, beta_suppression=0.3):
    """Add Event-Related Desynchronization pattern"""
    
    # Create filters for mu (8-12 Hz) and beta (18-26 Hz) bands
    nyquist = fs / 2
    
    # Mu band filter
    mu_low, mu_high = 8/nyquist, 12/nyquist
    b_mu, a_mu = signal.butter(4, [mu_low, mu_high], btype='band')
    mu_component = signal.filtfilt(b_mu, a_mu, eeg_signal)
    
    # Beta band filter  
    beta_low, beta_high = 18/nyquist, 26/nyquist
    b_beta, a_beta = signal.butter(4, [beta_low, beta_high], btype='band')
    beta_component = signal.filtfilt(b_beta, a_beta, eeg_signal)
    
    # Apply ERD (reduce power in these bands)
    mu_component *= (1 - mu_suppression)
    beta_component *= (1 - beta_suppression)
    
    # Reconstruct signal
    # Remove original mu/beta and add suppressed versions
    other_component = eeg_signal - signal.filtfilt(b_mu, a_mu, eeg_signal) - signal.filtfilt(b_beta, a_beta, eeg_signal)
    
    return other_component + mu_component + beta_component

def add_spatial_correlation(trial_data):
    """Add realistic spatial correlation between nearby electrodes"""
    
    n_channels = trial_data.shape[0]
    
    # Simple spatial mixing - nearby channels influence each other
    mixed_data = trial_data.copy()
    
    for ch in range(1, n_channels-1):
        # Mix with neighbors
        mixed_data[ch] = (0.7 * trial_data[ch] + 
                         0.15 * trial_data[ch-1] + 
                         0.15 * trial_data[ch+1])
    
    return mixed_data

def add_eeg_artifacts(trial_data, fs):
    """Add realistic EEG artifacts"""
    
    n_channels, n_samples = trial_data.shape
    
    # Eye blink artifact (affects frontal channels)
    if np.random.random() < 0.5:
        blink_time = np.random.randint(n_samples//4, 3*n_samples//4)
        blink_duration = int(0.3 * fs)  # 300ms blink
        
        # Blink waveform
        t_blink = np.arange(blink_duration) / fs
        blink_wave = 100 * np.exp(-t_blink/0.1) * np.sin(2*np.pi*3*t_blink)
        
        # Add to frontal channels (0-3)
        for ch in range(min(4, n_channels)):
            if blink_time + blink_duration < n_samples:
                trial_data[ch, blink_time:blink_time+blink_duration] += blink_wave
    
    # Muscle artifact (random channels)
    if np.random.random() < 0.3:
        artifact_ch = np.random.randint(0, n_channels)
        artifact_start = np.random.randint(0, n_samples//2)
        artifact_duration = int(1.0 * fs)  # 1 second
        
        # High frequency muscle noise
        muscle_noise = 50 * np.random.randn(artifact_duration)
        # Filter to high frequencies (30-100 Hz)
        b, a = signal.butter(4, [30/(fs/2), 100/(fs/2)], btype='band')
        muscle_noise = signal.filtfilt(b, a, muscle_noise)
        
        if artifact_start + artifact_duration < n_samples:
            trial_data[artifact_ch, artifact_start:artifact_start+artifact_duration] += muscle_noise
    
    return trial_data

# Test the realistic data
if __name__ == "__main__":
    print("Generating realistic motor imagery EEG data...")
    
    data, labels = generate_realistic_motor_imagery_eeg(n_trials=200, n_channels=22)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels: {np.unique(labels, return_counts=True)}")
    
    # Analyze the data
    print("\nData quality analysis:")
    
    # Check power in different frequency bands
    fs = 250
    for class_id in range(4):
        class_data = data[labels == class_id]
        
        # Calculate average power spectral density
        avg_psd = []
        for trial in class_data[:5]:  # Sample 5 trials
            for ch in [8, 9]:  # Motor channels
                freqs, psd = signal.welch(trial[ch], fs=fs, nperseg=fs)
                avg_psd.append(psd)
        
        avg_psd = np.mean(avg_psd, axis=0)
        
        # Power in mu band (8-12 Hz)
        mu_mask = (freqs >= 8) & (freqs <= 12)
        mu_power = np.mean(avg_psd[mu_mask])
        
        print(f"Class {class_id}: Mu power = {mu_power:.2f}")
    
    # Visualize sample trials
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    class_names = ['Rest', 'Left Hand', 'Right Hand', 'Both Hands']
    
    for i, ax in enumerate(axes.flat):
        class_trials = data[labels == i]
        if len(class_trials) > 0:
            # Plot motor channels for first trial
            trial = class_trials[0]
            t = np.arange(trial.shape[1]) / fs
            
            # Plot motor channels (7-10)
            for ch in range(7, min(11, trial.shape[0])):
                ax.plot(t, trial[ch] + ch*50, alpha=0.7, linewidth=0.8)
            
            ax.set_title(f'{class_names[i]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channel + Offset (μV)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_eeg_samples.png', dpi=150)
    plt.show()
    
    print("\n✅ Realistic EEG data generated!")
    print("This data has proper:")
    print("- 1/f noise characteristics")
    print("- ERD patterns in motor imagery")
    print("- Spatial correlation")
    print("- Realistic artifacts")
    print("- Proper frequency content")
