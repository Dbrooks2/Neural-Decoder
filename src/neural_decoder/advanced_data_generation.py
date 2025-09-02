"""
Advanced Neural Data Generation
Simulates various types of neural signals with realistic properties
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import scipy.signal as signal
import scipy.stats as stats
from enum import Enum


class NeuralSignalType(Enum):
    """Types of neural signals we can simulate"""
    EEG = "eeg"  # Scalp recordings
    ECOG = "ecog"  # Electrocorticography
    LFP = "lfp"  # Local field potentials
    SPIKES = "spikes"  # Single unit activity
    CALCIUM = "calcium"  # Calcium imaging
    FMRI = "fmri"  # Blood oxygen level


@dataclass
class NeuralDataConfig:
    """Configuration for neural data generation"""
    signal_type: NeuralSignalType = NeuralSignalType.EEG
    num_channels: int = 32
    sampling_rate: int = 1000  # Hz
    duration: float = 10.0  # seconds
    noise_level: float = 0.5
    
    # Signal-specific parameters
    eeg_bands: Dict[str, Tuple[float, float]] = None
    spike_rates: List[float] = None
    spatial_correlation: float = 0.3
    temporal_correlation: float = 0.7
    
    # Task parameters
    num_classes: int = 4  # e.g., left/right/up/down
    trial_length: float = 2.0  # seconds per trial
    inter_trial_interval: float = 0.5
    
    def __post_init__(self):
        if self.eeg_bands is None:
            self.eeg_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }


class AdvancedNeuralDataGenerator:
    """
    Generates realistic neural data with various signal types
    and complex spatiotemporal patterns
    """
    
    def __init__(self, config: NeuralDataConfig):
        self.config = config
        self.rng = np.random.default_rng(42)
        
        # Generate spatial layout
        self.channel_positions = self._generate_channel_layout()
        self.distance_matrix = self._compute_distances()
        
        # Generate connectivity
        self.connectivity = self._generate_connectivity()
        
    def _generate_channel_layout(self) -> np.ndarray:
        """Generate realistic channel positions"""
        if self.config.signal_type == NeuralSignalType.EEG:
            # 10-20 system approximation
            theta = np.linspace(0, 2*np.pi, self.config.num_channels, endpoint=False)
            r = 0.5 + 0.3 * np.sin(3 * theta)  # Varying radius
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0.2 * np.sin(2 * theta)  # Slight 3D curvature
            
        elif self.config.signal_type == NeuralSignalType.ECOG:
            # Grid layout
            grid_size = int(np.sqrt(self.config.num_channels))
            x, y = np.meshgrid(
                np.linspace(-1, 1, grid_size),
                np.linspace(-1, 1, grid_size)
            )
            x = x.flatten()[:self.config.num_channels]
            y = y.flatten()[:self.config.num_channels]
            z = np.zeros_like(x)
            
        else:
            # Random 3D positions
            x = self.rng.uniform(-1, 1, self.config.num_channels)
            y = self.rng.uniform(-1, 1, self.config.num_channels)
            z = self.rng.uniform(-0.5, 0.5, self.config.num_channels)
            
        return np.column_stack([x, y, z])
    
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise distances between channels"""
        pos = self.channel_positions
        distances = np.zeros((len(pos), len(pos)))
        
        for i in range(len(pos)):
            for j in range(len(pos)):
                distances[i, j] = np.linalg.norm(pos[i] - pos[j])
                
        return distances
    
    def _generate_connectivity(self) -> np.ndarray:
        """Generate functional connectivity between channels"""
        # Distance-based connectivity with randomness
        connectivity = np.exp(-self.distance_matrix / self.config.spatial_correlation)
        
        # Add random connections
        random_connections = self.rng.uniform(0, 0.3, connectivity.shape)
        connectivity += random_connections
        
        # Make symmetric and normalize
        connectivity = (connectivity + connectivity.T) / 2
        np.fill_diagonal(connectivity, 1)
        
        return connectivity
    
    def generate_base_signals(self, num_samples: int) -> np.ndarray:
        """Generate base neural signals based on type"""
        if self.config.signal_type == NeuralSignalType.EEG:
            return self._generate_eeg_signals(num_samples)
        elif self.config.signal_type == NeuralSignalType.SPIKES:
            return self._generate_spike_trains(num_samples)
        elif self.config.signal_type == NeuralSignalType.LFP:
            return self._generate_lfp_signals(num_samples)
        elif self.config.signal_type == NeuralSignalType.CALCIUM:
            return self._generate_calcium_signals(num_samples)
        else:
            return self._generate_generic_signals(num_samples)
    
    def _generate_eeg_signals(self, num_samples: int) -> np.ndarray:
        """Generate realistic EEG signals with frequency bands"""
        signals = np.zeros((self.config.num_channels, num_samples))
        
        for ch in range(self.config.num_channels):
            # Generate each frequency band
            for band_name, (low_freq, high_freq) in self.config.eeg_bands.items():
                # Band-specific amplitude (alpha typically strongest)
                if band_name == 'alpha':
                    amplitude = self.rng.uniform(0.5, 1.5)
                elif band_name == 'gamma':
                    amplitude = self.rng.uniform(0.1, 0.3)
                else:
                    amplitude = self.rng.uniform(0.2, 0.8)
                
                # Generate band-limited signal
                freqs = self.rng.uniform(low_freq, high_freq, 5)
                phases = self.rng.uniform(0, 2*np.pi, 5)
                
                t = np.arange(num_samples) / self.config.sampling_rate
                for f, p in zip(freqs, phases):
                    signals[ch] += amplitude * np.sin(2*np.pi*f*t + p)
        
        # Add 1/f noise (pink noise)
        pink_noise = self._generate_pink_noise(num_samples)
        signals += pink_noise * 0.5
        
        # Add spatial correlation
        signals = self._apply_spatial_correlation(signals)
        
        return signals
    
    def _generate_spike_trains(self, num_samples: int) -> np.ndarray:
        """Generate realistic spike trains"""
        signals = np.zeros((self.config.num_channels, num_samples))
        
        if self.config.spike_rates is None:
            # Random firing rates between 1-50 Hz
            spike_rates = self.rng.uniform(1, 50, self.config.num_channels)
        else:
            spike_rates = self.config.spike_rates
        
        for ch in range(self.config.num_channels):
            # Poisson process for spike times
            rate = spike_rates[ch] / self.config.sampling_rate
            spike_train = self.rng.poisson(rate, num_samples)
            
            # Add refractory period
            refractory_samples = int(0.002 * self.config.sampling_rate)  # 2ms
            for i in range(1, num_samples):
                if spike_train[i] > 0 and np.any(spike_train[max(0, i-refractory_samples):i] > 0):
                    spike_train[i] = 0
            
            # Convert to continuous signal with spike waveforms
            spike_waveform = self._generate_spike_waveform()
            signals[ch] = np.convolve(spike_train, spike_waveform, mode='same')
        
        return signals
    
    def _generate_spike_waveform(self) -> np.ndarray:
        """Generate realistic spike waveform"""
        duration = 0.003  # 3ms spike
        samples = int(duration * self.config.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Typical spike shape
        waveform = np.exp(-t/0.0005) * np.sin(2*np.pi*1000*t)
        waveform[:samples//3] *= np.linspace(0, 1, samples//3)  # Smooth onset
        
        return waveform
    
    def _generate_lfp_signals(self, num_samples: int) -> np.ndarray:
        """Generate local field potential signals"""
        # LFP is like EEG but with higher frequencies and more local
        signals = self._generate_eeg_signals(num_samples)
        
        # Add high-frequency oscillations
        for ch in range(self.config.num_channels):
            # Ripples (100-250 Hz)
            ripple_times = self.rng.choice(num_samples, size=10, replace=False)
            for t in ripple_times:
                duration = int(0.1 * self.config.sampling_rate)  # 100ms
                if t + duration < num_samples:
                    ripple = self._generate_ripple()
                    signals[ch, t:t+duration] += ripple[:duration]
        
        return signals
    
    def _generate_ripple(self) -> np.ndarray:
        """Generate high-frequency ripple oscillation"""
        duration = 0.1  # 100ms
        samples = int(duration * self.config.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Ripple with envelope
        freq = self.rng.uniform(120, 200)  # Hz
        envelope = np.exp(-((t - duration/2) / (duration/4))**2)
        ripple = envelope * np.sin(2*np.pi*freq*t)
        
        return ripple
    
    def _generate_calcium_signals(self, num_samples: int) -> np.ndarray:
        """Generate calcium imaging signals (slow dynamics)"""
        # Downsample time for realistic calcium dynamics
        slow_samples = num_samples // 100
        signals = np.zeros((self.config.num_channels, slow_samples))
        
        for ch in range(self.config.num_channels):
            # Generate sparse calcium transients
            num_transients = self.rng.poisson(5)  # Average 5 per recording
            transient_times = self.rng.choice(slow_samples, size=num_transients, replace=False)
            
            for t in transient_times:
                # Calcium transient shape
                duration = self.rng.uniform(1, 3)  # seconds
                amplitude = self.rng.uniform(0.5, 2.0)
                transient = self._generate_calcium_transient(duration, amplitude)
                
                # Add to signal
                end_idx = min(t + len(transient), slow_samples)
                signals[ch, t:end_idx] += transient[:end_idx-t]
        
        # Upsample back to original rate
        signals_upsampled = np.zeros((self.config.num_channels, num_samples))
        for ch in range(self.config.num_channels):
            signals_upsampled[ch] = np.interp(
                np.arange(num_samples),
                np.linspace(0, num_samples, slow_samples),
                signals[ch]
            )
        
        return signals_upsampled
    
    def _generate_calcium_transient(self, duration: float, amplitude: float) -> np.ndarray:
        """Generate single calcium transient"""
        samples = int(duration * 10)  # 10 Hz for calcium
        t = np.linspace(0, duration, samples)
        
        # Double exponential for calcium dynamics
        rise_time = duration * 0.1
        decay_time = duration * 0.7
        
        transient = amplitude * (
            np.exp(-t/decay_time) - np.exp(-t/rise_time)
        )
        transient[transient < 0] = 0
        
        return transient
    
    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """Generate 1/f (pink) noise"""
        # Generate white noise
        white = self.rng.normal(0, 1, (self.config.num_channels, num_samples))
        
        # FFT to frequency domain
        fft = np.fft.rfft(white, axis=1)
        freqs = np.fft.rfftfreq(num_samples)
        
        # Apply 1/f filter
        fft[:, 1:] = fft[:, 1:] / np.sqrt(freqs[1:])
        
        # Back to time domain
        pink = np.fft.irfft(fft, n=num_samples, axis=1)
        
        return pink
    
    def _generate_generic_signals(self, num_samples: int) -> np.ndarray:
        """Generate generic neural-like signals"""
        # Mix of oscillations and noise
        signals = np.zeros((self.config.num_channels, num_samples))
        
        for ch in range(self.config.num_channels):
            # Base oscillation
            freq = self.rng.uniform(5, 50)
            phase = self.rng.uniform(0, 2*np.pi)
            t = np.arange(num_samples) / self.config.sampling_rate
            signals[ch] = np.sin(2*np.pi*freq*t + phase)
            
            # Add harmonics
            for harmonic in [2, 3]:
                signals[ch] += 0.3 * np.sin(2*np.pi*freq*harmonic*t + phase)
        
        # Add noise
        signals += self.rng.normal(0, self.config.noise_level, signals.shape)
        
        return signals
    
    def _apply_spatial_correlation(self, signals: np.ndarray) -> np.ndarray:
        """Apply spatial correlation based on channel distances"""
        # Mix signals based on connectivity
        correlated = np.zeros_like(signals)
        
        for ch in range(self.config.num_channels):
            # Weighted average with neighbors
            weights = self.connectivity[ch]
            weights = weights / weights.sum()
            
            correlated[ch] = np.sum(signals * weights[:, np.newaxis], axis=0)
        
        return correlated
    
    def add_task_modulation(self, signals: np.ndarray, 
                          task_labels: np.ndarray) -> np.ndarray:
        """Add task-related modulation to signals"""
        modulated = signals.copy()
        
        # Define task-specific patterns
        task_patterns = {
            0: {'channels': [0, 1, 2], 'freq_band': 'alpha', 'power_change': -0.5},  # Left
            1: {'channels': [29, 30, 31], 'freq_band': 'alpha', 'power_change': -0.5},  # Right
            2: {'channels': [10, 11, 12], 'freq_band': 'beta', 'power_change': 0.7},  # Up
            3: {'channels': [20, 21, 22], 'freq_band': 'beta', 'power_change': 0.7},  # Down
        }
        
        samples_per_trial = int(self.config.trial_length * self.config.sampling_rate)
        
        for trial_idx, label in enumerate(task_labels):
            if label in task_patterns:
                pattern = task_patterns[label]
                start_idx = trial_idx * samples_per_trial
                end_idx = start_idx + samples_per_trial
                
                # Modulate specified channels
                for ch in pattern['channels']:
                    if ch < self.config.num_channels:
                        # Add task-specific modulation
                        if pattern['freq_band'] == 'alpha':
                            # Suppress alpha
                            freq = self.rng.uniform(8, 13)
                            modulation = pattern['power_change'] * np.sin(
                                2*np.pi*freq*np.arange(samples_per_trial)/self.config.sampling_rate
                            )
                        else:
                            # Enhance beta
                            freq = self.rng.uniform(15, 30)
                            modulation = pattern['power_change'] * np.sin(
                                2*np.pi*freq*np.arange(samples_per_trial)/self.config.sampling_rate
                            )
                        
                        if end_idx <= signals.shape[1]:
                            modulated[ch, start_idx:end_idx] += modulation
        
        return modulated
    
    def add_artifacts(self, signals: np.ndarray) -> np.ndarray:
        """Add realistic artifacts to neural signals"""
        artifacted = signals.copy()
        
        # Eye blink artifacts (frontal channels)
        if self.config.signal_type == NeuralSignalType.EEG:
            frontal_channels = [i for i in range(self.config.num_channels) 
                              if self.channel_positions[i, 1] > 0.3]
            
            num_blinks = self.rng.poisson(10)  # Average 10 blinks
            blink_times = self.rng.choice(signals.shape[1], size=num_blinks, replace=False)
            
            for t in blink_times:
                blink_artifact = self._generate_blink_artifact()
                for ch in frontal_channels:
                    if t + len(blink_artifact) < signals.shape[1]:
                        artifacted[ch, t:t+len(blink_artifact)] += blink_artifact
        
        # Movement artifacts (random channels)
        num_movements = self.rng.poisson(5)
        for _ in range(num_movements):
            affected_channels = self.rng.choice(
                self.config.num_channels, 
                size=self.rng.integers(1, 5), 
                replace=False
            )
            movement_time = self.rng.integers(0, signals.shape[1] - 1000)
            movement_artifact = self.rng.normal(0, 5, 1000)
            
            for ch in affected_channels:
                artifacted[ch, movement_time:movement_time+1000] += movement_artifact
        
        # Line noise (50/60 Hz)
        line_freq = 50 if self.rng.random() > 0.5 else 60
        t = np.arange(signals.shape[1]) / self.config.sampling_rate
        line_noise = 0.1 * np.sin(2*np.pi*line_freq*t)
        artifacted += line_noise
        
        return artifacted
    
    def _generate_blink_artifact(self) -> np.ndarray:
        """Generate eye blink artifact"""
        duration = 0.3  # 300ms
        samples = int(duration * self.config.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Blink shape (sharp onset, slow recovery)
        artifact = 10 * np.exp(-t/0.1) * (1 - np.exp(-t/0.02))
        
        return artifact
    
    def generate_dataset(self, num_trials: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset with trials and labels"""
        samples_per_trial = int(self.config.trial_length * self.config.sampling_rate)
        total_samples = num_trials * samples_per_trial
        
        # Generate base signals
        signals = self.generate_base_signals(total_samples)
        
        # Generate labels (balanced classes)
        labels = np.repeat(np.arange(self.config.num_classes), 
                          num_trials // self.config.num_classes)
        np.random.shuffle(labels)
        labels = labels[:num_trials]
        
        # Add task modulation
        signals = self.add_task_modulation(signals, labels)
        
        # Add artifacts
        signals = self.add_artifacts(signals)
        
        # Reshape into trials
        trial_data = np.zeros((num_trials, self.config.num_channels, samples_per_trial))
        for i in range(num_trials):
            start = i * samples_per_trial
            end = start + samples_per_trial
            trial_data[i] = signals[:, start:end]
        
        return trial_data, labels
    
    def visualize_data(self, signals: np.ndarray, title: str = "Neural Signals"):
        """Visualize generated neural signals"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Time series
        t = np.arange(signals.shape[1]) / self.config.sampling_rate
        for ch in range(min(5, self.config.num_channels)):
            axes[0].plot(t, signals[ch] + ch*5, alpha=0.7)
        axes[0].set_ylabel('Channels')
        axes[0].set_title(f'{title} - Time Series')
        axes[0].set_xlim(0, min(2, t[-1]))
        
        # Power spectrum
        for ch in range(min(5, self.config.num_channels)):
            freqs, psd = signal.welch(signals[ch], fs=self.config.sampling_rate, nperseg=1024)
            axes[1].semilogy(freqs, psd, alpha=0.7)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power')
        axes[1].set_title('Power Spectral Density')
        axes[1].set_xlim(0, 100)
        
        # Spatial correlation
        corr = np.corrcoef(signals)
        im = axes[2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_title('Channel Correlation Matrix')
        axes[2].set_xlabel('Channel')
        axes[2].set_ylabel('Channel')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate different types of neural data
    configs = [
        NeuralDataConfig(signal_type=NeuralSignalType.EEG, num_channels=32),
        NeuralDataConfig(signal_type=NeuralSignalType.SPIKES, num_channels=16),
        NeuralDataConfig(signal_type=NeuralSignalType.LFP, num_channels=8),
        NeuralDataConfig(signal_type=NeuralSignalType.CALCIUM, num_channels=100),
    ]
    
    for config in configs:
        print(f"\nGenerating {config.signal_type.value} data...")
        generator = AdvancedNeuralDataGenerator(config)
        
        # Generate dataset
        trials, labels = generator.generate_dataset(num_trials=100)
        print(f"Generated {trials.shape[0]} trials of shape {trials.shape[1:]}") 
        print(f"Labels: {np.unique(labels, return_counts=True)}")
        
        # Visualize first trial
        if config.signal_type == NeuralSignalType.EEG:
            generator.visualize_data(trials[0], title=config.signal_type.value.upper())
