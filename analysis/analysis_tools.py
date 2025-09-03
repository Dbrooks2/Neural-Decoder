"""
Analysis and Visualization Tools for Neural Decoders
Comprehensive tools for understanding model behavior and data patterns
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class NeuralDataAnalyzer:
    """Comprehensive analysis of neural data and model predictions"""
    
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        self.results = {}
        
    def analyze_signal_quality(self, data: np.ndarray) -> Dict:
        """Analyze signal quality metrics"""
        print("Analyzing signal quality...")
        
        quality_metrics = {
            'snr': [],
            'artifact_ratio': [],
            'channel_variance': [],
            'channel_correlation': []
        }
        
        for trial in data:
            # Signal-to-noise ratio
            signal_power = np.mean(trial**2, axis=1)
            noise_power = np.mean(np.diff(trial, axis=1)**2, axis=1)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            quality_metrics['snr'].append(np.mean(snr))
            
            # Artifact detection (high amplitude outliers)
            artifact_threshold = 5 * np.std(trial)
            artifact_samples = np.sum(np.abs(trial) > artifact_threshold)
            artifact_ratio = artifact_samples / trial.size
            quality_metrics['artifact_ratio'].append(artifact_ratio)
            
            # Channel variance
            channel_var = np.var(trial, axis=1)
            quality_metrics['channel_variance'].append(channel_var)
            
            # Channel correlation
            channel_corr = np.corrcoef(trial)
            quality_metrics['channel_correlation'].append(channel_corr)
        
        # Summary statistics
        summary = {
            'mean_snr': np.mean(quality_metrics['snr']),
            'std_snr': np.std(quality_metrics['snr']),
            'mean_artifact_ratio': np.mean(quality_metrics['artifact_ratio']),
            'dead_channels': self._detect_dead_channels(quality_metrics['channel_variance'])
        }
        
        self.results['signal_quality'] = quality_metrics
        self.results['quality_summary'] = summary
        
        return summary
    
    def _detect_dead_channels(self, channel_variances: List[np.ndarray]) -> List[int]:
        """Detect channels with very low variance"""
        mean_variances = np.mean(channel_variances, axis=0)
        threshold = np.percentile(mean_variances, 5)
        dead_channels = np.where(mean_variances < threshold)[0].tolist()
        return dead_channels
    
    def analyze_frequency_content(self, data: np.ndarray) -> Dict:
        """Analyze frequency content of signals"""
        print("Analyzing frequency content...")
        
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (70, 150)
        }
        
        band_powers = {band: [] for band in freq_bands}
        
        for trial in data:
            for ch in range(trial.shape[0]):
                # Compute power spectral density
                freqs, psd = signal.welch(trial[ch], fs=self.sampling_rate, nperseg=256)
                
                # Calculate band powers
                for band, (low, high) in freq_bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    band_powers[band].append(band_power)
        
        # Average band powers
        avg_band_powers = {band: np.mean(powers) for band, powers in band_powers.items()}
        
        self.results['frequency_analysis'] = {
            'band_powers': band_powers,
            'avg_band_powers': avg_band_powers,
            'dominant_frequency': self._find_dominant_frequency(data)
        }
        
        return avg_band_powers
    
    def _find_dominant_frequency(self, data: np.ndarray) -> float:
        """Find the dominant frequency across all channels"""
        all_psds = []
        
        for trial in data[:10]:  # Sample first 10 trials
            for ch in range(trial.shape[0]):
                freqs, psd = signal.welch(trial[ch], fs=self.sampling_rate, nperseg=256)
                all_psds.append(psd)
        
        avg_psd = np.mean(all_psds, axis=0)
        dominant_freq = freqs[np.argmax(avg_psd)]
        
        return dominant_freq
    
    def analyze_temporal_patterns(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Analyze temporal patterns in the data"""
        print("Analyzing temporal patterns...")
        
        temporal_analysis = {
            'event_related_potentials': {},
            'temporal_dynamics': {},
            'phase_consistency': {}
        }
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get trials for this class
            class_trials = data[labels == label]
            
            # Event-related potential (average waveform)
            erp = np.mean(class_trials, axis=0)
            temporal_analysis['event_related_potentials'][f'class_{label}'] = erp
            
            # Temporal dynamics (how signal evolves over time)
            temporal_std = np.std(class_trials, axis=0)
            temporal_analysis['temporal_dynamics'][f'class_{label}'] = temporal_std
            
            # Phase consistency across trials
            phase_consistency = self._calculate_phase_consistency(class_trials)
            temporal_analysis['phase_consistency'][f'class_{label}'] = phase_consistency
        
        self.results['temporal_analysis'] = temporal_analysis
        
        return temporal_analysis
    
    def _calculate_phase_consistency(self, trials: np.ndarray) -> np.ndarray:
        """Calculate inter-trial phase coherence"""
        n_trials, n_channels, n_samples = trials.shape
        phase_consistency = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Hilbert transform to get instantaneous phase
            analytic_signals = signal.hilbert(trials[:, ch, :], axis=1)
            phases = np.angle(analytic_signals)
            
            # Calculate phase consistency
            mean_vector = np.mean(np.exp(1j * phases), axis=0)
            phase_consistency[ch] = np.abs(mean_vector)
        
        return phase_consistency
    
    def analyze_spatial_patterns(self, data: np.ndarray, channel_positions: Optional[np.ndarray] = None) -> Dict:
        """Analyze spatial patterns in the data"""
        print("Analyzing spatial patterns...")
        
        spatial_analysis = {
            'correlation_matrix': [],
            'spatial_components': {},
            'connectivity': {}
        }
        
        # Channel correlation
        for trial in data[:100]:  # Sample first 100 trials
            corr_matrix = np.corrcoef(trial)
            spatial_analysis['correlation_matrix'].append(corr_matrix)
        
        avg_correlation = np.mean(spatial_analysis['correlation_matrix'], axis=0)
        
        # Spatial decomposition (PCA)
        pca = PCA(n_components=10)
        data_reshaped = data.reshape(data.shape[0], -1)
        pca_components = pca.fit_transform(data_reshaped)
        
        spatial_analysis['spatial_components'] = {
            'pca_components': pca.components_,
            'pca_variance_explained': pca.explained_variance_ratio_,
            'pca_scores': pca_components
        }
        
        # Spatial connectivity
        if channel_positions is not None:
            connectivity = self._calculate_spatial_connectivity(avg_correlation, channel_positions)
            spatial_analysis['connectivity'] = connectivity
        
        self.results['spatial_analysis'] = spatial_analysis
        
        return spatial_analysis
    
    def _calculate_spatial_connectivity(self, correlation_matrix: np.ndarray, 
                                      positions: np.ndarray) -> Dict:
        """Calculate spatial connectivity metrics"""
        n_channels = len(positions)
        
        # Distance matrix
        distances = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                distances[i, j] = np.linalg.norm(positions[i] - positions[j])
        
        # Connectivity strength vs distance
        connectivity_strength = []
        distance_bins = np.linspace(0, np.max(distances), 10)
        
        for i in range(len(distance_bins) - 1):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
            if np.any(mask):
                mean_corr = np.mean(np.abs(correlation_matrix[mask]))
                connectivity_strength.append(mean_corr)
        
        return {
            'distance_matrix': distances,
            'connectivity_by_distance': connectivity_strength,
            'distance_bins': distance_bins
        }
    
    def visualize_analysis(self):
        """Create comprehensive visualization of analysis results"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Signal Quality
        ax1 = plt.subplot(3, 4, 1)
        if 'signal_quality' in self.results:
            snr_values = self.results['signal_quality']['snr']
            ax1.hist(snr_values, bins=30, alpha=0.7, color='blue')
            ax1.set_xlabel('SNR (dB)')
            ax1.set_ylabel('Count')
            ax1.set_title('Signal-to-Noise Ratio Distribution')
            ax1.axvline(np.mean(snr_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(snr_values):.1f} dB')
            ax1.legend()
        
        # 2. Frequency Content
        ax2 = plt.subplot(3, 4, 2)
        if 'frequency_analysis' in self.results:
            bands = list(self.results['frequency_analysis']['avg_band_powers'].keys())
            powers = list(self.results['frequency_analysis']['avg_band_powers'].values())
            ax2.bar(bands, powers, color='green', alpha=0.7)
            ax2.set_xlabel('Frequency Band')
            ax2.set_ylabel('Average Power')
            ax2.set_title('Frequency Band Powers')
            ax2.set_yscale('log')
        
        # 3. Channel Variance
        ax3 = plt.subplot(3, 4, 3)
        if 'signal_quality' in self.results:
            channel_vars = np.mean(self.results['signal_quality']['channel_variance'], axis=0)
            ax3.plot(channel_vars, 'o-', color='orange')
            ax3.set_xlabel('Channel')
            ax3.set_ylabel('Variance')
            ax3.set_title('Channel Variance')
            
            # Mark dead channels
            if 'quality_summary' in self.results:
                dead_channels = self.results['quality_summary']['dead_channels']
                for ch in dead_channels:
                    ax3.axvline(ch, color='red', linestyle='--', alpha=0.5)
        
        # 4. Average Correlation Matrix
        ax4 = plt.subplot(3, 4, 4)
        if 'spatial_analysis' in self.results:
            avg_corr = np.mean(self.results['spatial_analysis']['correlation_matrix'], axis=0)
            im = ax4.imshow(avg_corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_xlabel('Channel')
            ax4.set_ylabel('Channel')
            ax4.set_title('Average Channel Correlation')
            plt.colorbar(im, ax=ax4)
        
        # 5. Event-Related Potentials
        ax5 = plt.subplot(3, 4, 5)
        if 'temporal_analysis' in self.results:
            erps = self.results['temporal_analysis']['event_related_potentials']
            time_axis = np.arange(list(erps.values())[0].shape[1]) / self.sampling_rate
            
            for label, erp in erps.items():
                # Plot average across channels
                avg_erp = np.mean(erp, axis=0)
                ax5.plot(time_axis, avg_erp, label=label)
            
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Amplitude')
            ax5.set_title('Event-Related Potentials')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. PCA Components
        ax6 = plt.subplot(3, 4, 6)
        if 'spatial_analysis' in self.results:
            var_explained = self.results['spatial_analysis']['spatial_components']['pca_variance_explained']
            ax6.plot(np.cumsum(var_explained), 'o-', color='purple')
            ax6.set_xlabel('Number of Components')
            ax6.set_ylabel('Cumulative Variance Explained')
            ax6.set_title('PCA Variance Explained')
            ax6.grid(True, alpha=0.3)
        
        # 7. Temporal Dynamics
        ax7 = plt.subplot(3, 4, 7)
        if 'temporal_analysis' in self.results:
            dynamics = self.results['temporal_analysis']['temporal_dynamics']
            time_axis = np.arange(list(dynamics.values())[0].shape[1]) / self.sampling_rate
            
            for label, std in dynamics.items():
                # Plot average std across channels
                avg_std = np.mean(std, axis=0)
                ax7.plot(time_axis, avg_std, label=label)
            
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Standard Deviation')
            ax7.set_title('Temporal Variability by Class')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Phase Consistency
        ax8 = plt.subplot(3, 4, 8)
        if 'temporal_analysis' in self.results:
            phase_consistency = self.results['temporal_analysis']['phase_consistency']
            
            for label, consistency in phase_consistency.items():
                # Plot average across channels
                avg_consistency = np.mean(consistency, axis=0)
                time_axis = np.arange(len(avg_consistency)) / self.sampling_rate
                ax8.plot(time_axis, avg_consistency, label=label)
            
            ax8.set_xlabel('Time (s)')
            ax8.set_ylabel('Phase Consistency')
            ax8.set_title('Inter-Trial Phase Coherence')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Connectivity by Distance
        ax9 = plt.subplot(3, 4, 9)
        if 'connectivity' in self.results.get('spatial_analysis', {}):
            connectivity = self.results['spatial_analysis']['connectivity']
            distances = connectivity['distance_bins'][:-1]
            strengths = connectivity['connectivity_by_distance']
            
            ax9.plot(distances, strengths, 'o-', color='brown')
            ax9.set_xlabel('Distance')
            ax9.set_ylabel('Connectivity Strength')
            ax9.set_title('Connectivity vs Distance')
            ax9.grid(True, alpha=0.3)
        
        # 10. Artifact Distribution
        ax10 = plt.subplot(3, 4, 10)
        if 'signal_quality' in self.results:
            artifact_ratios = self.results['signal_quality']['artifact_ratio']
            ax10.hist(artifact_ratios, bins=30, alpha=0.7, color='red')
            ax10.set_xlabel('Artifact Ratio')
            ax10.set_ylabel('Count')
            ax10.set_title('Artifact Distribution')
            ax10.axvline(np.mean(artifact_ratios), color='black', linestyle='--',
                        label=f'Mean: {np.mean(artifact_ratios):.3f}')
            ax10.legend()
        
        plt.tight_layout()
        plt.savefig('neural_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_path: str = 'analysis_report.txt'):
        """Generate comprehensive text report"""
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEURAL DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Signal Quality
            if 'quality_summary' in self.results:
                f.write("SIGNAL QUALITY METRICS\n")
                f.write("-" * 40 + "\n")
                summary = self.results['quality_summary']
                f.write(f"Average SNR: {summary['mean_snr']:.2f} ± {summary['std_snr']:.2f} dB\n")
                f.write(f"Average Artifact Ratio: {summary['mean_artifact_ratio']:.4f}\n")
                f.write(f"Dead Channels: {summary['dead_channels']}\n\n")
            
            # Frequency Analysis
            if 'frequency_analysis' in self.results:
                f.write("FREQUENCY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write("Average Band Powers:\n")
                for band, power in self.results['frequency_analysis']['avg_band_powers'].items():
                    f.write(f"  {band}: {power:.2e}\n")
                f.write(f"Dominant Frequency: {self.results['frequency_analysis']['dominant_frequency']:.2f} Hz\n\n")
            
            # Spatial Analysis
            if 'spatial_analysis' in self.results:
                f.write("SPATIAL ANALYSIS\n")
                f.write("-" * 40 + "\n")
                var_explained = self.results['spatial_analysis']['spatial_components']['pca_variance_explained']
                f.write(f"PCA Components needed for 95% variance: {np.argmax(np.cumsum(var_explained) > 0.95) + 1}\n")
                f.write(f"First 3 components explain: {np.sum(var_explained[:3])*100:.1f}% variance\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated successfully\n")


class ModelAnalyzer:
    """Analyze trained neural decoder models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def analyze_predictions(self, data_loader, num_classes: int) -> Dict:
        """Analyze model predictions"""
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # Get predictions and confidence
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        # Calculate metrics
        predictions_analysis = {
            'confusion_matrix': confusion_matrix(all_targets, all_predictions),
            'class_accuracies': self._calculate_class_accuracies(all_targets, all_predictions, num_classes),
            'confidence_by_correctness': self._analyze_confidence(all_targets, all_predictions, all_confidences),
            'error_analysis': self._analyze_errors(all_targets, all_predictions, all_confidences)
        }
        
        return predictions_analysis
    
    def _calculate_class_accuracies(self, targets, predictions, num_classes):
        """Calculate per-class accuracies"""
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        
        for t, p in zip(targets, predictions):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1
        
        class_accuracies = class_correct / (class_total + 1e-10)
        
        return {
            'accuracies': class_accuracies,
            'support': class_total
        }
    
    def _analyze_confidence(self, targets, predictions, confidences):
        """Analyze confidence scores"""
        correct_mask = np.array(targets) == np.array(predictions)
        
        return {
            'mean_confidence_correct': np.mean(np.array(confidences)[correct_mask]),
            'mean_confidence_incorrect': np.mean(np.array(confidences)[~correct_mask]),
            'confidence_histogram_correct': np.histogram(np.array(confidences)[correct_mask], bins=20),
            'confidence_histogram_incorrect': np.histogram(np.array(confidences)[~correct_mask], bins=20)
        }
    
    def _analyze_errors(self, targets, predictions, confidences):
        """Detailed error analysis"""
        errors = []
        
        for i, (t, p, c) in enumerate(zip(targets, predictions, confidences)):
            if t != p:
                errors.append({
                    'index': i,
                    'true_label': t,
                    'predicted_label': p,
                    'confidence': c
                })
        
        # Find most common confusions
        confusion_pairs = {}
        for error in errors:
            pair = (error['true_label'], error['predicted_label'])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(targets),
            'top_confusions': sorted_confusions[:10],
            'low_confidence_errors': [e for e in errors if e['confidence'] < 0.5]
        }
    
    def visualize_predictions(self, predictions_analysis: Dict):
        """Visualize prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        ax = axes[0, 0]
        sns.heatmap(predictions_analysis['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Class Accuracies
        ax = axes[0, 1]
        class_acc = predictions_analysis['class_accuracies']['accuracies']
        ax.bar(range(len(class_acc)), class_acc, color='green', alpha=0.7)
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy')
        ax.set_ylim(0, 1)
        
        # Confidence Analysis
        ax = axes[1, 0]
        conf_analysis = predictions_analysis['confidence_by_correctness']
        
        # Plot confidence histograms
        bins_correct = conf_analysis['confidence_histogram_correct'][1][:-1]
        counts_correct = conf_analysis['confidence_histogram_correct'][0]
        bins_incorrect = conf_analysis['confidence_histogram_incorrect'][1][:-1]
        counts_incorrect = conf_analysis['confidence_histogram_incorrect'][0]
        
        width = bins_correct[1] - bins_correct[0]
        ax.bar(bins_correct, counts_correct, width=width, alpha=0.5, 
               label='Correct', color='green')
        ax.bar(bins_incorrect, counts_incorrect, width=width, alpha=0.5, 
               label='Incorrect', color='red')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend()
        
        # Top Confusions
        ax = axes[1, 1]
        error_analysis = predictions_analysis['error_analysis']
        if error_analysis['top_confusions']:
            confusions = error_analysis['top_confusions'][:5]
            labels = [f"{t}->{p}" for (t, p), _ in confusions]
            counts = [count for _, count in confusions]
            
            ax.barh(labels, counts, color='red', alpha=0.7)
            ax.set_xlabel('Count')
            ax.set_title('Top 5 Confusions')
        
        plt.tight_layout()
        plt.savefig('model_predictions_analysis.png', dpi=300)
        plt.show()


def perform_comprehensive_analysis(data: np.ndarray, labels: np.ndarray, 
                                 model: Optional[nn.Module] = None,
                                 data_loader = None) -> Dict:
    """Perform comprehensive analysis of data and model"""
    
    print("Starting comprehensive neural data analysis...")
    
    # Data Analysis
    analyzer = NeuralDataAnalyzer()
    
    # Run all analyses
    signal_quality = analyzer.analyze_signal_quality(data)
    frequency_content = analyzer.analyze_frequency_content(data)
    temporal_patterns = analyzer.analyze_temporal_patterns(data, labels)
    spatial_patterns = analyzer.analyze_spatial_patterns(data)
    
    # Visualize
    analyzer.visualize_analysis()
    
    # Generate report
    analyzer.generate_report()
    
    # Model Analysis (if provided)
    if model is not None and data_loader is not None:
        print("\nAnalyzing model predictions...")
        model_analyzer = ModelAnalyzer(model)
        predictions_analysis = model_analyzer.analyze_predictions(
            data_loader, 
            num_classes=len(np.unique(labels))
        )
        model_analyzer.visualize_predictions(predictions_analysis)
    
    print("\nAnalysis complete!")
    
    return analyzer.results


# Example usage
if __name__ == "__main__":
    # Generate sample data
    from .advanced_data_generation import AdvancedNeuralDataGenerator, NeuralDataConfig
    
    config = NeuralDataConfig(num_channels=32, num_classes=4)
    generator = AdvancedNeuralDataGenerator(config)
    data, labels = generator.generate_dataset(num_trials=100)
    
    # Perform analysis
    results = perform_comprehensive_analysis(data, labels)
    
    print("\nAnalysis Summary:")
    print(f"Signal Quality - Mean SNR: {results['quality_summary']['mean_snr']:.2f} dB")
    print(f"Dominant Frequency: {results['frequency_analysis']['dominant_frequency']:.2f} Hz")
    print(f"Dead Channels: {results['quality_summary']['dead_channels']}")
