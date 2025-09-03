"""
Complete Neural Decoder Pipeline
Generates data, trains models, and performs comprehensive analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "data"))
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "training"))
sys.path.append(str(Path(__file__).parent.parent / "analysis"))

from advanced_data_generation import (
    AdvancedNeuralDataGenerator, 
    NeuralDataConfig, 
    NeuralSignalType
)
from advanced_models import build_advanced_model
from advanced_training import (
    AdvancedTrainer, 
    TrainingConfig,
    AdvancedNeuralDataset,
    DataAugmentation
)
from analysis_tools import perform_comprehensive_analysis


def run_complete_pipeline():
    """Run the complete neural decoder pipeline"""
    
    print("=" * 80)
    print("NEURAL DECODER COMPLETE PIPELINE")
    print("=" * 80)
    
    # Create output directories
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/models").mkdir(exist_ok=True)
    Path("outputs/figures").mkdir(exist_ok=True)
    Path("outputs/reports").mkdir(exist_ok=True)
    
    # ========================================
    # STEP 1: DATA GENERATION
    # ========================================
    print("\n[1/4] GENERATING NEURAL DATA...")
    print("-" * 40)
    
    # Configure data generation
    data_config = NeuralDataConfig(
        signal_type=NeuralSignalType.EEG,
        num_channels=32,
        sampling_rate=1000,  # 1 kHz
        duration=10.0,
        noise_level=0.3,
        num_classes=4,  # 4-direction cursor control
        trial_length=2.0,  # 2 second trials
        spatial_correlation=0.4,
        temporal_correlation=0.7
    )
    
    print(f"Signal Type: {data_config.signal_type.value}")
    print(f"Channels: {data_config.num_channels}")
    print(f"Sampling Rate: {data_config.sampling_rate} Hz")
    print(f"Classes: {data_config.num_classes} (Left/Right/Up/Down)")
    
    # Generate datasets
    generator = AdvancedNeuralDataGenerator(data_config)
    
    print("\nGenerating training data...")
    train_data, train_labels = generator.generate_dataset(num_trials=800)
    print(f"Training set: {train_data.shape[0]} trials")
    
    print("Generating validation data...")
    val_data, val_labels = generator.generate_dataset(num_trials=200)
    print(f"Validation set: {val_data.shape[0]} trials")
    
    print("Generating test data...")
    test_data, test_labels = generator.generate_dataset(num_trials=200)
    print(f"Test set: {test_data.shape[0]} trials")
    
    # Visualize sample data
    print("\nVisualizing sample data...")
    generator.visualize_data(train_data[0], title="Sample EEG Trial")
    plt.savefig("outputs/figures/sample_eeg_data.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data statistics
    data_stats = {
        "signal_type": data_config.signal_type.value,
        "num_channels": data_config.num_channels,
        "sampling_rate": data_config.sampling_rate,
        "num_classes": data_config.num_classes,
        "train_samples": int(train_data.shape[0]),
        "val_samples": int(val_data.shape[0]),
        "test_samples": int(test_data.shape[0]),
        "trial_length_seconds": data_config.trial_length,
        "trial_length_samples": int(train_data.shape[2])
    }
    
    with open("outputs/reports/data_statistics.json", "w") as f:
        json.dump(data_stats, f, indent=2)
    
    # ========================================
    # STEP 2: MODEL TRAINING
    # ========================================
    print("\n[2/4] TRAINING NEURAL DECODER MODELS...")
    print("-" * 40)
    
    # We'll train multiple models and compare
    model_types = ["multiscale", "transformer", "hybrid"]
    trained_models = {}
    training_results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        
        # Training configuration
        train_config = TrainingConfig(
            model_type=model_type,
            num_channels=data_config.num_channels,
            num_classes=data_config.num_classes,
            window_size=train_data.shape[2],  # Full trial length
            batch_size=32,
            epochs=30,  # Reduced for demo
            learning_rate=1e-3,
            weight_decay=1e-4,
            use_mixed_precision=torch.cuda.is_available(),
            gradient_clipping=1.0,
            scheduler_type="cosine",
            warmup_epochs=3,
            dropout_rate=0.5,
            label_smoothing=0.1,
            mixup_alpha=0.2,
            augment_data=True,
            eval_every_n_epochs=1,
            save_best_only=True,
            early_stopping_patience=10,
            use_wandb=False,  # Set to True if you have wandb
            experiment_name=f"neural_decoder_{model_type}",
            device="cpu"  # Force CPU due to RTX 5070 compatibility issues
        )
        
        # Create datasets with augmentation
        augmentation = DataAugmentation(train_config)
        train_dataset = AdvancedNeuralDataset(train_data, train_labels, train_config, augmentation)
        val_dataset = AdvancedNeuralDataset(val_data, val_labels, train_config)
        test_dataset = AdvancedNeuralDataset(test_data, test_labels, train_config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_config.batch_size,
            shuffle=True, 
            num_workers=0,  # Set to 0 for Windows
            pin_memory=train_config.pin_memory and torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=train_config.batch_size,
            shuffle=False, 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=train_config.batch_size,
            shuffle=False, 
            num_workers=0
        )
        
        # Build model
        model = build_advanced_model(
            model_type,
            train_config.num_channels,
            train_config.num_classes,
            train_config.window_size
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        start_time = time.time()
        trainer = AdvancedTrainer(model, train_config)
        
        try:
            trainer.train(train_loader, val_loader)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        training_time = time.time() - start_time
        
        # Save trained model
        model_save_path = f"outputs/models/{model_type}_decoder.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': train_config,
            'best_val_acc': trainer.best_val_acc,
            'training_time': training_time
        }, model_save_path)
        
        # Evaluate on test set
        print(f"\nEvaluating {model_type} on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Store results
        trained_models[model_type] = model
        training_results[model_type] = {
            'best_val_acc': trainer.best_val_acc,
            'test_acc': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'training_time': training_time,
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'class_accuracies': test_metrics['class_accuracies']
        }
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Training Time: {training_time/60:.1f} minutes")
        
        # Save training curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.train_losses, label='Train')
        plt.plot(np.arange(0, len(trainer.val_losses)) * train_config.eval_every_n_epochs, 
                trainer.val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_type.upper()} - Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(trainer.train_accs, label='Train')
        plt.plot(np.arange(0, len(trainer.val_accs)) * train_config.eval_every_n_epochs,
                trainer.val_accs, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{model_type.upper()} - Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"outputs/figures/{model_type}_training_curves.png", dpi=300)
        plt.close()
    
    # Save training results
    with open("outputs/reports/training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)
    
    # ========================================
    # STEP 3: MODEL COMPARISON
    # ========================================
    print("\n[3/4] COMPARING MODELS...")
    print("-" * 40)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy comparison
    ax = axes[0, 0]
    model_names = list(training_results.keys())
    test_accs = [training_results[m]['test_acc'] for m in model_names]
    bars = ax.bar(model_names, test_accs, color=['blue', 'green', 'orange'])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Training time comparison
    ax = axes[0, 1]
    train_times = [training_results[m]['training_time']/60 for m in model_names]
    ax.bar(model_names, train_times, color=['blue', 'green', 'orange'])
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison')
    
    # Class accuracies comparison
    ax = axes[1, 0]
    class_names = ['Left', 'Right', 'Up', 'Down']
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        class_accs = training_results[model_name]['class_accuracies']
        ax.bar(x + i*width, class_accs, width, label=model_name)
    
    ax.set_xlabel('Direction')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Best model confusion matrix
    best_model = max(training_results.items(), key=lambda x: x[1]['test_acc'])[0]
    ax = axes[1, 1]
    cm = np.array(training_results[best_model]['confusion_matrix'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Best Model ({best_model}) Confusion Matrix')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("outputs/figures/model_comparison.png", dpi=300)
    plt.close()
    
    print(f"\nBest performing model: {best_model} ({training_results[best_model]['test_acc']:.2f}% accuracy)")
    
    # ========================================
    # STEP 4: COMPREHENSIVE ANALYSIS
    # ========================================
    print("\n[4/4] PERFORMING COMPREHENSIVE ANALYSIS...")
    print("-" * 40)
    
    # Analyze the neural data and best model
    best_model_obj = trained_models[best_model]
    
    print("Analyzing neural data quality and patterns...")
    analysis_results = perform_comprehensive_analysis(
        data=test_data,
        labels=test_labels,
        model=best_model_obj,
        data_loader=test_loader
    )
    
    # Move generated plots to outputs folder
    import shutil
    if Path("neural_data_analysis.png").exists():
        shutil.move("neural_data_analysis.png", "outputs/figures/neural_data_analysis.png")
    if Path("model_predictions_analysis.png").exists():
        shutil.move("model_predictions_analysis.png", "outputs/figures/model_predictions_analysis.png")
    if Path("analysis_report.txt").exists():
        shutil.move("analysis_report.txt", "outputs/reports/analysis_report.txt")
    
    # ========================================
    # INFERENCE SPEED TEST
    # ========================================
    print("\n[BONUS] TESTING INFERENCE SPEED...")
    print("-" * 40)
    
    # Test inference speed for real-time applications
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_obj = best_model_obj.to(device)
    best_model_obj.eval()
    
    # Single sample inference
    test_input = torch.randn(1, data_config.num_channels, train_data.shape[2]).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = best_model_obj(test_input)
    
    # Time inference
    num_inferences = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_inferences):
            output = best_model_obj(test_input)
    
    inference_time = (time.time() - start_time) / num_inferences * 1000  # ms
    
    print(f"Average inference time: {inference_time:.2f} ms")
    print(f"Maximum real-time frequency: {1000/inference_time:.1f} Hz")
    
    if inference_time < 50:
        print("✓ Meets <50ms latency requirement for real-time BCI!")
    else:
        print("✗ Does not meet <50ms latency requirement")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    summary = {
        "data": {
            "type": data_config.signal_type.value,
            "channels": data_config.num_channels,
            "sampling_rate": data_config.sampling_rate,
            "total_samples": train_data.shape[0] + val_data.shape[0] + test_data.shape[0]
        },
        "models": {
            model_name: {
                "test_accuracy": results['test_acc'],
                "training_time_minutes": results['training_time']/60,
                "parameters": sum(p.numel() for p in trained_models[model_name].parameters())
            }
            for model_name, results in training_results.items()
        },
        "best_model": {
            "name": best_model,
            "test_accuracy": training_results[best_model]['test_acc'],
            "inference_time_ms": inference_time
        },
        "analysis": {
            "mean_snr_db": float(analysis_results.get('quality_summary', {}).get('mean_snr', 0)),
            "dominant_frequency_hz": float(analysis_results.get('frequency_analysis', {}).get('dominant_frequency', 0))
        }
    }
    
    with open("outputs/reports/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary:")
    print(f"- Generated {summary['data']['total_samples']} neural signal trials")
    print(f"- Trained {len(model_types)} different model architectures")
    print(f"- Best model: {best_model} with {summary['best_model']['test_accuracy']:.1f}% accuracy")
    print(f"- Inference latency: {summary['best_model']['inference_time_ms']:.1f} ms")
    print(f"- Signal quality: {summary['analysis']['mean_snr_db']:.1f} dB SNR")
    
    print("\nOutputs saved to:")
    print("- outputs/figures/  (visualizations)")
    print("- outputs/models/   (trained models)")
    print("- outputs/reports/  (analysis reports)")
    
    return summary


if __name__ == "__main__":
    # Run the complete pipeline
    try:
        results = run_complete_pipeline()
        print("\n✓ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
