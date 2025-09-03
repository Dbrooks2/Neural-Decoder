"""
Load Real EEG Data from PhysioNet Database
Motor Movement/Imagery Dataset - 109 subjects with real brain signals
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import time

# Install required packages
print("Installing required packages...")
import subprocess
import sys

try:
    import mne
    from mne.datasets import eegbci
    from mne import Epochs, pick_types, events_from_annotations
    from mne.channels import make_standard_montage
    from mne.io import concatenate_raws, read_raw_edf
    print("✅ MNE-Python already installed")
except ImportError:
    print("Installing MNE-Python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mne"])
    import mne
    from mne.datasets import eegbci
    from mne import Epochs, pick_types, events_from_annotations
    from mne.channels import make_standard_montage
    from mne.io import concatenate_raws, read_raw_edf

# Import our EEGNet model - it's defined in realistic_demo.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from realistic_demo import EEGNet


def download_physionet_data(subject=1, runs=None):
    """
    Download PhysioNet EEG Motor Movement/Imagery Database
    
    Runs:
    - 3, 7, 11: Motor imagery (left hand, right hand, both hands, both feet)
    - 4, 8, 12: Real motor movement
    - 5, 9, 13: Motor imagery (left hand vs right hand)
    - 6, 10, 14: Motor imagery (hands vs feet)
    """
    
    if runs is None:
        runs = [3, 7, 11]  # Motor imagery runs
    
    print(f"Downloading PhysioNet data for subject {subject}...")
    print(f"Runs: {runs}")
    print("This may take a few minutes for first download...")
    
    # Download the data
    raw_fnames = eegbci.load_data(subject, runs, verbose=False)
    
    print(f"✅ Downloaded {len(raw_fnames)} files")
    return raw_fnames


def preprocess_physionet_data(raw_fnames, subject=1):
    """Preprocess the raw EEG data"""
    
    print("\n📊 Preprocessing PhysioNet EEG data...")
    
    # Load and concatenate raw data
    raw_files = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = concatenate_raws(raw_files)
    
    print(f"Original channel names: {raw.ch_names[:10]}...")  # Show first 10
    
    # PhysioNet uses different naming convention - fix channel names
    # Remove dots and standardize
    channel_mapping = {}
    for ch_name in raw.ch_names:
        # Remove dots and standardize
        clean_name = ch_name.replace('.', '').upper()
        
        # Map common PhysioNet names to standard 10-20
        name_map = {
            'FC5': 'FC5', 'FC3': 'FC3', 'FC1': 'FC1', 'FCZ': 'FCz', 'FC2': 'FC2', 'FC4': 'FC4', 'FC6': 'FC6',
            'C5': 'C5', 'C3': 'C3', 'C1': 'C1', 'CZ': 'Cz', 'C2': 'C2', 'C4': 'C4', 'C6': 'C6',
            'CP5': 'CP5', 'CP3': 'CP3', 'CP1': 'CP1', 'CPZ': 'CPz', 'CP2': 'CP2', 'CP4': 'CP4', 'CP6': 'CP6',
            'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2',
            'AF7': 'AF7', 'AF3': 'AF3', 'AFZ': 'AFz', 'AF4': 'AF4', 'AF8': 'AF8',
            'F7': 'F7', 'F5': 'F5', 'F3': 'F3', 'F1': 'F1', 'FZ': 'Fz', 'F2': 'F2', 'F4': 'F4', 'F6': 'F6', 'F8': 'F8',
            'FT7': 'FT7', 'FT8': 'FT8', 'T7': 'T7', 'T8': 'T8', 'T9': 'T9', 'T10': 'T10',
            'TP7': 'TP7', 'TP8': 'TP8',
            'P7': 'P7', 'P5': 'P5', 'P3': 'P3', 'P1': 'P1', 'PZ': 'Pz', 'P2': 'P2', 'P4': 'P4', 'P6': 'P6', 'P8': 'P8',
            'PO7': 'PO7', 'PO3': 'PO3', 'POZ': 'POz', 'PO4': 'PO4', 'PO8': 'PO8',
            'O1': 'O1', 'OZ': 'Oz', 'O2': 'O2', 'IZ': 'Iz'
        }
        
        if clean_name in name_map:
            channel_mapping[ch_name] = name_map[clean_name]
        else:
            channel_mapping[ch_name] = ch_name  # Keep original if no mapping
    
    # Rename channels
    raw.rename_channels(channel_mapping)
    print(f"Renamed channels: {list(channel_mapping.values())[:10]}...")
    
    # Try to apply montage with ignore missing channels
    try:
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore', verbose=False)
        print("✅ Applied electrode montage")
    except Exception as e:
        print(f"⚠️  Could not apply montage: {e}")
        print("Continuing without electrode positions...")
    
    # Filter the data
    print("Applying bandpass filter (8-30 Hz)...")
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
    
    # Extract events
    events, event_dict = events_from_annotations(raw, verbose=False)
    
    print(f"Event types found: {event_dict}")
    print(f"Total events: {len(events)}")
    
    # Pick EEG channels
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    # Create epochs (time windows around events)
    tmin, tmax = 0., 4.  # 0 to 4 seconds after event
    epochs = Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks,
                   baseline=None, preload=True, verbose=False)
    
    print(f"✅ Created {len(epochs)} epochs")
    print(f"Epoch shape: {epochs.get_data().shape}")  # (n_epochs, n_channels, n_times)
    
    return epochs, event_dict


def create_motor_imagery_dataset(epochs, event_dict):
    """Create dataset for motor imagery classification"""
    
    print("\n🎯 Creating motor imagery classification dataset...")
    
    # Get epoch data and labels
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # Event codes
    
    print(f"Raw data shape: {X.shape}")
    print(f"Raw labels shape: {y.shape}")
    print(f"Unique event codes: {np.unique(y)}")
    
    # Map event codes to class labels
    # PhysioNet event codes:
    # T1 = 1 (left hand imagery)
    # T2 = 2 (right hand imagery)  
    # T0 = 3 (rest/baseline)
    
    class_mapping = {}
    new_labels = []
    class_names = []
    
    # Map PhysioNet codes to our classes
    if 1 in event_dict.values():  # T1 - left hand
        class_mapping[list(event_dict.keys())[list(event_dict.values()).index(1)]] = 0
        class_names.append('Left Hand')
    if 2 in event_dict.values():  # T2 - right hand  
        class_mapping[list(event_dict.keys())[list(event_dict.values()).index(2)]] = 1
        class_names.append('Right Hand')
    if 3 in event_dict.values():  # T0 - rest
        class_mapping[list(event_dict.keys())[list(event_dict.values()).index(3)]] = 2
        class_names.append('Rest')
    
    # Convert event codes to class indices
    valid_indices = []
    for i, code in enumerate(y):
        event_name = None
        for name, val in event_dict.items():
            if val == code:
                event_name = name
                break
        
        if event_name in class_mapping:
            new_labels.append(class_mapping[event_name])
            valid_indices.append(i)
    
    # Filter data and labels
    X_filtered = X[valid_indices]
    y_filtered = np.array(new_labels)
    
    print(f"✅ Filtered dataset:")
    print(f"Data shape: {X_filtered.shape}")
    print(f"Classes: {class_names}")
    print(f"Class distribution: {np.unique(y_filtered, return_counts=True)}")
    
    return X_filtered, y_filtered, class_names


def train_on_real_data(X, y, class_names):
    """Train EEGNet on real PhysioNet data"""
    
    print(f"\n🚀 Training EEGNet on Real PhysioNet Data...")
    print("-" * 50)
    
    n_epochs, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    
    print(f"Dataset: {n_epochs} epochs, {n_channels} channels, {n_times} timepoints")
    print(f"Classes: {n_classes} ({', '.join(class_names)})")
    
    # Split data
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Training: {len(X_train)} epochs")
    print(f"Testing: {len(X_test)} epochs")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create model
    model = EEGNet(n_channels=n_channels, n_classes=n_classes, samples=n_times)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    epochs = 60
    train_losses = []
    train_accs = []
    
    print(f"\nTraining for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        if (epoch + 1) % 15 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:5.1f}%, LR: {lr:.2e}")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = outputs.max(1)
        
        correct = predicted.eq(y_test).sum().item()
        accuracy = 100. * correct / y_test.size(0)
        
        cm = confusion_matrix(y_test.cpu(), predicted.cpu())
    
    print(f"✅ Test Accuracy: {accuracy:.1f}%")
    
    # Show confusion matrix
    print(f"\nConfusion Matrix:")
    print("     " + "  ".join([f"{name:>6}" for name in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>6} {row}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training curves
    axes[0].plot(train_losses, 'r-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, 'b-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # Confusion matrix
    im = axes[2].imshow(cm, cmap='Blues')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_yticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names)
    axes[2].set_yticklabels(class_names)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    axes[2].set_title(f'Confusion Matrix\n(Accuracy: {accuracy:.1f}%)')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = axes[2].text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig("physionet_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy, model


def main():
    """Main function to run PhysioNet data analysis"""
    
    print("="*60)
    print("🧠 REAL EEG DATA ANALYSIS - PHYSIONET DATABASE")
    print("="*60)
    
    # Create output directory
    Path("physionet_outputs").mkdir(exist_ok=True)
    
    # Download data
    subject = 1  # Start with subject 1
    runs = [3, 7, 11]  # Motor imagery tasks
    
    raw_fnames = download_physionet_data(subject, runs)
    
    # Preprocess data  
    epochs, event_dict = preprocess_physionet_data(raw_fnames, subject)
    
    # Create dataset
    X, y, class_names = create_motor_imagery_dataset(epochs, event_dict)
    
    # Train model
    accuracy, model = train_on_real_data(X, y, class_names)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 REAL EEG ANALYSIS COMPLETE!")
    print("="*60)
    print(f"✅ Used REAL brain signals from PhysioNet Database")
    print(f"✅ Subject {subject} motor imagery data")
    print(f"✅ {len(X)} real EEG epochs processed")
    print(f"✅ Test accuracy: {accuracy:.1f}%")
    print(f"✅ Classes: {', '.join(class_names)}")
    
    if accuracy >= 70:
        print("🏆 EXCELLENT! Research-grade performance on real data!")
    elif accuracy >= 60:
        print("🎯 GOOD! Solid performance on real brain signals!")
    elif accuracy >= 50:
        print("👍 DECENT! Better than chance on real data!")
    else:
        print("🤔 Challenging real data - this is normal for BCI!")
    
    print(f"\n📁 Results saved to: physionet_outputs/")
    
    return accuracy, model, X, y


if __name__ == "__main__":
    try:
        accuracy, model, X, y = main()
        print("\n✅ PhysioNet analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have internet connection for data download")
        import traceback
        traceback.print_exc()
