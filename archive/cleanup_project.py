"""
Project Cleanup Script
Organizes the Neural Decoder project into a clean, professional structure
"""

import os
import shutil
from pathlib import Path

def cleanup_neural_decoder():
    """Clean up and organize the Neural Decoder project"""
    
    print("🧹 CLEANING UP NEURAL DECODER PROJECT")
    print("=" * 50)
    
    # Create clean directory structure
    directories = {
        'demos': 'Main demonstration scripts',
        'data': 'Data generation and loading',
        'models': 'Neural network architectures', 
        'training': 'Training and evaluation scripts',
        'analysis': 'Analysis and visualization tools',
        'applications': 'Real-world applications',
        'outputs': 'Generated outputs and results',
        'docs': 'Documentation and examples',
        'tests': 'Test scripts',
        'archive': 'Old/backup files'
    }
    
    # Create directories
    for dir_name, description in directories.items():
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created: {dir_name}/ - {description}")
    
    # File organization plan
    file_moves = {
        # Core demos (keep in root for easy access)
        'quick_demo.py': 'demos/',
        'realistic_demo.py': 'demos/', 
        'load_physionet_data.py': 'demos/',
        'run_complete_pipeline.py': 'demos/',
        
        # Data generation
        'src/neural_decoder/advanced_data_generation.py': 'data/',
        'realistic_eeg_data.py': 'data/',
        
        # Models
        'src/neural_decoder/models.py': 'models/',
        'src/neural_decoder/advanced_models.py': 'models/',
        
        # Training
        'src/neural_decoder/train.py': 'training/',
        'src/neural_decoder/advanced_training.py': 'training/',
        
        # Analysis
        'src/neural_decoder/analysis_tools.py': 'analysis/',
        
        # Applications
        'examples/': 'applications/',
        'src/neural_decoder/robot_control.py': 'applications/',
        'src/neural_decoder/personalized_gesture_trainer.py': 'applications/',
        
        # API
        'src/api/': 'api/',
        
        # Scripts
        'scripts/': 'archive/',
        
        # Outputs
        'demo_outputs/': 'outputs/',
        'realistic_outputs/': 'outputs/',
        'physionet_outputs/': 'outputs/',
        
        # Temp files
        'gpu_test.py': 'archive/',
        'debug_data.py': 'archive/',
        'cleanup_project.py': 'archive/'
    }
    
    print(f"\n📦 Moving files to organized structure...")
    
    # Move files
    moved_count = 0
    for source, destination in file_moves.items():
        source_path = Path(source)
        dest_path = Path(destination)
        
        if source_path.exists():
            try:
                if source_path.is_dir():
                    # Move directory
                    if dest_path.name == source_path.name:
                        # If destination ends with same name, move into parent
                        final_dest = dest_path
                    else:
                        final_dest = dest_path / source_path.name
                    
                    if not final_dest.exists():
                        shutil.move(str(source_path), str(final_dest))
                        print(f"  📁 {source} → {final_dest}")
                        moved_count += 1
                else:
                    # Move file
                    dest_path.mkdir(parents=True, exist_ok=True)
                    final_dest = dest_path / source_path.name
                    if not final_dest.exists():
                        shutil.move(str(source_path), str(final_dest))
                        print(f"  📄 {source} → {final_dest}")
                        moved_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not move {source}: {e}")
    
    print(f"\n✅ Moved {moved_count} items")
    
    # Create main entry points in root
    create_main_scripts()
    
    # Create documentation
    create_documentation()
    
    # Clean up empty directories
    cleanup_empty_dirs()
    
    print(f"\n🎉 PROJECT CLEANUP COMPLETE!")
    print("=" * 50)
    print("📁 Clean project structure created")
    print("📋 Documentation generated") 
    print("🚀 Ready for professional use")


def create_main_scripts():
    """Create main entry point scripts in root directory"""
    
    # Main demo script
    main_demo = '''#!/usr/bin/env python3
"""
Neural Decoder - Main Demo
Easy entry point to run different demonstrations
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "demos"))
sys.path.append(str(Path(__file__).parent / "data"))
sys.path.append(str(Path(__file__).parent / "models"))

def main():
    print("🧠 NEURAL DECODER - MAIN DEMO")
    print("=" * 40)
    print("Choose a demonstration:")
    print("1. Quick Demo (3 min)")
    print("2. Realistic EEG Demo (5 min)")  
    print("3. Real PhysioNet Data (10 min)")
    print("4. Face Control Demo")
    print("5. Full Pipeline (30+ min)")
    print("0. Exit")
    
    choice = input("\\nEnter choice (0-5): ").strip()
    
    if choice == "1":
        print("\\n🚀 Running Quick Demo...")
        from demos.quick_demo import quick_demo
        quick_demo()
    elif choice == "2":
        print("\\n🧬 Running Realistic EEG Demo...")
        from demos.realistic_demo import realistic_demo
        realistic_demo()
    elif choice == "3":
        print("\\n🧠 Running Real PhysioNet Data Demo...")
        from demos.load_physionet_data import main as physionet_main
        physionet_main()
    elif choice == "4":
        print("\\n📹 Running Face Control Demo...")
        import subprocess
        subprocess.run([sys.executable, "applications/examples/face_controlled_mouse.py"])
    elif choice == "5":
        print("\\n⚡ Running Full Pipeline...")
        from demos.run_complete_pipeline import run_complete_pipeline
        run_complete_pipeline()
    elif choice == "0":
        print("Goodbye! 👋")
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
'''
    
    with open("neural_decoder_demo.py", "w") as f:
        f.write(main_demo)
    
    print("📄 Created: neural_decoder_demo.py - Main entry point")


def create_documentation():
    """Create comprehensive documentation"""
    
    # Project overview
    overview = '''# 🧠 Neural Decoder Project Structure

## 📁 Directory Organization

### Core Directories
- `demos/` - Main demonstration scripts
- `models/` - Neural network architectures  
- `data/` - Data generation and loading
- `training/` - Training and evaluation
- `analysis/` - Analysis and visualization
- `applications/` - Real-world applications
- `api/` - FastAPI web service

### Outputs
- `outputs/` - Generated results and visualizations

### Support
- `docs/` - Documentation and examples
- `archive/` - Old files and utilities

## 🚀 Quick Start

### Run Main Demo
```bash
python neural_decoder_demo.py
```

### Individual Demos
```bash
python demos/quick_demo.py           # 3 min - synthetic data
python demos/realistic_demo.py       # 5 min - realistic patterns  
python demos/load_physionet_data.py  # 10 min - real brain data
```

### Applications
```bash
python applications/examples/face_controlled_mouse.py  # Face control
python applications/examples/adaptive_control_demo.py  # Adaptive BCI
```

## 📊 What Each Demo Does

### 1. Quick Demo (`demos/quick_demo.py`)
- **Time**: 3 minutes
- **Data**: Simple synthetic patterns
- **Accuracy**: 60-80%
- **Purpose**: Fast validation

### 2. Realistic Demo (`demos/realistic_demo.py`) 
- **Time**: 5 minutes
- **Data**: Neuroscience-based synthetic (ERD patterns)
- **Accuracy**: 75-90%
- **Purpose**: Realistic neural patterns

### 3. PhysioNet Demo (`demos/load_physionet_data.py`)
- **Time**: 10 minutes (first run downloads data)
- **Data**: Real human brain recordings (109 subjects)
- **Accuracy**: 60-75% (realistic for real data)
- **Purpose**: True neuroscience research

### 4. Face Control (`applications/examples/face_controlled_mouse.py`)
- **Purpose**: Control computer with facial expressions
- **Requirements**: Webcam
- **Features**: Head tracking, eye blinks, mouth gestures

## 🏗️ Architecture

### Models Available
- **EEGNet**: State-of-the-art CNN for EEG
- **Transformer**: Attention-based neural decoder
- **CNN+LSTM**: Hybrid spatiotemporal model
- **Multi-scale CNN**: Multiple temporal scales

### Data Types Supported
- **Motor Imagery**: Left/right hand, feet movement thoughts
- **Real EEG**: PhysioNet database integration
- **Synthetic**: Realistic neural pattern generation
- **Custom**: Face tracking, gesture recognition

## 📈 Performance Benchmarks

| Demo | Data Type | Expected Accuracy | Training Time |
|------|-----------|-------------------|---------------|
| Quick | Synthetic | 60-80% | 2-3 min |
| Realistic | Neuro-synthetic | 75-90% | 3-5 min |
| PhysioNet | Real human EEG | 60-75% | 5-10 min |
| Full Pipeline | Multiple models | 70-85% | 30-60 min |

## 🔧 Installation

```bash
git clone https://github.com/Dbrooks2/Neural-Decoder.git
cd Neural-Decoder
pip install -r requirements.txt
```

### Optional Dependencies
```bash
# For real EEG data
pip install mne

# For face tracking
pip install opencv-python mediapipe pyautogui

# For advanced features  
pip install scipy scikit-learn matplotlib seaborn
```

## 🎯 Use Cases

### Research
- EEG signal analysis
- BCI algorithm development
- Motor imagery classification
- Neural pattern recognition

### Applications  
- Assistive technology
- Gaming interfaces
- Medical rehabilitation
- Human-computer interaction

### Education
- Neuroscience demonstrations
- Machine learning examples
- Signal processing tutorials
- BCI concept illustration

---

**⭐ This is a complete brain-computer interface research platform!**
'''
    
    with open("docs/PROJECT_OVERVIEW.md", "w") as f:
        f.write(overview)
    
    print("📄 Created: docs/PROJECT_OVERVIEW.md - Complete documentation")


def cleanup_empty_dirs():
    """Remove empty directories"""
    empty_dirs = []
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if not any(dir_path.iterdir()):
                    empty_dirs.append(dir_path)
            except:
                pass
    
    for dir_path in empty_dirs:
        try:
            if str(dir_path) not in ['.git', '__pycache__', '.venv']:
                dir_path.rmdir()
                print(f"🗑️  Removed empty directory: {dir_path}")
        except:
            pass


if __name__ == "__main__":
    cleanup_neural_decoder()
