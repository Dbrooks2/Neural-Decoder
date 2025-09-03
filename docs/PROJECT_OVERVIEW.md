# Neural Decoder Project Structure

## Directory Organization

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

## Quick Start

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

## What Each Demo Does

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

## Architecture

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

## Performance Benchmarks

| Demo | Data Type | Expected Accuracy | Training Time |
|------|-----------|-------------------|---------------|
| Quick | Synthetic | 60-80% | 2-3 min |
| Realistic | Neuro-synthetic | 75-90% | 3-5 min |
| PhysioNet | Real human EEG | 60-75% | 5-10 min |
| Full Pipeline | Multiple models | 70-85% | 30-60 min |

## Installation

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

## Use Cases

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

**This is a complete brain-computer interface research platform!**
