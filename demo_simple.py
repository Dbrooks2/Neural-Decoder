"""
Simple demo of the Neural Decoder without web dependencies.
Simulates real-time cursor control in the terminal.
"""

import os
import sys
import time
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neural_decoder.models import build_model
from neural_decoder.data import DataConfig, SyntheticNeuralDataset


def simulate_realtime_demo():
    # Load model
    model = build_model(32, 64)
    model.eval()
    
    # Load trained weights if available
    model_path = "artifacts/model.pt"
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        print("✓ Loaded trained model")
    else:
        print("! Using random initialization (train first for better results)")
    
    # Load normalization stats
    norm_path = "artifacts/normalizer.npz"
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        mean = torch.from_numpy(norm_data["mean"]).view(1, -1, 1)
        std = torch.from_numpy(norm_data["std"]).view(1, -1, 1)
        print("✓ Loaded normalization stats")
    else:
        mean = torch.zeros(1, 32, 1)
        std = torch.ones(1, 32, 1)
        print("! Using default normalization")
    
    print("\n" + "="*50)
    print("NEURAL DECODER DEMO - Terminal Version")
    print("="*50)
    print("Simulating cursor control from neural signals...")
    print("Press Ctrl+C to stop\n")
    
    # Initialize cursor position
    x, y = 40, 12  # center of 80x24 terminal
    
    # Simulate streaming neural data
    rng = np.random.default_rng(42)
    signal_buffer = rng.standard_normal((32, 64)).astype(np.float32)
    
    try:
        while True:
            # Simulate new neural data (shift buffer and add new sample)
            signal_buffer[:, :-1] = signal_buffer[:, 1:]
            signal_buffer[:, -1] = rng.standard_normal(32) * 0.5 + signal_buffer[:, -2] * 0.8
            
            # Prepare input
            x_tensor = torch.from_numpy(signal_buffer).unsqueeze(0)
            x_tensor = (x_tensor - mean) / std
            
            # Inference
            start = time.perf_counter()
            with torch.inference_mode():
                velocity = model(x_tensor).squeeze().numpy()
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Update position
            x += velocity[0] * 2
            y += velocity[1] * 2
            x = max(0, min(79, x))
            y = max(0, min(23, y))
            
            # Clear screen and draw
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Draw border
            print("┌" + "─"*78 + "┐")
            for row in range(22):
                line = "│" + " "*78 + "│"
                if int(y) == row:
                    line = line[:int(x)+1] + "●" + line[int(x)+2:]
                print(line)
            print("└" + "─"*78 + "┘")
            
            print(f"\nVelocity: dx={velocity[0]:+.3f}, dy={velocity[1]:+.3f}")
            print(f"Position: ({int(x)}, {int(y)})")
            print(f"Latency: {latency_ms:.1f} ms")
            
            time.sleep(0.05)  # 20 Hz update
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")


if __name__ == "__main__":
    simulate_realtime_demo()
