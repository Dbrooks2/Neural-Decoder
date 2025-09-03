#!/usr/bin/env python3
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
    print("NEURAL DECODER - MAIN DEMO")
    print("=" * 40)
    print("Choose a demonstration:")
    print("1. Quick Demo (3 min)")
    print("2. Realistic EEG Demo (5 min)")  
    print("3. Real PhysioNet Data (10 min)")
    print("4. Face Control Demo")
    print("5. Full Pipeline (30+ min)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == "1":
        print("\nRunning Quick Demo...")
        from demos.quick_demo import quick_demo
        quick_demo()
    elif choice == "2":
        print("\nRunning Realistic EEG Demo...")
        from demos.realistic_demo import realistic_demo
        realistic_demo()
    elif choice == "3":
        print("\nRunning Real PhysioNet Data Demo...")
        from demos.load_physionet_data import main as physionet_main
        physionet_main()
    elif choice == "4":
        print("\nRunning Face Control Demo...")
        import subprocess
        subprocess.run([sys.executable, "applications/examples/face_controlled_mouse.py"])
    elif choice == "5":
        print("\nRunning Full Pipeline...")
        from demos.run_complete_pipeline import run_complete_pipeline
        run_complete_pipeline()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
