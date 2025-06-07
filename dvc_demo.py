#!/usr/bin/env python
"""
DVC Tutorial 2 Demonstration Script
Shows complete DVC workflow with data versioning and rollback
"""

import subprocess
import os
import sys
import time


def run_command(cmd, description=""):
    """Run shell command and print output"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("Error:")
            print(result.stderr)
            return False
        
        return True
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def check_file_exists(filepath, description=""):
    """Check if file exists and show info"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} not found")
        return False


def main():
    """Complete DVC demonstration workflow"""
    
    print("DVC Tutorial 2: Complete Data Versioning Demonstration")
    print("=" * 80)
    print("This script demonstrates:")
    print("1. DVC initialization")
    print("2. Data preparation and versioning")
    print("3. Training with different data versions")
    print("4. Data rollback and re-training")
    print("=" * 80)
    
    # Step 1: Initialize Git and DVC
    print("\n🚀 PHASE 1: Setup")
    
    if not run_command("git init", "Initialize Git repository"):
        return False
    
    if not run_command("dvc init --no-scm", "Initialize DVC"):
        return False
    
    # Set up local DVC remote
    os.makedirs("../dvc-storage", exist_ok=True)
    if not run_command("dvc remote add -d myremote ../dvc-storage", "Add local DVC remote"):
        return False
    
    # Step 2: Create initial dataset (20 samples)
    print("\n📊 PHASE 2: Initial Dataset (20 samples)")
    
    if not run_command("python prepare_expanded_data.py --initial", "Create initial 20-sample dataset"):
        return False
    
    check_file_exists("data/processed/train.pt", "Training data")
    check_file_exists("data/processed/test.pt", "Test data")
    
    # Step 3: Train with initial dataset
    print("\n🏋️ PHASE 3: Train with 20 samples")
    
    if not run_command("python train.py", "Train model with 20 samples"):
        return False
    
    check_file_exists("models/mnist_cnn.pth", "Trained model")
    
    # Step 4: Track data with DVC
    print("\n📦 PHASE 4: Track data with DVC")
    
    if not run_command("dvc add data/processed/train.pt", "Track training data"):
        return False
    
    if not run_command("dvc add data/processed/test.pt", "Track test data"):
        return False
    
    # Commit to Git
    if not run_command("git add data/processed/train.pt.dvc data/processed/test.pt.dvc .dvc/config", "Stage DVC files"):
        return False
    
    if not run_command('git commit -m "Add 20-sample dataset v1"', "Commit version 1"):
        return False
    
    print("\n✅ Version 1 (20 samples) committed!")
    
    # Step 5: Create expanded dataset (40 samples)
    print("\n📈 PHASE 5: Expanded Dataset (40 samples)")
    
    if not run_command("python prepare_expanded_data.py", "Create expanded 40-sample dataset"):
        return False
    
    # Train with expanded dataset
    print("\n🏋️ PHASE 6: Train with 40 samples")
    
    if not run_command("python train.py", "Train model with 40 samples"):
        return False
    
    # Track new data version
    if not run_command("dvc add data/processed/train.pt", "Track updated training data"):
        return False
    
    if not run_command("dvc add data/processed/test.pt", "Track updated test data"):
        return False
    
    # Commit version 2
    if not run_command("git add data/processed/train.pt.dvc data/processed/test.pt.dvc", "Stage updated DVC files"):
        return False
    
    if not run_command('git commit -m "Add 40-sample dataset v2"', "Commit version 2"):
        return False
    
    print("\n✅ Version 2 (40 samples) committed!")
    
    # Step 6: Demonstrate rollback
    print("\n⏪ PHASE 7: Demonstrate Data Rollback")
    
    print("\nCurrent state (40 samples):")
    run_command("python -c \"import torch; data=torch.load('data/processed/train.pt'); print(f'Samples: {len(data[\\\"data\\\"])}')\"", "Check current data size")
    
    # Rollback to version 1
    if not run_command("git checkout HEAD~1 -- data/processed/train.pt.dvc data/processed/test.pt.dvc", "Rollback DVC files to v1"):
        return False
    
    if not run_command("dvc checkout", "Checkout data version 1"):
        return False
    
    print("\nAfter rollback (should be 20 samples):")
    run_command("python -c \"import torch; data=torch.load('data/processed/train.pt'); print(f'Samples: {len(data[\\\"data\\\"])}')\"", "Check rolled back data size")
    
    # Train with rolled back data
    print("\n🏋️ PHASE 8: Train with rolled back data (20 samples)")
    
    if not run_command("python train.py", "Train model with rolled back data"):
        return False
    
    # Return to latest version
    print("\n⏩ PHASE 9: Return to latest version")
    
    if not run_command("git checkout HEAD -- data/processed/train.pt.dvc data/processed/test.pt.dvc", "Return to latest DVC files"):
        return False
    
    if not run_command("dvc checkout", "Checkout latest data version"):
        return False
    
    print("\nBack to latest (should be 40 samples):")
    run_command("python -c \"import torch; data=torch.load('data/processed/train.pt'); print(f'Samples: {len(data[\\\"data\\\"])}')\"", "Check restored data size")
    
    # Final summary
    print("\n🎉 SUMMARY")
    print("=" * 60)
    print("✅ DVC initialization complete")
    print("✅ Data versioning demonstrated")
    print("✅ Model training with different data versions")
    print("✅ Data rollback and restore demonstrated")
    print("✅ MLflow tracking throughout all experiments")
    
    print("\n📊 Check your results:")
    print("- MLflow UI: run 'mlflow ui' and open http://localhost:5000")
    print("- Git log: run 'git log --oneline'")
    print("- DVC status: run 'dvc status'")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 DVC Tutorial 2 completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tutorial failed. Check the error messages above.")
        sys.exit(1)