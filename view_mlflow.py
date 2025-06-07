#!/usr/bin/env python
"""
Simple script to view MLflow results and demonstrate model registry functionality
"""

import mlflow
import mlflow.pytorch
from model import SimpleCNN
import torch

def view_experiments():
    """View all experiments and runs"""
    print("MLflow Experiments and Runs:")
    print("=" * 50)
    
    # List all experiments
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        
        # Get runs for this experiment
        runs = mlflow.search_runs(exp.experiment_id)
        
        if len(runs) > 0:
            print(f"Number of runs: {len(runs)}")
            print("\nRecent runs:")
            
            for idx, run in runs.head(3).iterrows():
                print(f"  Run ID: {run['run_id'][:8]}...")
                print(f"  Status: {run['status']}")
                print(f"  Train Accuracy: {run.get('metrics.train_accuracy', 'N/A')}")
                print(f"  Test Accuracy: {run.get('metrics.test_accuracy', 'N/A')}")
                print(f"  Epochs: {run.get('params.epochs', 'N/A')}")
                print(f"  Learning Rate: {run.get('params.learning_rate', 'N/A')}")
                print()
        else:
            print("  No runs found")

def view_registered_models():
    """View registered models in MLflow Model Registry"""
    print("\nMLflow Model Registry:")
    print("=" * 50)
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        registered_models = client.search_registered_models()
        
        if not registered_models:
            print("No registered models found")
            return
        
        for model in registered_models:
            print(f"\nModel: {model.name}")
            print(f"Description: {model.description or 'No description'}")
            
            # Get model versions
            versions = client.search_model_versions(f"name='{model.name}'")
            print(f"Versions: {len(versions)}")
            
            for version in versions:
                print(f"  Version {version.version}:")
                print(f"    Stage: {version.current_stage}")
                print(f"    Status: {version.status}")
                print(f"    Run ID: {version.run_id[:8]}...")
                print(f"    Source: {version.source}")
                
    except Exception as e:
        print(f"Error accessing model registry: {e}")

def load_and_test_model():
    """Load model from MLflow and test it"""
    print("\nTesting Model Loading from MLflow:")
    print("=" * 50)
    
    try:
        # Load latest version of the model
        model_name = "mnist_simple_cnn"
        model_version = 1
        
        model_uri = f"models:/{model_name}/{model_version}"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        
        print(f"Successfully loaded model: {model_name} version {model_version}")
        
        # Test with dummy data
        test_input = torch.randn(1, 1, 28, 28)
        
        loaded_model.eval()
        with torch.no_grad():
            output = loaded_model(test_input)
            predicted_class = output.argmax(dim=1).item()
            confidence = torch.exp(output).max().item()
        
        print(f"Test prediction:")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

def compare_runs():
    """Compare different runs"""
    print("\nRun Comparison:")
    print("=" * 50)
    
    try:
        # Get all runs from the experiment
        experiment = mlflow.get_experiment_by_name("mnist_simple_cnn")
        if experiment:
            runs = mlflow.search_runs(experiment.experiment_id)
            
            if len(runs) > 0:
                print("Run Performance Summary:")
                print(f"{'Run ID':<10} {'Train Acc':<10} {'Test Acc':<10} {'Epochs':<8} {'LR':<8}")
                print("-" * 50)
                
                for idx, run in runs.iterrows():
                    run_id = run['run_id'][:8]
                    train_acc = f"{run.get('metrics.train_accuracy', 0):.1f}%"
                    test_acc = f"{run.get('metrics.test_accuracy', 0):.1f}%"
                    epochs = run.get('params.epochs', 'N/A')
                    lr = run.get('params.learning_rate', 'N/A')
                    
                    print(f"{run_id:<10} {train_acc:<10} {test_acc:<10} {epochs:<8} {lr:<8}")
            else:
                print("No runs found for comparison")
        else:
            print("Experiment 'mnist_simple_cnn' not found")
            
    except Exception as e:
        print(f"Error comparing runs: {e}")

if __name__ == "__main__":
    print("Tutorial 1: MLflow Results Viewer")
    print("=" * 50)
    
    # View experiments and runs
    view_experiments()
    
    # View registered models
    view_registered_models()
    
    # Load and test model
    load_and_test_model()
    
    # Compare runs
    compare_runs()
    
    print("\n" + "=" * 50)
    print("To start MLflow UI, run: mlflow ui")
    print("Then open: http://localhost:5000")