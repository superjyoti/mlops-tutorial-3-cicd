import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
from datetime import datetime
import numpy as np

from model import SimpleCNN
from data_loader import get_mnist_subset


def load_processed_data():
    """Load preprocessed data if available, otherwise use data loader"""
    processed_train = './data/processed/train.pt'
    processed_test = './data/processed/test.pt'
    
    if os.path.exists(processed_train) and os.path.exists(processed_test):
        print("Loading preprocessed data...")
        train_data_dict = torch.load(processed_train)
        test_data_dict = torch.load(processed_test)
        
        train_data = train_data_dict['data']
        train_labels = train_data_dict['labels']
        test_data = test_data_dict['data']
        test_labels = test_data_dict['labels']
        
        return train_data, train_labels, test_data, test_labels, len(train_data)
    else:
        return None, None, None, None, None


def train_model(num_samples=None, epochs=5, learning_rate=0.001, batch_size=4):
    """
    Train SimpleCNN with MLflow tracking
    
    Args:
        num_samples: Number of training samples (auto-detected from processed data)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
    """
    
    # Try to load preprocessed data first (DVC tracked)
    train_data, train_labels, test_data, test_labels, detected_samples = load_processed_data()
    
    if train_data is not None:
        # Use preprocessed data
        if num_samples is None:
            num_samples = detected_samples
        print(f"Using preprocessed data with {num_samples} samples")
        use_processed = True
    else:
        # Fall back to data loader
        if num_samples is None:
            num_samples = 20
        print(f"No preprocessed data found, using data loader with {num_samples} samples")
        use_processed = False
    
    # Set up MLflow
    mlflow.set_experiment("mnist_dvc_tutorial")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "num_samples": num_samples,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "loss_function": "NLLLoss",
            "data_source": "processed" if use_processed else "loader"
        })
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", str(device))
        
        if use_processed:
            # Create data loaders from preprocessed data
            from torch.utils.data import TensorDataset, DataLoader
            
            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            dataset_info = {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'num_classes': 10,
                'input_shape': (1, 28, 28),
                'batch_size': batch_size
            }
        else:
            # Load data using data loader
            train_loader, test_loader, dataset_info = get_mnist_subset(
                num_samples=num_samples, 
                batch_size=batch_size
            )
        
        # Log dataset info
        mlflow.log_params(dataset_info)
        
        # Initialize model
        model = SimpleCNN().to(device)
        model_info = model.get_model_info()
        mlflow.log_params(model_info)
        
        print(f"Model created with {model_info['total_parameters']} parameters")
        
        # Loss and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        train_losses = []
        train_accuracies = []
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Log batch metrics
                if batch_idx % 2 == 0:  # Log every 2 batches
                    batch_acc = 100. * correct / total
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, '
                          f'Loss: {loss.item():.4f}, Accuracy: {batch_acc:.2f}%')
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc
            }, step=epoch)
            
            print(f'Epoch {epoch+1}/{epochs} Summary: '
                  f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Evaluation on test set
        print("Evaluating on test set...")
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_loss /= len(test_loader)
        test_accuracy = 100. * test_correct / test_total
        
        # Log final test metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "final_train_loss": train_losses[-1],
            "final_train_accuracy": train_accuracies[-1]
        })
        
        print(f'Test Results: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_path = "models/mnist_cnn.pth"
        torch.save(model.state_dict(), model_path)
        
        # Log model as artifact
        mlflow.log_artifact(model_path, "model")
        
        # Register model in MLflow Model Registry
        model_name = "mnist_simple_cnn"
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="pytorch_model",
            registered_model_name=model_name,
            signature=mlflow.models.signature.infer_signature(
                data.cpu().numpy(), 
                output.cpu().numpy()
            )
        )
        
        # Log training summary
        summary = {
            "training_completed": True,
            "timestamp": datetime.now().isoformat(),
            "samples_used": num_samples,
            "epochs_completed": epochs,
            "best_train_accuracy": max(train_accuracies),
            "final_test_accuracy": test_accuracy
        }
        
        mlflow.log_dict(summary, "training_summary.json")
        
        print(f"\nTraining completed!")
        print(f"Model saved to: {model_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Model registered as: {model_name}")
        
        return model, {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }


if __name__ == "__main__":
    # Run training with auto-detected parameters
    print("Tutorial 2: DVC + MLflow Integration")
    print("=" * 50)
    
    model, metrics = train_model(
        epochs=5,
        learning_rate=0.001,
        batch_size=4
    )
    
    print("\nTo view results in MLflow UI, run:")
    print("mlflow ui")
    print("Then open: http://localhost:5000")