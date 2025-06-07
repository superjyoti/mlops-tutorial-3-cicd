#!/usr/bin/env python
"""
Script to prepare expanded dataset (40 samples) for DVC versioning demonstration
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import os


def create_expanded_dataset(save_dir='./data/processed'):
    """
    Create expanded dataset with 40 samples and save for DVC tracking
    """
    print("Creating expanded dataset with 40 samples...")
    
    # Create processed data directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Create expanded training subset (40 samples)
    train_subset = Subset(train_dataset, range(40))
    test_subset = Subset(test_dataset, range(20))  # Also expand test set
    
    # Extract data and labels
    train_data = []
    train_labels = []
    
    for i in range(len(train_subset)):
        data, label = train_subset[i]
        train_data.append(data)
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    
    for i in range(len(test_subset)):
        data, label = test_subset[i]
        test_data.append(data)
        test_labels.append(label)
    
    # Convert to tensors
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)
    
    # Save processed data
    train_file = os.path.join(save_dir, 'train.pt')
    test_file = os.path.join(save_dir, 'test.pt')
    
    torch.save({
        'data': train_data,
        'labels': train_labels
    }, train_file)
    
    torch.save({
        'data': test_data,
        'labels': test_labels
    }, test_file)
    
    print(f"Expanded dataset created:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Training data saved to: {train_file}")
    print(f"  Test data saved to: {test_file}")
    
    # Display some statistics
    print(f"\nDataset Statistics:")
    print(f"  Training data shape: {train_data.shape}")
    print(f"  Test data shape: {test_data.shape}")
    print(f"  Label distribution (train): {torch.bincount(train_labels).tolist()}")
    print(f"  Label distribution (test): {torch.bincount(test_labels).tolist()}")
    
    return train_file, test_file


def create_initial_dataset(save_dir='./data/processed'):
    """
    Create initial dataset with 20 samples (same as Tutorial 1)
    """
    print("Creating initial dataset with 20 samples...")
    
    # Create processed data directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Create initial training subset (20 samples)
    train_subset = Subset(train_dataset, range(20))
    test_subset = Subset(test_dataset, range(10))
    
    # Extract data and labels
    train_data = []
    train_labels = []
    
    for i in range(len(train_subset)):
        data, label = train_subset[i]
        train_data.append(data)
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    
    for i in range(len(test_subset)):
        data, label = test_subset[i]
        test_data.append(data)
        test_labels.append(label)
    
    # Convert to tensors
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)
    
    # Save processed data
    train_file = os.path.join(save_dir, 'train.pt')
    test_file = os.path.join(save_dir, 'test.pt')
    
    torch.save({
        'data': train_data,
        'labels': train_labels
    }, train_file)
    
    torch.save({
        'data': test_data,
        'labels': test_labels
    }, test_file)
    
    print(f"Initial dataset created:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Training data saved to: {train_file}")
    print(f"  Test data saved to: {test_file}")
    
    return train_file, test_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--initial":
        print("DVC Tutorial 2: Creating Initial Dataset (20 samples)")
        print("=" * 60)
        create_initial_dataset()
    else:
        print("DVC Tutorial 2: Creating Expanded Dataset (40 samples)")
        print("=" * 60)
        create_expanded_dataset()
    
    print("\nDataset preparation complete!")
    print("Ready for DVC tracking.")