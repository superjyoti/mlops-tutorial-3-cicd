import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os


def get_mnist_subset(num_samples=20, batch_size=4, data_dir='./data'):
    """
    Load MNIST dataset and return a subset for quick experimentation
    
    Args:
        num_samples: Number of training samples to use (supports 20 or 40)
        batch_size: Batch size for data loader
        data_dir: Directory to store MNIST data
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Create subset for quick training
    train_subset = Subset(train_dataset, range(num_samples))
    test_subset = Subset(test_dataset, range(min(10, len(test_dataset))))  # 10 test samples
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Dataset information
    dataset_info = {
        'train_samples': len(train_subset),
        'test_samples': len(test_subset),
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'batch_size': batch_size
    }
    
    return train_loader, test_loader, dataset_info


def get_sample_batch(train_loader):
    """Get a single batch for testing"""
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels


if __name__ == "__main__":
    # Test the data loader
    train_loader, test_loader, info = get_mnist_subset(num_samples=20)
    
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test loading a batch
    images, labels = get_sample_batch(train_loader)
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Label range: {labels.min().item()} - {labels.max().item()}")