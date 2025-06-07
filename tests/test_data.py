import pytest
import torch
import os
import tempfile
import shutil
from data_loader import get_mnist_subset, get_sample_batch


class TestDataLoader:
    """Test suite for data loading functionality"""
    
    def test_mnist_subset_creation(self):
        """Test that MNIST subset is created correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader, dataset_info = get_mnist_subset(
                num_samples=10, 
                batch_size=2,
                data_dir=temp_dir
            )
            
            # Check dataset info
            assert dataset_info['train_samples'] == 10
            assert dataset_info['test_samples'] == 10
            assert dataset_info['num_classes'] == 10
            assert dataset_info['input_shape'] == (1, 28, 28)
            assert dataset_info['batch_size'] == 2
    
    def test_data_loader_batches(self):
        """Test data loader batch properties"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader, _ = get_mnist_subset(
                num_samples=8, 
                batch_size=4,
                data_dir=temp_dir
            )
            
            # Test training loader
            batch_count = 0
            for images, labels in train_loader:
                batch_count += 1
                assert images.shape[0] <= 4  # Batch size
                assert images.shape[1:] == (1, 28, 28)  # Image dimensions
                assert labels.shape[0] == images.shape[0]  # Same batch size
                assert torch.all(labels >= 0) and torch.all(labels < 10)  # Valid labels
            
            assert batch_count == 2  # 8 samples / 4 batch_size = 2 batches
    
    def test_sample_batch_function(self):
        """Test get_sample_batch function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, _, _ = get_mnist_subset(
                num_samples=6, 
                batch_size=3,
                data_dir=temp_dir
            )
            
            images, labels = get_sample_batch(train_loader)
            
            assert images.shape == (3, 1, 28, 28)
            assert labels.shape == (3,)
            assert torch.all(labels >= 0) and torch.all(labels < 10)
    
    def test_data_normalization(self):
        """Test that data is properly normalized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, _, _ = get_mnist_subset(
                num_samples=4, 
                batch_size=2,
                data_dir=temp_dir
            )
            
            images, _ = get_sample_batch(train_loader)
            
            # Check that images are normalized (approximately)
            # MNIST normalization: mean=0.1307, std=0.3081
            mean = images.mean().item()
            std = images.std().item()
            
            # Should be approximately zero-centered with reasonable std
            assert abs(mean) < 1.0  # Not exact due to small sample size
            assert 0.1 < std < 2.0   # Reasonable std deviation
    
    def test_data_loader_reproducibility(self):
        """Test data loader with different parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with minimum samples
            train_loader1, test_loader1, info1 = get_mnist_subset(
                num_samples=2, 
                batch_size=1,
                data_dir=temp_dir
            )
            
            assert info1['train_samples'] == 2
            assert info1['batch_size'] == 1
            
            # Test with larger batch than samples
            train_loader2, test_loader2, info2 = get_mnist_subset(
                num_samples=3, 
                batch_size=5,  # Larger than num_samples
                data_dir=temp_dir
            )
            
            assert info2['train_samples'] == 3
            
            # Should still work, just with smaller actual batch size
            images, labels = get_sample_batch(train_loader2)
            assert images.shape[0] == 3  # All samples in one batch
    
    def test_data_types_and_device(self):
        """Test data types and tensor properties"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, _, _ = get_mnist_subset(
                num_samples=4, 
                batch_size=2,
                data_dir=temp_dir
            )
            
            images, labels = get_sample_batch(train_loader)
            
            # Check data types
            assert images.dtype == torch.float32
            assert labels.dtype == torch.int64
            
            # Check value ranges
            assert images.min() >= -3.0  # Normalized values shouldn't be too extreme
            assert images.max() <= 3.0
            assert labels.min() >= 0
            assert labels.max() < 10
    
    def test_data_directory_creation(self):
        """Test that data directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "new_data_dir")
            assert not os.path.exists(data_path)
            
            train_loader, _, _ = get_mnist_subset(
                num_samples=2, 
                data_dir=data_path
            )
            
            assert os.path.exists(data_path)
            assert os.path.exists(os.path.join(data_path, "MNIST"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])