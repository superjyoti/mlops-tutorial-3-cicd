import pytest
import torch
import torch.nn as nn
from model import SimpleCNN


class TestSimpleCNN:
    """Test suite for SimpleCNN model"""
    
    def test_model_initialization(self):
        """Test model can be initialized correctly"""
        model = SimpleCNN()
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
    
    def test_model_forward_pass(self):
        """Test model forward pass with correct input/output shapes"""
        model = SimpleCNN()
        
        # Test with single sample
        batch_size = 1
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)  # 10 classes
        assert output.dtype == torch.float32
    
    def test_model_forward_pass_batch(self):
        """Test model forward pass with batch input"""
        model = SimpleCNN()
        
        # Test with batch
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert output.dtype == torch.float32
    
    def test_model_output_properties(self):
        """Test model output properties (log softmax)"""
        model = SimpleCNN()
        model.eval()  # Set to evaluation mode
        
        input_tensor = torch.randn(2, 1, 28, 28)
        output = model(input_tensor)
        
        # Check that output is log probabilities (should be negative)
        assert torch.all(output <= 0)
        
        # Check that exp(output) sums to approximately 1 (softmax property)
        probs = torch.exp(output)
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = SimpleCNN()
        
        input_tensor = torch.randn(2, 1, 28, 28, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)  # At least some gradients should be non-zero
    
    def test_model_info_function(self):
        """Test model info function returns correct information"""
        model = SimpleCNN()
        info = model.get_model_info()
        
        # Check required keys
        required_keys = ['total_parameters', 'trainable_parameters', 'model_size_mb', 'architecture']
        for key in required_keys:
            assert key in info
        
        # Check values are reasonable
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['trainable_parameters'] <= info['total_parameters']
        assert info['model_size_mb'] > 0
        assert info['architecture'] == 'SimpleCNN'
    
    def test_model_parameter_count(self):
        """Test that parameter count is reasonable"""
        model = SimpleCNN()
        info = model.get_model_info()
        
        # SimpleCNN should have reasonable number of parameters
        total_params = info['total_parameters']
        assert 50000 < total_params < 500000  # Reasonable range for simple CNN
    
    def test_model_different_input_sizes(self):
        """Test model behavior with different batch sizes"""
        model = SimpleCNN()
        
        # Test various batch sizes
        for batch_size in [1, 3, 8, 16]:
            input_tensor = torch.randn(batch_size, 1, 28, 28)
            output = model(input_tensor)
            assert output.shape == (batch_size, 10)
    
    def test_model_device_compatibility(self):
        """Test model works on different devices"""
        model = SimpleCNN()
        
        # Test CPU
        input_cpu = torch.randn(2, 1, 28, 28)
        output_cpu = model(input_cpu)
        assert output_cpu.device == torch.device('cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_gpu = input_cpu.cuda()
            output_gpu = model_gpu(input_gpu)
            assert output_gpu.device.type == 'cuda'
    
    def test_model_eval_vs_train_mode(self):
        """Test model behavior in train vs eval mode"""
        model = SimpleCNN()
        input_tensor = torch.randn(4, 1, 28, 28)
        
        # Train mode
        model.train()
        assert model.training == True
        output_train = model(input_tensor)
        
        # Eval mode
        model.eval()
        assert model.training == False
        output_eval = model(input_tensor)
        
        # Outputs should have same shape
        assert output_train.shape == output_eval.shape
        
        # Due to dropout, outputs might be different
        # But this is expected behavior
    
    def test_model_weight_initialization(self):
        """Test that model weights are properly initialized"""
        model = SimpleCNN()
        
        # Check that weights are not all zeros or all the same
        conv1_weights = model.conv1.weight.data
        assert not torch.all(conv1_weights == 0)
        assert not torch.all(conv1_weights == conv1_weights[0, 0, 0, 0])
        
        fc1_weights = model.fc1.weight.data
        assert not torch.all(fc1_weights == 0)
        assert not torch.all(fc1_weights == fc1_weights[0, 0])
    
    def test_model_num_classes_parameter(self):
        """Test model with different number of classes"""
        # Test with default 10 classes
        model_10 = SimpleCNN(num_classes=10)
        output_10 = model_10(torch.randn(1, 1, 28, 28))
        assert output_10.shape[1] == 10
        
        # Test with different number of classes
        model_5 = SimpleCNN(num_classes=5)
        output_5 = model_5(torch.randn(1, 1, 28, 28))
        assert output_5.shape[1] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])