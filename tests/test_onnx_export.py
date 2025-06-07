import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch

from model import SimpleCNN
from export_onnx import export_to_onnx, validate_onnx_model


class TestONNXExport:
    """Test suite for ONNX export functionality"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory with a trained model"""
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'models')
        os.makedirs(model_dir)
        
        # Create and save a simple model
        model = SimpleCNN()
        model_path = os.path.join(model_dir, 'mnist_cnn.pth')
        torch.save(model.state_dict(), model_path)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_model_exists_for_export(self, temp_model_dir):
        """Test that export requires existing PyTorch model"""
        model_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.pth')
        onnx_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.onnx')
        
        # Should work with existing model
        metadata = export_to_onnx(model_path, onnx_path)
        assert metadata['export_successful'] == True
        assert os.path.exists(onnx_path)
        
        # Should fail with non-existent model
        with pytest.raises(FileNotFoundError):
            export_to_onnx('nonexistent_model.pth', onnx_path)
    
    def test_onnx_export_metadata(self, temp_model_dir):
        """Test that export returns correct metadata"""
        model_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.pth')
        onnx_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.onnx')
        
        metadata = export_to_onnx(model_path, onnx_path)
        
        # Check required metadata fields
        required_fields = [
            'pytorch_model_path', 'onnx_model_path', 'pytorch_size_mb',
            'onnx_size_mb', 'max_inference_diff', 'opset_version',
            'export_successful', 'input_shape', 'output_shape'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Check metadata values
        assert metadata['pytorch_model_path'] == model_path
        assert metadata['onnx_model_path'] == onnx_path
        assert metadata['opset_version'] == 11
        assert metadata['export_successful'] == True
        assert metadata['input_shape'] == [1, 1, 28, 28]
        assert metadata['output_shape'] == [1, 10]
        assert metadata['max_inference_diff'] < 1e-5
    
    def test_onnx_model_inference_accuracy(self, temp_model_dir):
        """Test that ONNX model produces similar outputs to PyTorch"""
        model_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.pth')
        onnx_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.onnx')
        
        # Export model
        metadata = export_to_onnx(model_path, onnx_path)
        
        # Load PyTorch model
        pytorch_model = SimpleCNN()
        pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        pytorch_model.eval()
        
        # Load ONNX model
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        
        # Test with multiple random inputs
        for _ in range(5):
            test_input = torch.randn(1, 1, 28, 28)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input)
            
            # ONNX inference
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # Check output shapes
            assert pytorch_output.shape == (1, 10)
            assert onnx_output.shape == (1, 10)
            
            # Check output similarity
            max_diff = np.max(np.abs(pytorch_output.numpy() - onnx_output))
            assert max_diff < 1e-5, f"Outputs differ too much: {max_diff}"
    
    def test_onnx_batch_inference(self, temp_model_dir):
        """Test ONNX model with different batch sizes"""
        model_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.pth')
        onnx_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.onnx')
        
        # Export model
        export_to_onnx(model_path, onnx_path)
        
        # Test batch inference
        validate_onnx_model(onnx_path)  # This should not raise any exceptions
        
        # Additional batch size tests
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        
        batch_sizes = [1, 3, 8, 16]
        for batch_size in batch_sizes:
            test_input = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            expected_shape = (batch_size, 10)
            assert onnx_output.shape == expected_shape, f"Expected {expected_shape}, got {onnx_output.shape}"
            
            # Check that outputs are valid log probabilities
            assert np.all(onnx_output <= 0), "Log softmax outputs should be <= 0"
    
    def test_onnx_export_creates_directory(self):
        """Test that export creates output directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model
            model = SimpleCNN()
            model_path = os.path.join(temp_dir, 'mnist_cnn.pth')
            torch.save(model.state_dict(), model_path)
            
            # Export to non-existent directory
            onnx_path = os.path.join(temp_dir, 'new_dir', 'mnist_cnn.onnx')
            assert not os.path.exists(os.path.dirname(onnx_path))
            
            metadata = export_to_onnx(model_path, onnx_path)
            
            assert os.path.exists(onnx_path)
            assert metadata['export_successful'] == True
    
    def test_onnx_model_validation(self, temp_model_dir):
        """Test that exported ONNX model passes validation"""
        model_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.pth')
        onnx_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.onnx')
        
        # Export model
        export_to_onnx(model_path, onnx_path)
        
        # Load and validate ONNX model
        import onnx
        onnx_model = onnx.load(onnx_path)
        
        # This should not raise any exceptions
        onnx.checker.check_model(onnx_model)
        
        # Check model properties
        assert len(onnx_model.graph.input) == 1
        assert len(onnx_model.graph.output) == 1
        
        input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        
        # Note: batch dimension might be 0 (dynamic) or -1
        assert input_shape[1:] == [1, 28, 28], f"Expected input shape [?, 1, 28, 28], got {input_shape}"
        assert output_shape[1:] == [10], f"Expected output shape [?, 10], got {output_shape}"
    
    def test_file_sizes_reasonable(self, temp_model_dir):
        """Test that model file sizes are reasonable"""
        model_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.pth')
        onnx_path = os.path.join(temp_model_dir, 'models', 'mnist_cnn.onnx')
        
        metadata = export_to_onnx(model_path, onnx_path)
        
        # Check file sizes are reasonable (not too small or too large)
        pytorch_size = metadata['pytorch_size_mb']
        onnx_size = metadata['onnx_size_mb']
        
        assert 0.5 < pytorch_size < 10, f"PyTorch model size seems wrong: {pytorch_size} MB"
        assert 0.5 < onnx_size < 10, f"ONNX model size seems wrong: {onnx_size} MB"
        
        # ONNX model should be roughly similar size (within 2x)
        size_ratio = max(pytorch_size, onnx_size) / min(pytorch_size, onnx_size)
        assert size_ratio < 3, f"Model sizes differ too much: PyTorch {pytorch_size}MB, ONNX {onnx_size}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])