#!/usr/bin/env python
"""
ONNX Export Script for Tutorial 3
Converts trained PyTorch model to ONNX format for production deployment
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
from model import SimpleCNN


def export_to_onnx(model_path='./models/mnist_cnn.pth', output_path='./models/mnist_cnn.onnx'):
    """
    Export trained PyTorch model to ONNX format
    
    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
    
    Returns:
        dict: Export metadata and validation results
    """
    
    print("Tutorial 3: ONNX Model Export")
    print("=" * 50)
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Device: {device}")
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to ONNX format...")
    
    # Export model to ONNX
    torch.onnx.export(
        model,                     # model being run
        dummy_input,               # model input (sample)
        output_path,               # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to: {output_path}")
    
    # Verify the exported model
    print("Verifying ONNX model...")
    
    # Load and check ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")
    
    # Test inference with ONNX Runtime
    print("Testing ONNX Runtime inference...")
    
    ort_session = ort.InferenceSession(output_path)
    
    # Test with dummy input
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare with PyTorch output
    with torch.no_grad():
        torch_output = model(dummy_input.cpu())
    
    # Check if outputs are close
    max_diff = np.max(np.abs(torch_output.numpy() - ort_outputs[0]))
    print(f"Maximum difference between PyTorch and ONNX: {max_diff:.6f}")
    
    # Verify outputs are close (tolerance for floating point differences)
    assert max_diff < 1e-5, f"ONNX output differs too much from PyTorch: {max_diff}"
    print("✓ ONNX Runtime inference test passed")
    
    # Get file sizes
    pytorch_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)    # MB
    
    # Prepare export metadata
    export_metadata = {
        "pytorch_model_path": model_path,
        "onnx_model_path": output_path,
        "pytorch_size_mb": pytorch_size,
        "onnx_size_mb": onnx_size,
        "max_inference_diff": max_diff,
        "opset_version": 11,
        "export_successful": True,
        "input_shape": list(dummy_input.shape),
        "output_shape": list(ort_outputs[0].shape)
    }
    
    print("\nExport Summary:")
    print(f"  PyTorch model: {pytorch_size:.2f} MB")
    print(f"  ONNX model: {onnx_size:.2f} MB")
    print(f"  Size difference: {((onnx_size - pytorch_size) / pytorch_size * 100):+.1f}%")
    print(f"  Max inference difference: {max_diff:.6f}")
    print(f"  Export successful: ✓")
    
    return export_metadata


def validate_onnx_model(onnx_path='./models/mnist_cnn.onnx'):
    """
    Test ONNX model with multiple inputs
    
    Args:
        onnx_path: Path to ONNX model
    """
    
    print("\nTesting ONNX model with batch inputs...")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        test_input = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
        
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        expected_shape = (batch_size, 10)
        actual_shape = ort_outputs[0].shape
        
        assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
        
        # Check that outputs are valid probabilities (log softmax)
        assert np.all(ort_outputs[0] <= 0), "Log softmax outputs should be <= 0"
        
        print(f"  ✓ Batch size {batch_size}: {actual_shape}")
    
    print("✓ All batch size tests passed")


if __name__ == "__main__":
    try:
        # Export model
        metadata = export_to_onnx()
        
        # Test exported model
        validate_onnx_model()
        
        print("\n🎉 ONNX export completed successfully!")
        print("Ready for production deployment.")
        
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        exit(1)