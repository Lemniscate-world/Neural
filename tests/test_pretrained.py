import pytest
from unittest.mock import patch, MagicMock
import torch
from pretrained_models.pretrained import PretrainedModelHub, fuse_conv_bn_weights

@patch('pretrained.hf_hub_download')
def test_load_model_success(mock_hf):
    # Setup mock Hugging Face download
    mock_hf.return_value = "dummy_path"
    hub = PretrainedModelHub()
    
    # Test loading a valid model
    model = hub.load("resnet50", pretrained=True)
    assert model is not None  # Simplified; replace with actual checks

def test_load_model_not_found():
    hub = PretrainedModelHub()
    with pytest.raises(ValueError):
        hub.load("invalid_model")

def test_fuse_conv_bn_weights():
    # Create dummy weights
    conv_w = torch.randn(64, 3, 7, 7)
    conv_b = torch.randn(64)
    bn_rm = torch.randn(64)
    bn_rv = torch.randn(64).abs()  # Variance must be positive
    bn_w = torch.randn(64)
    bn_b = torch.randn(64)
    eps = 1e-5
    
    fused_w, fused_b = fuse_conv_bn_weights(
        conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b, eps
    )
    
    assert fused_w.shape == conv_w.shape
    assert fused_b.shape == conv_b.shape

@patch('pretrained.torch.load')
def test_convert_torch_weights(mock_torch_load):
    # Create a model with Conv2D and BN
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 7),
        torch.nn.BatchNorm2d(64)
    )
    hub = PretrainedModelHub()
    converted = hub._convert_torch_weights(model)
    
    # Check if weights are fused
    assert '0.weight' in converted
    assert '0.bias' in converted

def test_optimized_model_creation():
    config = {
        'layers': [
            {'type': 'Conv2D', 'fused_conv_bn': True},
            {'type': 'Linear', 'params': {'in_features': 128, 'out_features': 10}}
        ]
    }
    model = OptimizedModel(config)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], FusedConvBNLayer)
    assert isinstance(model.layers[1], torch.nn.Linear)