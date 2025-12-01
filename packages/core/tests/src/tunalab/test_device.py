import pytest
import torch
import torch.nn as nn
from unittest import mock

from tunalab.device import (
    get_default_device,
    to_device,
    dtype_str_parse,
    to_dtype,
    get_available_devices,
)


# --- get_default_device tests ---

def test_get_default_device_returns_device():
    """Test that get_default_device returns a torch.device object."""
    device = get_default_device()
    assert isinstance(device, torch.device)


def test_get_default_device_priority():
    """Test device selection priority: CUDA > MPS > CPU."""
    device = get_default_device()
    
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.current_device', return_value=0)
def test_get_default_device_cuda(mock_current, mock_available):
    """Test that CUDA is selected when available."""
    device = get_default_device()
    assert device.type == "cuda"


@mock.patch('torch.cuda.is_available', return_value=False)
@mock.patch('torch.backends.mps.is_available', return_value=True)
def test_get_default_device_mps(mock_mps, mock_cuda):
    """Test that MPS is selected when CUDA unavailable but MPS available."""
    device = get_default_device()
    assert device.type == "mps"


@mock.patch('torch.cuda.is_available', return_value=False)
@mock.patch('torch.backends.mps.is_available', return_value=False)
def test_get_default_device_cpu_fallback(mock_mps, mock_cuda):
    """Test that CPU is selected when no accelerators are available."""
    device = get_default_device()
    assert device.type == "cpu"


# --- to_device tests ---

def test_to_device_tensor():
    """Test moving a single tensor to device."""
    tensor = torch.randn(3, 4)
    device = torch.device("cpu")
    result = to_device(tensor, device)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"
    assert torch.allclose(result, tensor)


def test_to_device_list():
    """Test moving a list of tensors to device."""
    tensors = [torch.randn(2, 3), torch.randn(4, 5)]
    device = torch.device("cpu")
    result = to_device(tensors, device)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(t.device.type == "cpu" for t in result)


def test_to_device_tuple():
    """Test moving a tuple of tensors to device."""
    tensors = (torch.randn(2, 3), torch.randn(4, 5))
    device = torch.device("cpu")
    result = to_device(tensors, device)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(t.device.type == "cpu" for t in result)


def test_to_device_dict():
    """Test moving a dict of tensors to device."""
    tensors = {
        "a": torch.randn(2, 3),
        "b": torch.randn(4, 5),
    }
    device = torch.device("cpu")
    result = to_device(tensors, device)
    
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}
    assert all(t.device.type == "cpu" for t in result.values())


def test_to_device_nested_structure():
    """Test moving nested data structures to device."""
    data = {
        "tensors": [torch.randn(2, 3), torch.randn(4, 5)],
        "nested": {
            "tensor": torch.randn(3, 3),
        },
        "tuple": (torch.randn(1, 1),),
    }
    device = torch.device("cpu")
    result = to_device(data, device)
    
    assert isinstance(result, dict)
    assert isinstance(result["tensors"], list)
    assert isinstance(result["nested"], dict)
    assert isinstance(result["tuple"], tuple)
    assert result["tensors"][0].device.type == "cpu"
    assert result["nested"]["tensor"].device.type == "cpu"
    assert result["tuple"][0].device.type == "cpu"


def test_to_device_module():
    """Test moving an nn.Module to device."""
    model = nn.Linear(10, 5)
    device = torch.device("cpu")
    result = to_device(model, device)
    
    assert isinstance(result, nn.Module)
    assert next(result.parameters()).device.type == "cpu"


def test_to_device_string_device():
    """Test that string device specifications work."""
    tensor = torch.randn(3, 4)
    result = to_device(tensor, "cpu")
    assert result.device.type == "cpu"


def test_to_device_non_tensor_passthrough():
    """Test that non-tensor objects without .to() are passed through unchanged."""
    obj = "not a tensor"
    result = to_device(obj, "cpu")
    assert result == obj
    
    num = 42
    result = to_device(num, "cpu")
    assert result == num


def test_to_device_pin_memory():
    """Test pin_memory parameter (basic check, actual pinning requires CUDA)."""
    tensor = torch.randn(3, 4)
    # Should not raise even if CUDA unavailable
    result = to_device(tensor, "cpu", pin_memory=True)
    assert isinstance(result, torch.Tensor)


def test_to_device_non_blocking():
    """Test non_blocking parameter."""
    tensor = torch.randn(3, 4)
    result = to_device(tensor, "cpu", non_blocking=True)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"


# --- dtype_str_parse tests ---

@pytest.mark.parametrize("dtype_str,expected", [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("FP32", torch.float32),  # Test case insensitivity
    ("FLOAT16", torch.float16),
])
def test_dtype_str_parse_valid(dtype_str, expected):
    """Test parsing valid dtype strings."""
    result = dtype_str_parse(dtype_str)
    assert result == expected


def test_dtype_str_parse_invalid():
    """Test that invalid dtype strings are returned as-is."""
    invalid = "invalid_dtype"
    result = dtype_str_parse(invalid)
    assert result == invalid


# --- to_dtype tests ---

def test_to_dtype_tensor_with_dtype_object():
    """Test converting a tensor to a dtype using torch.dtype."""
    tensor = torch.randn(3, 4, dtype=torch.float32)
    result = to_dtype(tensor, torch.float16)
    assert result.dtype == torch.float16


def test_to_dtype_tensor_with_string():
    """Test converting a tensor to a dtype using a string."""
    tensor = torch.randn(3, 4, dtype=torch.float32)
    result = to_dtype(tensor, "fp16")
    assert result.dtype == torch.float16


@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16", "float32"])
def test_to_dtype_various_strings(dtype_str):
    """Test various dtype string conversions."""
    tensor = torch.randn(3, 4)
    result = to_dtype(tensor, dtype_str)
    expected_dtype = dtype_str_parse(dtype_str)
    assert result.dtype == expected_dtype


def test_to_dtype_list():
    """Test converting a list of tensors to dtype."""
    tensors = [torch.randn(2, 3), torch.randn(4, 5)]
    result = to_dtype(tensors, torch.float16)
    
    assert isinstance(result, list)
    assert all(t.dtype == torch.float16 for t in result)


def test_to_dtype_tuple():
    """Test converting a tuple of tensors to dtype."""
    tensors = (torch.randn(2, 3), torch.randn(4, 5))
    result = to_dtype(tensors, torch.float16)
    
    assert isinstance(result, tuple)
    assert all(t.dtype == torch.float16 for t in result)


def test_to_dtype_dict():
    """Test converting a dict of tensors to dtype."""
    tensors = {
        "a": torch.randn(2, 3),
        "b": torch.randn(4, 5),
    }
    result = to_dtype(tensors, "fp16")
    
    assert isinstance(result, dict)
    assert all(t.dtype == torch.float16 for t in result.values())


def test_to_dtype_nested_structure():
    """Test converting nested data structures to dtype."""
    data = {
        "tensors": [torch.randn(2, 3), torch.randn(4, 5)],
        "nested": {
            "tensor": torch.randn(3, 3),
        },
    }
    result = to_dtype(data, torch.float16)
    
    assert result["tensors"][0].dtype == torch.float16
    assert result["nested"]["tensor"].dtype == torch.float16


def test_to_dtype_module():
    """Test converting an nn.Module's parameters to dtype."""
    model = nn.Linear(10, 5)
    result = to_dtype(model, torch.float16)
    
    assert isinstance(result, nn.Module)
    assert next(result.parameters()).dtype == torch.float16


def test_to_dtype_invalid_string():
    """Test that invalid dtype strings return the item unchanged."""
    tensor = torch.randn(3, 4, dtype=torch.float32)
    result = to_dtype(tensor, "invalid_dtype")
    # Should return unchanged since invalid dtype
    assert result.dtype == torch.float32


def test_to_dtype_non_tensor_passthrough():
    """Test that non-tensor objects without .to() are passed through unchanged."""
    obj = "not a tensor"
    result = to_dtype(obj, torch.float16)
    assert result == obj


# --- get_available_devices tests ---

def test_get_available_devices_returns_lists():
    """Test that get_available_devices returns two lists."""
    devices, devices_with_ranks = get_available_devices()
    assert isinstance(devices, list)
    assert isinstance(devices_with_ranks, list)


def test_get_available_devices_cpu_always_available_when_no_accelerators():
    """Test that CPU is returned when no accelerators are available."""
    with mock.patch('torch.cuda.is_available', return_value=False), \
         mock.patch('torch.backends.mps.is_available', return_value=False):
        devices, devices_with_ranks = get_available_devices()
        assert "cpu" in devices
        assert "cpu" in devices_with_ranks


def test_get_available_devices_cuda():
    """Test detection of CUDA devices."""
    if torch.cuda.is_available():
        devices, devices_with_ranks = get_available_devices()
        assert "cuda" in devices
        assert any(d.startswith("cuda:") for d in devices_with_ranks)


def test_get_available_devices_mps():
    """Test detection of MPS devices."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices, devices_with_ranks = get_available_devices()
        assert "mps" in devices
        assert "mps" in devices_with_ranks


@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('torch.cuda.get_device_name', return_value="Mock GPU")
def test_get_available_devices_multiple_cuda(mock_name, mock_count, mock_available):
    """Test detection of multiple CUDA devices."""
    devices, devices_with_ranks = get_available_devices()
    
    assert "cuda" in devices
    assert "cuda:0" in devices_with_ranks
    assert "cuda:1" in devices_with_ranks


def test_get_available_devices_exclude_cuda():
    """Test excluding CUDA devices."""
    devices, devices_with_ranks = get_available_devices(exclude=["cuda"])
    
    assert "cuda" not in devices
    assert not any(d.startswith("cuda:") for d in devices_with_ranks)


def test_get_available_devices_exclude_mps():
    """Test excluding MPS devices."""
    devices, devices_with_ranks = get_available_devices(exclude=["mps"])
    
    assert "mps" not in devices
    assert "mps" not in devices_with_ranks


def test_get_available_devices_exclude_multiple():
    """Test excluding multiple device types."""
    devices, devices_with_ranks = get_available_devices(exclude=["cuda", "mps"])
    
    assert "cuda" not in devices
    assert "mps" not in devices
    # CPU should be present as fallback
    assert "cpu" in devices


@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.device_count', return_value=1)
@mock.patch('torch.cuda.get_device_name', return_value="Mock GPU")
def test_get_available_devices_cpu_not_included_with_accelerators(mock_name, mock_count, mock_available):
    """Test that CPU is not included when accelerators are available."""
    devices, devices_with_ranks = get_available_devices()
    
    # CPU should not be in the list when CUDA is available
    if "cuda" in devices:
        assert "cpu" not in devices


# --- Integration tests ---

def test_to_device_and_dtype_chaining():
    """Test that to_device and to_dtype can be chained."""
    tensor = torch.randn(3, 4, dtype=torch.float32)
    device = torch.device("cpu")
    
    result = to_device(tensor, device)
    result = to_dtype(result, torch.float16)
    
    assert result.device.type == "cpu"
    assert result.dtype == torch.float16


def test_complex_nested_structure():
    """Test moving and converting a complex nested structure."""
    model = nn.Linear(5, 3)
    data = {
        "model": model,
        "inputs": [torch.randn(2, 5), torch.randn(3, 5)],
        "metadata": {
            "tensor": torch.randn(1, 1),
            "string": "test",
            "number": 42,
        },
    }
    
    device = torch.device("cpu")
    result = to_device(data, device)
    result = to_dtype(result, torch.float16)
    
    # Check device
    assert next(result["model"].parameters()).device.type == "cpu"
    assert all(t.device.type == "cpu" for t in result["inputs"])
    assert result["metadata"]["tensor"].device.type == "cpu"
    
    # Check dtype
    assert next(result["model"].parameters()).dtype == torch.float16
    assert all(t.dtype == torch.float16 for t in result["inputs"])
    assert result["metadata"]["tensor"].dtype == torch.float16
    
    # Check passthrough values
    assert result["metadata"]["string"] == "test"
    assert result["metadata"]["number"] == 42

