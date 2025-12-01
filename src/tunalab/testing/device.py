"""Testing device utilities - simplified wrappers around tunalab.device.

These are lightweight wrappers focused on test parametrization and simple
device/dtype conversions without the extra features needed for production.
"""

from typing import List, Union, Tuple, Any
import torch

from tunalab.device import (
    get_available_devices as _get_available_devices,
    to_device as _to_device_main,
    to_dtype as _to_dtype_main,
)


def get_available_devices() -> List[str]:
    """
    Get simple device list for test parametrization.
    
    Returns:
        List of device strings like ['cuda', 'mps', 'cpu']
        Just the device types, not individual device ranks.
    """
    devices, _ = _get_available_devices()
    return devices


def get_test_dtypes() -> List[torch.dtype]:
    """
    Returns standard list of dtypes to test.
    
    Returns:
        List of torch dtypes [float32, float16, bfloat16]
    """
    return [torch.float32, torch.float16, torch.bfloat16]


def to_device(
    obj: Union[torch.Tensor, torch.nn.Module, Tuple, List],
    device: str
) -> Union[torch.Tensor, torch.nn.Module, Tuple, List]:
    """
    Move tensor, module, or collection to device.
    
    Simplified wrapper without pin_memory and non_blocking flags.
    
    Args:
        obj: Tensor, Module, tuple, or list to move
        device: Target device string
    
    Returns:
        Object moved to device
    """
    return _to_device_main(obj, device, pin_memory=False, non_blocking=False)


def to_dtype(
    obj: Union[torch.Tensor, torch.nn.Module, Tuple, List],
    dtype: Union[torch.dtype, str]
) -> Union[torch.Tensor, torch.nn.Module, Tuple, List]:
    """
    Convert tensor, module, or collection to dtype.
    
    Args:
        obj: Tensor, Module, tuple, or list to convert
        dtype: Target dtype (torch.dtype or string like 'fp32')
    
    Returns:
        Object converted to dtype
    """
    return _to_dtype_main(obj, dtype)
