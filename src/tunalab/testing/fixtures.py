"""Pytest fixture factories and examples."""

from typing import List, Optional
import pytest
import torch


def make_device_fixture(devices: Optional[List[str]] = None):
    """
    Create a pytest fixture that parametrizes over available devices.
    
    Args:
        devices: List of device strings, or None to auto-detect
    
    Returns:
        Pytest fixture function
    
    Example:
        # In conftest.py:
        from tunalab.testing import make_device_fixture
        device = make_device_fixture()
    """
    from tunalab.testing.device import get_available_devices
    
    if devices is None:
        devices = get_available_devices()
    
    @pytest.fixture(params=devices)
    def device(request):
        return request.param
    
    return device


def make_dtype_fixture(dtypes: Optional[List[torch.dtype]] = None):
    """
    Create a pytest fixture that parametrizes over test dtypes.
    
    Args:
        dtypes: List of torch dtypes, or None for defaults [fp32, fp16, bf16]
    
    Returns:
        Pytest fixture function
    
    Example:
        # In conftest.py:
        from tunalab.testing import make_dtype_fixture
        dtype = make_dtype_fixture()
    """
    from tunalab.testing.device import get_test_dtypes
    
    if dtypes is None:
        dtypes = get_test_dtypes()
    
    @pytest.fixture(params=dtypes)
    def dtype(request):
        return request.param
    
    return dtype


# Example conftest.py content for easy copy-paste
EXAMPLE_CONFTEST = '''"""Pytest configuration for tunalab catalog tests."""

import pytest
import torch
from tunalab.testing import get_available_devices, get_test_dtypes


@pytest.fixture(params=get_available_devices())
def device(request):
    """Fixture that parametrizes tests over available devices."""
    return request.param


@pytest.fixture(params=get_test_dtypes())
def dtype(request):
    """Fixture that parametrizes tests over standard dtypes."""
    return request.param
'''

