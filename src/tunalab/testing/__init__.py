"""
Pytest-native testing utilities for tunalab.

This package provides pure helper functions and fixture factories for testing
neural network modules, optimizers, and training loops. Unlike the old validation
package, these are designed to work with standard pytest patterns.

Usage:
    # In your test file:
    from tunalab.testing import compare_modules, get_tolerances_for_dtype
    
    # In your conftest.py:
    from tunalab.testing import get_available_devices, get_test_dtypes
"""

# Core utilities
from tunalab.testing.device import (
    get_available_devices,
    get_test_dtypes,
    to_device,
    to_dtype,
)

# NN Module testing helpers
from tunalab.testing.nn_modules import (
    ComparisonResult,
    compare_modules,
    get_tolerances_for_dtype,
    get_total_loss,
)

# Optimizer testing helpers
from tunalab.testing.optimizers import (
    run_learning_test,
)

# Training loop testing helpers
from tunalab.testing.train_loops import (
    SimpleTestTrainingModel,
    run_training_smoke_test,
    run_base_loop_compliance_test,
)

# Benchmarking helpers
from tunalab.testing.benchmarking import (
    measure_performance,
)
from tunalab.testing.benchmark_runner import (
    BenchmarkRunner,
)

# Fixture factories
from tunalab.testing.fixtures import (
    make_device_fixture,
    make_dtype_fixture,
    EXAMPLE_CONFTEST,
)

__all__ = [
    # Device/dtype utilities
    "get_available_devices",
    "get_test_dtypes",
    "to_device",
    "to_dtype",
    # NN Module helpers
    "ComparisonResult",
    "compare_modules",
    "get_tolerances_for_dtype",
    "get_total_loss",
    # Optimizer helpers
    "run_learning_test",
    # Training loop helpers
    "SimpleTestTrainingModel",
    "run_training_smoke_test",
    "run_base_loop_compliance_test",
    # Benchmarking
    "measure_performance",
    "BenchmarkRunner",
    # Fixtures
    "make_device_fixture",
    "make_dtype_fixture",
    "EXAMPLE_CONFTEST",
]

