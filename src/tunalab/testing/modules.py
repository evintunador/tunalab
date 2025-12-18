from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence, Any
import copy

import torch
import torch.nn as nn


@dataclass
class ComparisonResult:
    """Result of comparing two modules' outputs and gradients."""
    passed: bool
    failures: List[str]
    max_output_abs_diff: Optional[float] = None
    max_output_rel_diff: Optional[float] = None
    max_grad_abs_diff: Optional[float] = None
    max_grad_rel_diff: Optional[float] = None


def get_tolerances_for_dtype(dtype: torch.dtype) -> Dict[str, float]:
    """
    Get standard tolerance values for a given dtype.
    
    Args:
        dtype: PyTorch dtype
    
    Returns:
        Dictionary with 'atol' and 'rtol' keys
    """
    tolerance_map = {
        torch.float32: {'atol': 1e-5, 'rtol': 1e-4},
        torch.float16: {'atol': 1e-3, 'rtol': 1e-2},
        torch.bfloat16: {'atol': 1e-2, 'rtol': 1e-1},
        torch.float64: {'atol': 1e-8, 'rtol': 1e-6},
    }
    return tolerance_map.get(dtype, {'atol': 1e-5, 'rtol': 1e-4})


def get_total_loss(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """
    Compute a scalar loss from module outputs for backward pass.
    
    Args:
        outputs: Single tensor or tuple of tensors from module forward
    
    Returns:
        Scalar tensor suitable for .backward()
    """
    if isinstance(outputs, torch.Tensor):
        return outputs.sum()
    
    # Handle tuple outputs, summing only floating point tensors
    total_loss = torch.tensor(0.0, device=outputs[0].device if outputs else 'cpu')
    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            total_loss = total_loss + out.sum()
    return total_loss


def compare_modules(
    ref_module: nn.Module,
    test_module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    tolerances: Optional[Dict[str, float]] = None,
    check_gradients: bool = True,
    copy_state_dict: bool = True,
) -> ComparisonResult:
    """
    Compare outputs and gradients of two modules.
    
    Args:
        ref_module: Reference module (ground truth)
        test_module: Module to test
        inputs: Tuple of input tensors
        tolerances: Dict with 'atol' and 'rtol' keys, or None for dtype-based defaults
        check_gradients: Whether to compare gradients
        copy_state_dict: Whether to copy ref state dict to test module first
    
    Returns:
        ComparisonResult with comparison details
    """
    # Set default tolerances based on input dtype
    if tolerances is None:
        dtype = inputs[0].dtype if inputs else torch.float32
        tolerances = get_tolerances_for_dtype(dtype)
    
    # Copy weights if requested
    if copy_state_dict:
        test_module.load_state_dict(ref_module.state_dict())
    
    # Clone inputs for independent runs
    ref_inputs = tuple(
        t.clone().detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else copy.deepcopy(t)
        for t in inputs
    )
    test_inputs = tuple(
        t.clone().detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else copy.deepcopy(t)
        for t in inputs
    )
    
    # Enable gradient retention for non-leaf tensors
    for t in ref_inputs:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            t.retain_grad()
    for t in test_inputs:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            t.retain_grad()
    
    failures = []
    max_output_abs_diff = None
    max_output_rel_diff = None
    max_grad_abs_diff = None
    max_grad_rel_diff = None
    
    # Forward passes
    ref_outputs = ref_module(*ref_inputs)
    test_outputs = test_module(*test_inputs)
    
    # Normalize to tuples
    ref_outputs_tuple = ref_outputs if isinstance(ref_outputs, tuple) else (ref_outputs,)
    test_outputs_tuple = test_outputs if isinstance(test_outputs, tuple) else (test_outputs,)
    
    # Compare outputs
    if len(ref_outputs_tuple) != len(test_outputs_tuple):
        failures.append(
            f"Output count mismatch: ref has {len(ref_outputs_tuple)}, test has {len(test_outputs_tuple)}"
        )
    else:
        for i, (ref_out, test_out) in enumerate(zip(ref_outputs_tuple, test_outputs_tuple)):
            if isinstance(ref_out, torch.Tensor) and isinstance(test_out, torch.Tensor):
                if not torch.allclose(ref_out, test_out, **tolerances):
                    abs_diff = (ref_out - test_out).abs()
                    rel_diff = abs_diff / (ref_out.abs() + 1e-8)
                    max_abs = abs_diff.max().item()
                    max_rel = rel_diff.max().item()
                    
                    if max_output_abs_diff is None or max_abs > max_output_abs_diff:
                        max_output_abs_diff = max_abs
                    if max_output_rel_diff is None or max_rel > max_output_rel_diff:
                        max_output_rel_diff = max_rel
                    
                    failures.append(
                        f"Output {i} mismatch: max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}"
                    )
    
    # Compare gradients if requested
    if check_gradients:
        ref_loss = get_total_loss(ref_outputs)
        test_loss = get_total_loss(test_outputs)
        
        if ref_loss.requires_grad:
            ref_loss.backward()
        if test_loss.requires_grad:
            test_loss.backward()
        
        # Compare parameter gradients
        ref_params = list(ref_module.parameters())
        test_params = list(test_module.parameters())
        
        if len(ref_params) != len(test_params):
            failures.append(
                f"Parameter count mismatch: ref has {len(ref_params)}, test has {len(test_params)}"
            )
        else:
            for i, (ref_p, test_p) in enumerate(zip(ref_params, test_params)):
                if ref_p.grad is not None and test_p.grad is not None:
                    if not torch.allclose(ref_p.grad, test_p.grad, **tolerances):
                        abs_diff = (ref_p.grad - test_p.grad).abs()
                        rel_diff = abs_diff / (ref_p.grad.abs() + 1e-8)
                        max_abs = abs_diff.max().item()
                        max_rel = rel_diff.max().item()
                        
                        if max_grad_abs_diff is None or max_abs > max_grad_abs_diff:
                            max_grad_abs_diff = max_abs
                        if max_grad_rel_diff is None or max_rel > max_grad_rel_diff:
                            max_grad_rel_diff = max_rel
                        
                        failures.append(
                            f"Param {i} grad mismatch: max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}"
                        )
                elif ref_p.grad is not None or test_p.grad is not None:
                    failures.append(f"Param {i} grad existence mismatch")
        
        # Compare input gradients
        for i, (ref_in, test_in) in enumerate(zip(ref_inputs, test_inputs)):
            if isinstance(ref_in, torch.Tensor) and isinstance(test_in, torch.Tensor):
                if ref_in.requires_grad and test_in.requires_grad:
                    if ref_in.grad is not None and test_in.grad is not None:
                        if not torch.allclose(ref_in.grad, test_in.grad, **tolerances):
                            abs_diff = (ref_in.grad - test_in.grad).abs()
                            rel_diff = abs_diff / (ref_in.grad.abs() + 1e-8)
                            max_abs = abs_diff.max().item()
                            max_rel = rel_diff.max().item()
                            
                            if max_grad_abs_diff is None or max_abs > max_grad_abs_diff:
                                max_grad_abs_diff = max_abs
                            if max_grad_rel_diff is None or max_rel > max_grad_rel_diff:
                                max_grad_rel_diff = max_rel
                            
                            failures.append(
                                f"Input {i} grad mismatch: max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}"
                            )
    
    return ComparisonResult(
        passed=len(failures) == 0,
        failures=failures,
        max_output_abs_diff=max_output_abs_diff,
        max_output_rel_diff=max_output_rel_diff,
        max_grad_abs_diff=max_grad_abs_diff,
        max_grad_rel_diff=max_grad_rel_diff,
    )

