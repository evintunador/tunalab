"""Benchmarking utilities for neural network modules."""

from typing import Dict, Tuple
import time
import torch
import torch.nn as nn

from tunalab.testing.nn_modules import get_total_loss


def measure_performance(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: str,
    num_repeats: int = 10,
) -> Dict[str, float]:
    """
    Measure forward/backward time and memory for a module.
    
    Args:
        module: Module to benchmark
        inputs: Tuple of input tensors
        device: Device string ('cuda', 'mps', 'cpu')
        num_repeats: Number of iterations to average over
    
    Returns:
        Dictionary with metrics that were successfully measured:
            - 'Forward Time (ms)': Average forward pass time
            - 'Backward Time (ms)': Average backward pass time (if applicable)
            - 'Forward Peak Memory (GB)': Peak memory during forward (CUDA only)
            - 'Backward Peak Memory (GB)': Peak memory during backward (CUDA only)
    """
    device_type = torch.device(device).type

    # Warmup runs
    for _ in range(5):
        outputs = module(*inputs)
        loss = get_total_loss(outputs)
        if loss is not None and loss.requires_grad:
            loss.backward()
            module.zero_grad(set_to_none=True)
    
    # Synchronize before measurements
    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'mps':
        torch.mps.synchronize()

    results = {}

    # --- Time Measurement: Forward Pass ---
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            outputs = module(*inputs)
        end_event.record()
        torch.cuda.synchronize()
        fwd_time_ms = start_event.elapsed_time(end_event) / num_repeats
    else:  # MPS and CPU
        start_time = time.perf_counter()
        for _ in range(num_repeats):
            outputs = module(*inputs)
        if device_type == 'mps':
            torch.mps.synchronize()
        end_time = time.perf_counter()
        fwd_time_ms = (end_time - start_time) * 1000 / num_repeats

    results['Forward Time (ms)'] = fwd_time_ms

    # --- Time Measurement: Backward Pass ---
    # Measure forward+backward, then subtract forward time
    outputs = module(*inputs)
    loss = get_total_loss(outputs)

    if loss is not None and loss.requires_grad:
        if device_type == 'cuda':
            start_event.record()
            for _ in range(num_repeats):
                outputs = module(*inputs)
                loss = get_total_loss(outputs)
                loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event) / num_repeats
            bwd_time_ms = total_time_ms - fwd_time_ms
        else:  # MPS and CPU
            start_time = time.perf_counter()
            for _ in range(num_repeats):
                outputs = module(*inputs)
                loss = get_total_loss(outputs)
                loss.backward()
            if device_type == 'mps':
                torch.mps.synchronize()
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000 / num_repeats
            bwd_time_ms = total_time_ms - fwd_time_ms
        
        results['Backward Time (ms)'] = bwd_time_ms
            
    module.zero_grad(set_to_none=True)
    if device_type == 'cuda':
        torch.cuda.empty_cache()

    # --- Memory Measurement (CUDA only) ---
    if device_type == 'cuda':
        # Forward memory
        torch.cuda.reset_peak_memory_stats(device)
        outputs = module(*inputs)
        results['Forward Peak Memory (GB)'] = torch.cuda.max_memory_allocated(device) / 1e9

        # Backward memory
        torch.cuda.reset_peak_memory_stats(device)
        loss = get_total_loss(outputs)
        if loss is not None and loss.requires_grad:
            loss.backward()
            results['Backward Peak Memory (GB)'] = torch.cuda.max_memory_allocated(device) / 1e9

        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    
    return results

