import json
import os
import tempfile
import time
import threading
import pytest
import logging
import torch
import numpy as np
from tunalab import tracking

def test_tracking_init_and_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize tracker
        tracking.init(tmpdir, rank=0)
        
        # Log some data
        data = {"loss": 0.5, "step": 1}
        tracking.log(data)
        
        # Wait for background thread to flush (flush interval is 2s, but we can trigger shutdown)
        tracking.shutdown()
        
        # Check file
        log_file = os.path.join(tmpdir, "metrics_rank_0.jsonl")
        assert os.path.exists(log_file)
        
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["loss"] == 0.5
            assert entry["step"] == 1
            assert entry["rank"] == 0
            assert "timestamp" in entry
            assert "hostname" in entry

def test_tracking_tensor_conversion():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking._TRACKER = tracking._MetricTracker() # Reset singleton for test
        tracking.init(tmpdir, rank=1)
        
        tensor_scalar = torch.tensor(3.14)
        tensor_array = torch.tensor([1.0, 2.0, 3.0])
        numpy_scalar = np.float32(0.123)
        numpy_array = np.array([4, 5, 6])
        numpy_bool = np.bool_(True)
        
        tracking.log({
            "scalar": tensor_scalar,
            "array": tensor_array,
            "np_scalar": numpy_scalar,
            "np_array": numpy_array,
            "np_bool": numpy_bool
        })
        
        tracking.shutdown()
        
        log_file = os.path.join(tmpdir, "metrics_rank_1.jsonl")
        with open(log_file, "r") as f:
            entry = json.loads(f.readline())
            
            assert abs(entry["scalar"] - 3.14) < 1e-4
            assert entry["array"] == [1.0, 2.0, 3.0]
            assert abs(entry["np_scalar"] - 0.123) < 1e-4
            assert entry["np_array"] == [4, 5, 6]
            assert entry["np_bool"] is True

def test_tracking_no_init_no_error():
    # Reset singleton
    tracking._TRACKER = tracking._MetricTracker()
    
    # Should not error
    tracking.log({"foo": "bar"})
    tracking.shutdown()

def test_threading_flush():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking._TRACKER = tracking._MetricTracker()
        tracking.init(tmpdir, rank=0)
        
        # Log many items to force buffer flush
        for i in range(150):
            tracking.log({"i": i})
            
        # Wait a bit for the thread to process
        # (The worker loop checks queue.empty() so it might be fast, 
        # but we want to ensure at least one flush happened)
        time.sleep(0.5) 
        tracking.shutdown()
        
        log_file = os.path.join(tmpdir, "metrics_rank_0.jsonl")
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 150
            last_entry = json.loads(lines[-1])
            assert last_entry["i"] == 149

def test_log_levels():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Case 1: Init with INFO (default), log DEBUG -> Should be ignored
        tracking._TRACKER = tracking._MetricTracker()
        tracking.init(tmpdir, rank=0, level=logging.INFO)
        
        tracking.log({"msg": "visible"}, level=logging.INFO)
        tracking.log({"msg": "hidden"}, level=logging.DEBUG)
        tracking.log({"msg": "visible_warning"}, level=logging.WARNING)
        
        tracking.shutdown()
        
        log_file = os.path.join(tmpdir, "metrics_rank_0.jsonl")
        with open(log_file, "r") as f:
            lines = [json.loads(line) for line in f.readlines()]
            
        assert len(lines) == 2
        assert lines[0]["msg"] == "visible"
        assert lines[1]["msg"] == "visible_warning"
        
        # Case 2: Init with DEBUG, log DEBUG -> Should be visible
        tracking._TRACKER = tracking._MetricTracker()
        tracking.init(tmpdir, rank=0, level=logging.DEBUG)
        
        tracking.log({"msg": "debug_msg"}, level=logging.DEBUG)
        tracking.shutdown()
        
        with open(log_file, "r") as f:
            # Note: file is appended to if not deleted, but we are in the same tmpdir
            # Actually _MetricTracker opens in "a" mode.
            # We should read all lines. The first 2 are from previous run.
            lines = [json.loads(line) for line in f.readlines()]
            
        assert len(lines) == 3
        assert lines[2]["msg"] == "debug_msg"
