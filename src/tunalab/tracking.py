import atexit
import datetime
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _MetricTracker:
    """
    Internal singleton class for handling metric logging.
    Uses a background thread to write metrics to disk to minimize blocking the training loop.
    """

    def __init__(self):
        self.initialized = False
        self.log_file_path: Optional[str] = None
        self.rank: int = 0
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._warned_large_tensor = False
        self.min_level = logging.INFO

    def init(self, log_dir: str, rank: int = 0, level: int = logging.INFO):
        """
        Initialize the tracker with a specific output directory and rank.
        """
        if self.initialized:
            logger.warning("Tracker already initialized. Ignoring new initialization.")
            return

        self.rank = rank
        self.min_level = level
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f"metrics_rank_{rank}.jsonl")

        self.initialized = True
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        atexit.register(self.shutdown)
        logger.info(f"Metric tracker initialized. Writing to: {self.log_file_path}")

    def _sanitize_value(self, value: Any) -> Any:
        """
        Converts tensors and numpy arrays to native Python types for JSON serialization.
        This happens in the MAIN THREAD to ensure we capture the value *now*,
        before any in-place operations might change it later.
        """
        if isinstance(value, torch.Tensor):
            return self._sanitize_value(value.detach().cpu().numpy())

        elif isinstance(value, np.ndarray):
            # Warn if logging huge arrays
            if value.size > 1000 and not self._warned_large_tensor:
                logger.warning(
                    f"Logging a large tensor/array ({value.size} elements). "
                    "This may impact performance. Only warning once."
                )
                self._warned_large_tensor = True
            if value.size == 1:
                return value.item()
            else:
                return value.tolist()
        
        elif isinstance(value, (np.float32, np.float64, np.float16)):
            return float(value)

        elif isinstance(value, (np.int32, np.int64, np.int16, np.int8, np.uint8)):
            return int(value)
        
        elif isinstance(value, np.bool_):
            return bool(value)
        
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
            
        return value

    def log(self, data: Dict[str, Any], level: int = logging.INFO):
        """
        Log a dictionary of metrics.
        """
        if not self.initialized:
            return

        if level < self.min_level:
            return

        timestamp = datetime.datetime.now().isoformat()
        
        # This prevents "time travel" bugs where a tensor is modified in-place
        # before the background thread gets to read it.
        try:
            sanitized_data = {k: self._sanitize_value(v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to sanitize metric data: {e}")
            return

        payload = {
            "timestamp": timestamp,
            "rank": self.rank,
            "pid": self.pid,
            "hostname": self.hostname,
            **sanitized_data
        }
        self.queue.put(payload)

    def _worker_loop(self):
        """
        Background thread that drains the queue and writes to disk.
        """
        # Buffer for batch writes
        buffer = []
        last_flush = time.time()
        FLUSH_INTERVAL = 2.0  # seconds

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Wait for data, but timeout specifically to check for stop_event / flush
                item = self.queue.get(timeout=0.5)
                buffer.append(item)
            except queue.Empty:
                pass

            # Write if buffer is full or time to flush
            current_time = time.time()
            if buffer and (len(buffer) >= 100 or (current_time - last_flush) > FLUSH_INTERVAL):
                self._flush_buffer(buffer)
                buffer = []
                last_flush = current_time
        
        # Final flush on exit
        if buffer:
            self._flush_buffer(buffer)

    def _flush_buffer(self, buffer):
        if not self.log_file_path:
            return
            
        try:
            with open(self.log_file_path, "a") as f:
                for item in buffer:
                    f.write(json.dumps(item) + "\n")
        except Exception as e:
            # Last resort logging to stderr if file write fails
            print(f"Error writing metrics to {self.log_file_path}: {e}", file=sys.stderr)

    def shutdown(self):
        """
        Gracefully shut down the worker thread, ensuring all queued logs are written.
        """
        if not self.initialized:
            return
            
        logger.debug("Shutting down metric tracker...")
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        self.initialized = False
        logger.debug("Metric tracker shutdown complete.")


# Global singleton instance
_TRACKER = _MetricTracker()


# Public API
def init(log_dir: str, rank: int = 0, level: int = logging.INFO):
    """
    Initialize the global metric tracker.
    
    Args:
        log_dir: The directory where metrics files will be saved.
        rank: The rank of the current process (default: 0).
        level: The minimum logging level to record (default: logging.INFO).
               Metrics logged with a level lower than this will be ignored.
    """
    _TRACKER.init(log_dir, rank, level)


def log(data: Dict[str, Any], level: int = logging.INFO):
    """
    Log a dictionary of metrics to the configured output file.
    
    If the tracker has not been initialized, this function does nothing.
    
    Args:
        data: A dictionary of metrics (e.g., {'loss': 0.5, 'accuracy': 0.9}).
              Tensors will be automatically detached and converted to Python types.
        level: The logging level for this metric (default: logging.INFO).
               If this level is lower than the initialized minimum level,
               the metric will be ignored (low cost).
    """
    _TRACKER.log(data, level)


def shutdown():
    """
    Manually shut down the tracker and flush remaining logs.
    Usually handled automatically by atexit.
    """
    _TRACKER.shutdown()
