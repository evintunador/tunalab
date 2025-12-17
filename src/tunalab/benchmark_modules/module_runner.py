from typing import Type, Dict, List, Any, Callable, Optional
from pathlib import Path
import itertools

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from tunalab.benchmark_modules.measurements import measure_performance
from tunalab.testing.device import to_device, to_dtype
from tunalab.paths import get_artifact_root


class ModuleBenchmarkRunner:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (get_artifact_root() / "nn_modules")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_module_benchmark(
        self,
        module_class: Type[nn.Module],
        module_name: str,
        parameter_space: Dict[str, List[Any]],
        init_arg_builder: Callable[[Dict[str, Any]], Dict[str, Any]],
        input_provider: Callable[[Dict[str, Any]], tuple],
        devices: Optional[List[str]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        num_repeats: int = 10,
    ) -> pd.DataFrame:
        """
        Run benchmarks across parameter space and return standardized DataFrame.
        
        Args:
            module_class: nn.Module class to benchmark
            module_name: Human-readable name for the module
            parameter_space: Dict mapping parameter names to lists of values to sweep
            init_arg_builder: Function that converts parameter dict to module init args
            input_provider: Function that generates input tuple from parameter dict
            devices: List of devices to benchmark on. Defaults to ['cuda'] if available.
            dtypes: List of dtypes to benchmark. Defaults to [fp32, fp16, bf16].
            num_repeats: Number of timing repeats per measurement
        
        Returns:
            DataFrame with columns: [param1, ..., dtype, class, device, measurement, value]
            Also saves CSV files to output_dir/{module_name}_{device}.csv
        """
        if devices is None:
            if torch.cuda.is_available():
                devices = ['cuda']
            elif torch.backends.mps.is_available():
                devices = ['mps']
            else:
                devices = ['cpu']
        
        if dtypes is None:
            dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        param_names = list(parameter_space.keys())
        param_values = parameter_space.values()
        param_combinations = list(itertools.product(*param_values))
        
        all_results = []
        
        for device in devices:
            device_results = []
            
            desc = f"Benchmarking {module_name} on {device}"
            total_iterations = len(dtypes) * len(param_combinations)
            
            for dtype in tqdm(dtypes, desc=f"{desc} (dtypes)", leave=False):
                for combo in tqdm(param_combinations, desc=f"{desc} ({dtype})", leave=False):
                    params = dict(zip(param_names, combo))
                    
                    try:
                        init_args = init_arg_builder(params)
                        module = module_class(**init_args)
                        module = to_dtype(to_device(module, device), dtype)
                        
                        inputs = input_provider(init_args)
                        inputs = to_dtype(to_device(inputs, device), dtype)
                        
                        perf_metrics = measure_performance(module, inputs, device, num_repeats)
                        
                        for metric_name, value in perf_metrics.items():
                            result_row = params.copy()
                            result_row['dtype'] = str(dtype).split('.')[-1]  # e.g., 'float32'
                            result_row['class'] = module_name
                            result_row['device'] = device
                            result_row['measurement'] = metric_name
                            result_row['value'] = value
                            device_results.append(result_row)
                    
                    except Exception as e:
                        param_str = ', '.join(f'{k}={v}' for k, v in params.items())
                        tqdm.write(f"[WARNING] Skipping {module_name}({param_str}) on {device}/{dtype}: {e}")
                        continue
            
            if device_results:
                df = pd.DataFrame(device_results)
                
                device_type = torch.device(device).type
                csv_filename = f"{module_name}_{device_type}.csv"
                csv_path = self.output_dir / csv_filename
                df.to_csv(csv_path, index=False)
                tqdm.write(f"Saved {len(device_results)} results to {csv_path}")
                
                all_results.extend(device_results)
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()

