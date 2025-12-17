from typing import Type, Dict, List, Any, Callable, Optional
from pathlib import Path
import itertools
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

from tunalab.paths import get_artifact_root


class OptimizerBenchmarkRunner:
    """
    Standardized benchmark runner for optimizers that produces CSV files
    compatible with the optimizer benchmark visualization notebook.
    
    The output format:
        Columns: [param1, param2, ..., class, device, measurement, value]
    
    Measurements include:
        - avg_step_time_ms: Average time per optimizer.step()
        - loss_reduction_pct: Percentage loss reduction
        - loss_reduction_per_ms: Efficiency metric (loss reduction / time)
    
    Example:
        runner = OptimizerBenchmarkRunner()
        results = runner.run_optimizer_benchmark(
            optimizer_class=AdamW,
            optimizer_name='AdamW',
            parameter_space={'lr': [1e-4, 1e-3], 'weight_decay': [0.0, 0.01]},
            optimizer_kwargs_builder=lambda p: {'lr': p['lr'], 'weight_decay': p['weight_decay']},
        )
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (get_artifact_root() / "optimizers")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_optimizer_benchmark(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_name: str,
        parameter_space: Dict[str, List[Any]],
        optimizer_kwargs_builder: Callable[[Dict[str, Any]], Dict[str, Any]],
        devices: Optional[List[str]] = None,
        num_steps: int = 100,
        model_builder: Optional[Callable[[str], nn.Module]] = None,
        task_builder: Optional[Callable[[str], tuple]] = None,
    ) -> pd.DataFrame:
        """
        Run optimizer benchmarks across parameter space.
        
        Args:
            optimizer_class: Optimizer class to benchmark
            optimizer_name: Human-readable name for the optimizer
            parameter_space: Dict mapping parameter names to lists of values to sweep
            optimizer_kwargs_builder: Function that converts parameter dict to optimizer kwargs
            devices: List of devices to benchmark on. Defaults to ['cuda'] if available.
            num_steps: Number of optimization steps to run
            model_builder: Optional custom model builder (device) -> model
            task_builder: Optional custom task builder (device) -> (dataloader, loss_fn)
        
        Returns:
            DataFrame with columns: [param1, ..., class, device, measurement, value]
            Also saves CSV files to output_dir/{optimizer_name}_{device}.csv
        """
        if devices is None:
            if torch.cuda.is_available():
                devices = ['cuda']
            elif torch.backends.mps.is_available():
                devices = ['mps']
            else:
                devices = ['cpu']
        
        param_names = list(parameter_space.keys())
        param_values = parameter_space.values()
        param_combinations = list(itertools.product(*param_values))
        
        all_results = []
        
        for device in devices:
            device_results = []
            
            desc = f"Benchmarking {optimizer_name} on {device}"
            
            for combo in tqdm(param_combinations, desc=desc):
                params = dict(zip(param_names, combo))
                
                try:
                    if model_builder is not None:
                        model = model_builder(device)
                    else:
                        model = self._default_model(device)
                    
                    if task_builder is not None:
                        dataloader, loss_fn = task_builder(device)
                    else:
                        dataloader, loss_fn = self._default_task(device)
                    
                    optimizer_kwargs = optimizer_kwargs_builder(params)
                    
                    try:
                        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
                    except (AssertionError, TypeError):
                        # Try with list (for Muon-style optimizers)
                        optimizer = optimizer_class(list(model.parameters()), **optimizer_kwargs)
                    
                    metrics = self._benchmark_optimizer(
                        model=model,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        loss_fn=loss_fn,
                        device=device,
                        num_steps=num_steps,
                    )
                    
                    for metric_name, value in metrics.items():
                        result_row = params.copy()
                        result_row['class'] = optimizer_name
                        result_row['device'] = device
                        result_row['measurement'] = metric_name
                        result_row['value'] = value
                        device_results.append(result_row)
                
                except Exception as e:
                    param_str = ', '.join(f'{k}={v}' for k, v in params.items())
                    tqdm.write(f"[WARNING] Skipping {optimizer_name}({param_str}) on {device}: {e}")
                    continue
            
            if device_results:
                df = pd.DataFrame(device_results)
                csv_path = self.output_dir / f"{optimizer_name}_{device}.csv"
                df.to_csv(csv_path, index=False)
                tqdm.write(f"Saved {len(device_results)} results to {csv_path}")
                all_results.extend(device_results)
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    def _default_model(self, device: str) -> nn.Module:
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        ).to(device)
    
    def _default_task(self, device: str) -> tuple:
        torch.manual_seed(42)
        
        X = torch.randn(2048, 32).to(device)
        true_weights = torch.randn(32).to(device)
        y = ((X * true_weights).sum(dim=1) > 0.0).long().to(device)
        
        ds = TensorDataset(X, y)
        dataloader = DataLoader(ds, batch_size=64, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        
        return dataloader, loss_fn
    
    def _benchmark_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        num_steps: int,
    ) -> Dict[str, float]:
        
        def measure_loss(model, dataloader):
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                total_samples = 0
                for batch_X, batch_y in dataloader:
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    total_loss += loss.item() * batch_X.size(0)
                    total_samples += batch_X.size(0)
            return total_loss / total_samples
        
        initial_loss = measure_loss(model, dataloader)
        
        model.train()
        step_times = []
        
        data_iter = iter(dataloader)
        for _ in range(num_steps):
            try:
                batch_X, batch_y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_X, batch_y = next(data_iter)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            
            if device == 'cuda':
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                optimizer.step()
                end = time.perf_counter()
            
            step_times.append((end - start) * 1000)  # Convert to ms
        
        final_loss = measure_loss(model, dataloader)
        
        avg_step_time = sum(step_times) / len(step_times)
        loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100
        
        return {
            'avg_step_time_ms': avg_step_time,
            'loss_reduction_pct': loss_reduction_pct,
            'loss_reduction_per_ms': loss_reduction_pct / avg_step_time if avg_step_time > 0 else 0.0,
        }
