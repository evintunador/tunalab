from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import orjson


class Run:
    """Container for experiment run data."""
    def __init__(self, id: str | Path, df: pd.DataFrame, static: Dict):
        self.id = id
        self.df = df
        self.static = static

    @classmethod
    def from_path(cls, path: str | Path) -> "Run":
        path = Path(path)
        
        entries = []
        with open(path, 'rb') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = orjson.loads(line)
                    entries.append(entry)
                except orjson.JSONDecodeError:
                    continue
        
        if not entries:
            return cls(id=path, df=pd.DataFrame(), static={})
        
        df = pd.DataFrame(entries)
        
        static = {}
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                unique_values = non_null_values.unique()
                if len(unique_values) == 1:
                    static[col] = unique_values[0]
        columns_to_drop = list(static.keys())
        df = df.drop(columns=columns_to_drop)
        
        return cls(id=path, df=df, static=static)


def normalize_metrics(
    runs: List[Run],
    metric_configs: List[Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """
    Normalize metric column names across runs using priority aliasing.
    
    For each run, finds the first available key from each metric config's keys list
    and renames that column to the display name. This allows different runs to use
    different column names (e.g., "dino_loss" vs "loss") while normalizing them
    to a common display name (e.g., "Loss").
    
    Args:
        runs: List of Run objects to normalize
        metric_configs: List of metric configurations, each with:
            - name: Display name for the metric (e.g., "Loss")
            - keys: List of possible column names in priority order (e.g., ["dino_loss", "loss"])
    
    Returns:
        Dictionary mapping run_id to normalized DataFrame with renamed columns.
        Runs are kept separate (not merged) to maintain distinct data for plotting.
    """
    normalized = {}
    
    valid_configs = [
        config for config in metric_configs
        if isinstance(config, dict) and "name" in config and "keys" in config
    ]
    
    for run in runs:
        df = run.df.copy()
        
        renamed_columns = {}
        for config in valid_configs:
            display_name = config["name"]
            keys = config["keys"]
            
            found_key = None
            for key in keys:
                if key in df.columns:
                    found_key = key
                    break
            
            if found_key is not None:
                if display_name not in df.columns or display_name == found_key:
                    renamed_columns[found_key] = display_name
        
        df = df.rename(columns=renamed_columns)
        
        run_id = str(run.id) if isinstance(run.id, Path) else run.id
        normalized[run_id] = df
    
    return normalized


def flatten_config(config: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    result = {}
    
    def _flatten(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{sep}{key}" if prefix else key
                _flatten(value, new_key)
        else:
            result[prefix] = obj
    
    _flatten(config)
    return result

