from pathlib import Path
from typing import Dict, List, TypedDict, Any

import pandas as pd
import orjson
import yaml


class MetricConfig(TypedDict):
    """Configuration for metric normalization."""
    name: str
    keys: List[str]


class DashboardDefaults(TypedDict):
    """Default visualization settings for a dashboard."""
    x_axis_key: str | None
    x_axis_scale: float
    smoothing: float


class DashboardConfig(TypedDict):
    """Configuration for a dashboard."""
    name: str
    defaults: DashboardDefaults
    experiments: List[str]  # Glob patterns
    metrics: List[MetricConfig]


class Run:
    def __init__(self, id: str | Path, df: pd.DataFrame, static: Dict):
        self.id = id
        self.df = df
        self.static = static

    @classmethod
    def from_path(cls, path: str | Path) -> "Run":
        """
        Load a Run from a JSONL log file into a DataFrame
        
        Args:
            path: Path to the JSONL log file
            
        Returns:
            A Run instance with loaded data
        """
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
                if (
                    len(unique_values) == 1
                    and not pd.api.types.is_numeric_dtype(type(unique_values[0]))
                ):
                    static[col] = unique_values[0]
        columns_to_drop = list(static.keys())
        df = df.drop(columns=columns_to_drop)
        
        return cls(id=path, df=df, static=static)


def normalize_metrics(
    runs: List[Run],
    metric_configs: List[MetricConfig],
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
    
    for run in runs:
        df = run.df.copy()
        
        renamed_columns = {}
        for config in metric_configs:
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


def find_dashboards(root_dir: str | Path = ".") -> List[Path]:
    """
    Recursively search for dashboard configuration files.
    
    Args:
        root_dir: Root directory to search in (default: current directory)
        
    Returns:
        Sorted list of paths to *.dashboard.yaml files
    """
    root_path = Path(root_dir)
    dashboards = sorted(root_path.rglob("*.dashboard.yaml"))
    return dashboards


def load_dashboard(path: str | Path) -> DashboardConfig:
    """
    Load a dashboard configuration from a YAML file.
    
    Gracefully handles missing sections by filling in reasonable defaults:
    - defaults.x_axis_scale: 1.0
    - defaults.x_axis_key: None
    - defaults.smoothing: 0.0
    
    Args:
        path: Path to the dashboard YAML file
        
    Returns:
        DashboardConfig dictionary matching the schema
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    defaults = data.get("defaults", {})
    if "x_axis_scale" not in defaults:
        defaults["x_axis_scale"] = 1.0
    if "x_axis_key" not in defaults:
        defaults["x_axis_key"] = None
    if "smoothing" not in defaults:
        defaults["smoothing"] = 0.0
    
    config: DashboardConfig = {
        "name": data.get("name", ""),
        "defaults": defaults,
        "experiments": data.get("experiments", []),
        "metrics": data.get("metrics", []),
    }
    
    return config


def save_dashboard(path: str | Path, config: DashboardConfig) -> None:
    """
    Save a dashboard configuration to a YAML file.
    
    Args:
        path: Path where the YAML file should be written
        config: DashboardConfig dictionary to save
    """
    path = Path(path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def flatten_config(config: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.
    
    Args:
        config: Nested dictionary to flatten
        sep: Separator to use between keys (default: ".")
        
    Returns:
        Flattened dictionary with keys like "parent.child" instead of nested structure
        
    Examples:
        >>> flatten_config({"git": {"hash": "abc"}})
        {'git.hash': 'abc'}
        >>> flatten_config({"a": {"b": {"c": 1}, "d": 2}})
        {'a.b.c': 1, 'a.d': 2}
    """
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