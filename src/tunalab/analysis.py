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
                if len(unique_values) == 1:
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
        
    Raises:
        yaml.YAMLError: If the YAML file is malformed
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing dashboard file {path}: {e}") from e
    
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


def extract_metrics_section(path: str | Path) -> str:
    """
    Extract the raw metrics section from a YAML file, preserving comments.
    
    Args:
        path: Path to the dashboard YAML file
        
    Returns:
        Raw YAML text for the metrics section (including comments), or empty string if not found
    """
    path = Path(path)
    
    if not path.exists():
        return ""
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Find the metrics section
    metrics_start = None
    metrics_end = None
    metrics_indent = None
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Check if this is the metrics key
        if stripped.startswith('metrics:'):
            metrics_start = i
            metrics_indent = len(line) - len(stripped)
            continue
        
        # If we found metrics section, find where it ends
        if metrics_start is not None:
            # Check if we've hit the next top-level key (same or less indentation)
            if stripped and not stripped.startswith('#'):
                current_indent = len(line) - len(stripped)
                # End if we hit a top-level key (same indent as metrics:)
                if current_indent == metrics_indent and not line.strip().startswith('-'):
                    metrics_end = i
                    break
    
    if metrics_start is None:
        return ""
    
    if metrics_end is None:
        metrics_end = len(lines)
    
    # Extract the metrics section (including the "metrics:" line and everything after)
    metrics_lines = lines[metrics_start:metrics_end]
    # Remove the "metrics:" line and normalize indentation
    # Skip the first line if it's the "metrics:" key
    if metrics_lines and metrics_lines[0].lstrip().startswith('metrics:'):
        metrics_key_line = metrics_lines[0]
        metrics_indent_level = len(metrics_key_line) - len(metrics_key_line.lstrip())
        metrics_lines = metrics_lines[1:]
        
        # Find the minimum indent of list items (lines starting with '-')
        # This tells us how much items are indented relative to "metrics:"
        min_content_indent = None
        for line in metrics_lines:
            if line.strip() and not line.strip().startswith('#'):
                stripped = line.lstrip()
                # Look specifically for list items
                if stripped.startswith('-'):
                    indent = len(line) - len(stripped)
                    if min_content_indent is None or indent < min_content_indent:
                        min_content_indent = indent
        
        # If no list items found, use minimum indent of any content line
        if min_content_indent is None:
            for line in metrics_lines:
                if line.strip() and not line.strip().startswith('#'):
                    indent = len(line) - len(line.lstrip())
                    if min_content_indent is None or indent < min_content_indent:
                        min_content_indent = indent
        
        # Default to 0 if still no indent found
        if min_content_indent is None:
            min_content_indent = 0
        
        # Normalize indentation: reduce all lines by min_content_indent
        # This makes items start at column 0 in the editor
        normalized_lines = []
        for line in metrics_lines:
            if line.strip() and not line.strip().startswith('#'):
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                # Reduce indent by min_content_indent to get to column 0
                if min_content_indent is not None and min_content_indent > 0:
                    new_indent = max(0, current_indent - min_content_indent)
                else:
                    new_indent = 0
                normalized_lines.append(' ' * new_indent + stripped)
            else:
                # Handle comments - reduce their indent too, but preserve relative position
                if line.strip().startswith('#'):
                    stripped = line.lstrip()
                    current_indent = len(line) - len(stripped)
                    if min_content_indent is not None and min_content_indent > 0:
                        if current_indent >= min_content_indent:
                            new_indent = current_indent - min_content_indent
                        else:
                            # Comment is less indented than content, keep at 0
                            new_indent = 0
                        normalized_lines.append(' ' * new_indent + stripped)
                    else:
                        # No content indent found, keep comment as-is
                        normalized_lines.append(line)
                else:
                    # Empty line
                    normalized_lines.append(line)
        metrics_lines = normalized_lines
    
    return ''.join(metrics_lines)


def save_dashboard(path: str | Path, config: DashboardConfig, metrics_yaml: str | None = None) -> None:
    """
    Save a dashboard configuration to a YAML file.
    
    If metrics_yaml is provided, it will be used to preserve comments in the metrics section.
    Otherwise, the metrics will be dumped from the config dict (losing comments).
    
    Args:
        path: Path where the YAML file should be written
        config: DashboardConfig dictionary to save
        metrics_yaml: Optional raw YAML text for metrics section (preserves comments)
    """
    path = Path(path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # If metrics_yaml is provided and file exists, preserve structure
    if metrics_yaml and path.exists():
        # Read the original file
        with open(path, 'r') as f:
            original_lines = f.readlines()
        
        # Find metrics section boundaries
        metrics_start = None
        metrics_end = None
        metrics_indent = None
        
        for i, line in enumerate(original_lines):
            stripped = line.lstrip()
            if stripped.startswith('metrics:'):
                metrics_start = i
                metrics_indent = len(line) - len(stripped)
                continue
            
            if metrics_start is not None:
                if stripped and not stripped.startswith('#'):
                    current_indent = len(line) - len(stripped)
                    # End if we hit a top-level key (same indent as metrics:)
                    if current_indent == metrics_indent and not line.strip().startswith('-'):
                        metrics_end = i
                        break
        
        if metrics_start is not None:
            if metrics_end is None:
                metrics_end = len(original_lines)
            
            # Split metrics_yaml into lines and ensure proper formatting
            metrics_lines = metrics_yaml.splitlines(keepends=True)
            # If the last line doesn't end with newline, add one
            if metrics_lines and not metrics_lines[-1].endswith('\n'):
                metrics_lines[-1] = metrics_lines[-1] + '\n'
            
            # Normalize item indentation: items in editor are at column 0, need to add metrics_indent + 2
            # Find the minimum indent of list items (lines starting with '-') in the editor content
            min_indent = None
            for line in metrics_lines:
                if line.strip() and not line.strip().startswith('#'):
                    stripped = line.lstrip()
                    if stripped.startswith('-'):
                        indent = len(line) - len(stripped)
                        if min_indent is None or indent < min_indent:
                            min_indent = indent
            
            # Adjust indentation: add metrics_indent + 2 spaces for list items
            # Items in editor should be at 0, so we add metrics_indent + 2 to get them to the right level
            normalized_metrics_lines = []
            base_indent = metrics_indent + 2  # metrics_indent for "metrics:" + 2 for list items
            for line in metrics_lines:
                if line.strip() and not line.strip().startswith('#'):
                    stripped = line.lstrip()
                    current_indent = len(line) - len(stripped)
                    # Preserve relative indentation, but add base_indent
                    if min_indent is not None and min_indent >= 0:
                        relative_indent = current_indent - min_indent
                        new_indent = base_indent + relative_indent
                    else:
                        # No list items found, use base_indent
                        new_indent = base_indent
                    normalized_metrics_lines.append(' ' * new_indent + stripped)
                else:
                    # Handle comments - preserve their relative position
                    if line.strip().startswith('#'):
                        stripped = line.lstrip()
                        current_indent = len(line) - len(stripped)
                        # Comments should maintain relative position, but add base_indent
                        if min_indent is not None:
                            relative_indent = current_indent - min_indent
                            new_indent = base_indent + relative_indent
                        else:
                            new_indent = base_indent
                        normalized_metrics_lines.append(' ' * new_indent + stripped)
                    else:
                        # Empty line
                        normalized_metrics_lines.append(line)
            
            # Always prepend "metrics:" since the editor content doesn't include it
            # Determine the indent level from the original file
            metrics_indent_str = ' ' * metrics_indent if metrics_indent else ''
            metrics_lines = [f'{metrics_indent_str}metrics:\n'] + normalized_metrics_lines
            
            new_lines = (
                original_lines[:metrics_start] +
                metrics_lines +
                original_lines[metrics_end:]
            )
            
            # Also update name, defaults, and experiments from config
            result_lines = []
            i = 0
            while i < len(new_lines):
                line = new_lines[i]
                stripped = line.lstrip()
                
                # Update name
                if stripped.startswith('name:'):
                    result_lines.append(f"name: {config['name']}\n")
                    i += 1
                    continue
                
                # Update defaults section
                if stripped.startswith('defaults:'):
                    result_lines.append(line)
                    i += 1
                    # Skip existing defaults content until next top-level key
                    defaults_indent = len(line) - len(stripped)
                    while i < len(new_lines):
                        next_line = new_lines[i]
                        next_stripped = next_line.lstrip()
                        if next_stripped and not next_stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_stripped)
                            if next_indent <= defaults_indent and not next_line.strip().startswith('-'):
                                break
                        i += 1
                    # Write new defaults
                    defaults_dict = config.get('defaults', {})
                    x_key = defaults_dict.get('x_axis_key')
                    result_lines.append(f"  x_axis_key: {x_key}\n" if x_key is not None else "  x_axis_key: null\n")
                    result_lines.append(f"  x_axis_scale: {defaults_dict.get('x_axis_scale', 1.0)}\n")
                    result_lines.append(f"  smoothing: {defaults_dict.get('smoothing', 0.0)}\n")
                    continue
                
                # Update experiments section
                if stripped.startswith('experiments:'):
                    result_lines.append(line)
                    i += 1
                    # Skip existing experiments content until next top-level key
                    exp_indent = len(line) - len(stripped)
                    while i < len(new_lines):
                        next_line = new_lines[i]
                        next_stripped = next_line.lstrip()
                        if next_stripped and not next_stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_stripped)
                            if next_indent <= exp_indent and not next_line.strip().startswith('-'):
                                break
                        i += 1
                    # Write new experiments
                    for exp in config.get('experiments', []):
                        result_lines.append(f"- {exp}\n")
                    continue
                
                result_lines.append(line)
                i += 1
            
            with open(path, 'w') as f:
                f.writelines(result_lines)
            return
    
    # Fallback: write entire config (will lose comments in metrics)
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