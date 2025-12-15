from pathlib import Path
from typing import Dict, List, TypedDict

import yaml


class MetricConfig(TypedDict):
    name: str
    keys: List[str]


class AnalysisDefaults(TypedDict):
    x_axis_key: str | None
    x_axis_scale: float
    smoothing: float


class AnalysisConfig(TypedDict):
    name: str
    defaults: AnalysisDefaults
    experiments: List[str]  # Glob patterns
    metrics: List[MetricConfig]


def find_configs(root_dir: str | Path = ".") -> List[Path]:
    root_path = Path(root_dir)
    configs = sorted(root_path.rglob("*.experiments.yaml"))
    return configs


def load_config(path: str | Path) -> AnalysisConfig:
    path = Path(path)
    
    with open(path, 'r') as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file {path}: {e}") from e
    
    defaults = data.get("defaults", {})
    if "x_axis_scale" not in defaults:
        defaults["x_axis_scale"] = 1.0
    if "x_axis_key" not in defaults:
        defaults["x_axis_key"] = None
    if "smoothing" not in defaults:
        defaults["smoothing"] = 0.0
    
    config: AnalysisConfig = {
        "name": data.get("name", ""),
        "defaults": defaults,
        "experiments": data.get("experiments", []),
        "metrics": data.get("metrics", []),
    }
    
    return config


def extract_metrics_section(path: str | Path) -> str:
    """Extract the raw metrics section from a YAML file, preserving comments."""
    path = Path(path)
    
    if not path.exists():
        return ""
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    metrics_start = None
    metrics_end = None
    metrics_indent = None
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('metrics:'):
            metrics_start = i
            metrics_indent = len(line) - len(stripped)
            continue
        
        if metrics_start is not None:
            if stripped and not stripped.startswith('#'):
                current_indent = len(line) - len(stripped)
                if current_indent == metrics_indent and not line.strip().startswith('-'):
                    metrics_end = i
                    break
    
    if metrics_start is None:
        return ""
    
    if metrics_end is None:
        metrics_end = len(lines)
    
    metrics_lines = lines[metrics_start:metrics_end]
    if metrics_lines and metrics_lines[0].lstrip().startswith('metrics:'):
        metrics_key_line = metrics_lines[0]
        metrics_indent_level = len(metrics_key_line) - len(metrics_key_line.lstrip())
        metrics_lines = metrics_lines[1:]
        
        min_content_indent = None
        for line in metrics_lines:
            if line.strip() and not line.strip().startswith('#'):
                stripped = line.lstrip()
                if stripped.startswith('-'):
                    indent = len(line) - len(stripped)
                    if min_content_indent is None or indent < min_content_indent:
                        min_content_indent = indent
        
        if min_content_indent is None:
            for line in metrics_lines:
                if line.strip() and not line.strip().startswith('#'):
                    indent = len(line) - len(line.lstrip())
                    if min_content_indent is None or indent < min_content_indent:
                        min_content_indent = indent
        
        if min_content_indent is None:
            min_content_indent = 0
        
        normalized_lines = []
        for line in metrics_lines:
            if line.strip() and not line.strip().startswith('#'):
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                if min_content_indent is not None and min_content_indent > 0:
                    new_indent = max(0, current_indent - min_content_indent)
                else:
                    new_indent = 0
                normalized_lines.append(' ' * new_indent + stripped)
            else:
                if line.strip().startswith('#'):
                    stripped = line.lstrip()
                    current_indent = len(line) - len(stripped)
                    if min_content_indent is not None and min_content_indent > 0:
                        if current_indent >= min_content_indent:
                            new_indent = current_indent - min_content_indent
                        else:
                            new_indent = 0
                        normalized_lines.append(' ' * new_indent + stripped)
                    else:
                        normalized_lines.append(line)
                else:
                    normalized_lines.append(line)
        metrics_lines = normalized_lines
    
    return ''.join(metrics_lines)


def save_config(path: str | Path, config: AnalysisConfig, metrics_yaml: str | None = None) -> None:
    path = Path(path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if metrics_yaml and path.exists():
        with open(path, 'r') as f:
            original_lines = f.readlines()
        
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
                    if current_indent == metrics_indent and not line.strip().startswith('-'):
                        metrics_end = i
                        break
        
        if metrics_start is not None:
            if metrics_end is None:
                metrics_end = len(original_lines)
            
            metrics_lines = metrics_yaml.splitlines(keepends=True)
            if metrics_lines and not metrics_lines[-1].endswith('\n'):
                metrics_lines[-1] = metrics_lines[-1] + '\n'
            
            min_indent = None
            for line in metrics_lines:
                if line.strip() and not line.strip().startswith('#'):
                    stripped = line.lstrip()
                    if stripped.startswith('-'):
                        indent = len(line) - len(stripped)
                        if min_indent is None or indent < min_indent:
                            min_indent = indent
            
            normalized_metrics_lines = []
            base_indent = metrics_indent + 2
            for line in metrics_lines:
                if line.strip() and not line.strip().startswith('#'):
                    stripped = line.lstrip()
                    current_indent = len(line) - len(stripped)
                    if min_indent is not None and min_indent >= 0:
                        relative_indent = current_indent - min_indent
                        new_indent = base_indent + relative_indent
                    else:
                        new_indent = base_indent
                    normalized_metrics_lines.append(' ' * new_indent + stripped)
                else:
                    if line.strip().startswith('#'):
                        stripped = line.lstrip()
                        current_indent = len(line) - len(stripped)
                        if min_indent is not None:
                            relative_indent = current_indent - min_indent
                            new_indent = base_indent + relative_indent
                        else:
                            new_indent = base_indent
                        normalized_metrics_lines.append(' ' * new_indent + stripped)
                    else:
                        normalized_metrics_lines.append(line)
            
            metrics_indent_str = ' ' * metrics_indent if metrics_indent else ''
            metrics_lines = [f'{metrics_indent_str}metrics:\n'] + normalized_metrics_lines
            
            new_lines = (
                original_lines[:metrics_start] +
                metrics_lines +
                original_lines[metrics_end:]
            )
            
            result_lines = []
            i = 0
            while i < len(new_lines):
                line = new_lines[i]
                stripped = line.lstrip()
                
                if stripped.startswith('name:'):
                    result_lines.append(f"name: {config['name']}\n")
                    i += 1
                    continue
                
                if stripped.startswith('defaults:'):
                    result_lines.append(line)
                    i += 1
                    defaults_indent = len(line) - len(stripped)
                    while i < len(new_lines):
                        next_line = new_lines[i]
                        next_stripped = next_line.lstrip()
                        if next_stripped and not next_stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_stripped)
                            if next_indent <= defaults_indent and not next_line.strip().startswith('-'):
                                break
                        i += 1
                    defaults_dict = config.get('defaults', {})
                    x_key = defaults_dict.get('x_axis_key')
                    result_lines.append(f"  x_axis_key: {x_key}\n" if x_key is not None else "  x_axis_key: null\n")
                    result_lines.append(f"  x_axis_scale: {defaults_dict.get('x_axis_scale', 1.0)}\n")
                    result_lines.append(f"  smoothing: {defaults_dict.get('smoothing', 0.0)}\n")
                    continue
                
                if stripped.startswith('experiments:'):
                    result_lines.append(line)
                    i += 1
                    exp_indent = len(line) - len(stripped)
                    while i < len(new_lines):
                        next_line = new_lines[i]
                        next_stripped = next_line.lstrip()
                        if next_stripped and not next_stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_stripped)
                            if next_indent <= exp_indent and not next_line.strip().startswith('-'):
                                break
                        i += 1
                    for exp in config.get('experiments', []):
                        result_lines.append(f"- {exp}\n")
                    continue
                
                result_lines.append(line)
                i += 1
            
            with open(path, 'w') as f:
                f.writelines(result_lines)
            return
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

