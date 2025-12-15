"""Multi-run hyperparameter sweep tool with reproducibility tracking and retry logic."""

import argparse
import datetime
import json
import logging
import queue
import subprocess
import sys
import time
import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from tunalab.configuration import compose_config
from tunalab.reproducibility import ReproducibilityManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class RunEntry(dict):
    """Manifest entry for a single run."""
    
    def __init__(
        self,
        run_id: str,
        run_index: int,
        parameters: Dict[str, Any],
        command: str,
        status: Literal['pending', 'running', 'success', 'failed'] = 'pending',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        exit_code: Optional[int] = None,
        job_id: Optional[str] = None
    ):
        super().__init__(
            run_id=run_id,
            run_index=run_index,
            parameters=parameters,
            command=command,
            status=status,
            start_time=start_time,
            end_time=end_time,
            exit_code=exit_code,
            job_id=job_id
        )


def create_sweep_directory(config: Dict[str, Any]) -> Path:
    name = config.get('name', 'sweep')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = Path('sweeps') / f"{name}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / 'logs').mkdir(exist_ok=True)
    
    with open(sweep_dir / 'sweep_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created sweep directory: {sweep_dir}")
    return sweep_dir


def write_manifest_skeleton(sweep_dir: Path, runs: List[RunEntry]):
    manifest_path = sweep_dir / 'runs_manifest.jsonl'
    with open(manifest_path, 'w') as f:
        for run in runs:
            f.write(json.dumps(dict(run)) + '\n')
    logger.info(f"Wrote manifest skeleton with {len(runs)} runs")


def update_manifest_entry(sweep_dir: Path, run_id: str, updates: Dict[str, Any]):
    manifest_path = sweep_dir / 'runs_manifest.jsonl'
    
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['run_id'] == run_id:
                entry.update(updates)
            entries.append(entry)
    
    with open(manifest_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def load_failed_runs(sweep_dir: Path) -> List[RunEntry]:
    manifest_path = sweep_dir / 'runs_manifest.jsonl'
    failed_runs = []
    
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['status'] == 'failed':
                entry['status'] = 'pending'
                entry['exit_code'] = None
                entry['start_time'] = None
                entry['end_time'] = None
                failed_runs.append(RunEntry(**entry))
    
    return failed_runs


def detect_old_parameter_format(parameters: Dict[str, Any]) -> bool:
    if 'grid' in parameters or 'static' in parameters or 'paired' in parameters:
        return False
    return any(isinstance(v, list) for v in parameters.values())


def print_migration_warning():
    logger.warning("=" * 60)
    logger.warning("DEPRECATED PARAMETER FORMAT")
    logger.warning("")
    logger.warning("Old: parameters: {learning_rate: [0.001, 0.01], device: cuda}")
    logger.warning("New: parameters: {grid: {learning_rate: [0.001, 0.01]}, static: {device: cuda}}")
    logger.warning("=" * 60)


def expand_parameters(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    
    parameters = config.get('parameters', {})
    
    grid_params = parameters.get('grid', {})
    static_params = parameters.get('static', {})
    paired_groups = parameters.get('paired', [])
    
    grid_keys = set(grid_params.keys())
    paired_keys = set()
    for group in paired_groups:
        for combo in group.get('combinations', []):
            paired_keys.update(combo.keys())
    
    conflicts = grid_keys & paired_keys
    if conflicts:
        raise ValueError(f"Parameters appear in both grid and paired: {conflicts}")
    
    grid_combos = _expand_grid(grid_params)
    paired_combos = _expand_paired(paired_groups)
    
    all_combos = []
    for g in grid_combos:
        for p in paired_combos:
            combo = {**static_params, **g, **p}
            all_combos.append(combo)
    
    return all_combos


def _expand_grid(grid_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not grid_params:
        return [{}]
    
    expanded = {}
    for key, value in grid_params.items():
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], list):
                expanded[key] = [value[0]]  # [[...]] -> single list value
            else:
                expanded[key] = value
        else:
            expanded[key] = [value]
    
    keys = list(expanded.keys())
    values = [expanded[k] for k in keys]
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def _expand_paired(paired_groups: List[Dict]) -> List[Dict[str, Any]]:
    if not paired_groups:
        return [{}]
    
    all_combinations = [{}]
    for group in paired_groups:
        combos = group.get('combinations', [])
        all_combinations = [
            {**base, **combo}
            for base in all_combinations
            for combo in combos
        ]
    
    return all_combinations


def build_command(command_config: Dict[str, Any], params: Dict[str, Any]) -> str:
    cmd_type = command_config.get('type', 'python')
    
    if cmd_type not in ['python', 'torchrun', 'slurm']:
        raise ValueError(f"Unsupported command type: {cmd_type}")
    
    script = command_config['script']
    
    param_parts = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                param_parts.append(f"--{key}")
        elif isinstance(value, list):
            import json
            param_parts.append(f"--{key} '{json.dumps(value)}'")
        else:
            param_parts.append(f"--{key} {value}")
    
    param_string = " ".join(param_parts)
    
    if cmd_type == 'python':
        command = f"python {script} {param_string}"
    elif cmd_type == 'torchrun':
        nproc = command_config.get('nproc_per_node', 1)
        torchrun_flags = []
        if 'nnodes' in command_config:
            torchrun_flags.append(f"--nnodes={command_config['nnodes']}")
        if 'node_rank' in command_config:
            torchrun_flags.append(f"--node_rank={command_config['node_rank']}")
        if 'master_addr' in command_config:
            torchrun_flags.append(f"--master_addr={command_config['master_addr']}")
        if 'master_port' in command_config:
            torchrun_flags.append(f"--master_port={command_config['master_port']}")
        
        torchrun_flag_str = " ".join(torchrun_flags)
        command = f"torchrun --nproc_per_node={nproc} {torchrun_flag_str} {script} {param_string}".strip()
    elif cmd_type == 'slurm':
        command = f"python {script} {param_string}"
    
    return command.strip()


def execute_command_with_logs(
    command: str,
    sweep_dir: Path,
    run_id: str,
    run_index: int
) -> Tuple[int, str, str]:
    stdout_path = sweep_dir / 'logs' / f'run_{run_index}.stdout.txt'
    stderr_path = sweep_dir / 'logs' / f'run_{run_index}.stderr.txt'
    
    with open(stdout_path, 'w') as stdout_f, \
         open(stderr_path, 'w') as stderr_f:
        result = subprocess.run(
            command,
            shell=True,
            stdout=stdout_f,
            stderr=stderr_f
        )
    
    return result.returncode, str(stdout_path), str(stderr_path)


class CommandExecutor(ABC):
    @abstractmethod
    def execute(self, command: str, sweep_dir: Path, run_entry: RunEntry) -> RunEntry:
        pass


class PythonExecutor(CommandExecutor):
    def execute(self, command: str, sweep_dir: Path, run_entry: RunEntry) -> RunEntry:
        run_entry['status'] = 'running'
        run_entry['start_time'] = datetime.datetime.now().isoformat()
        update_manifest_entry(sweep_dir, run_entry['run_id'], run_entry)
        
        exit_code, stdout_path, stderr_path = execute_command_with_logs(
            command, sweep_dir, run_entry['run_id'], run_entry['run_index']
        )
        
        run_entry['status'] = 'success' if exit_code == 0 else 'failed'
        run_entry['exit_code'] = exit_code
        run_entry['end_time'] = datetime.datetime.now().isoformat()
        return run_entry


class TorchrunExecutor(PythonExecutor):
    pass


class SlurmExecutor(CommandExecutor):
    def execute(self, command: str, sweep_dir: Path, run_entry: RunEntry) -> RunEntry:
        stdout_path = sweep_dir / 'logs' / f"run_{run_entry['run_index']}.stdout.txt"
        stderr_path = sweep_dir / 'logs' / f"run_{run_entry['run_index']}.stderr.txt"
        
        sbatch_cmd = [
            'sbatch',
            '--parsable',
            f'--output={stdout_path}',
            f'--error={stderr_path}',
            '--wrap',
            command
        ]
        
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to submit SLURM job: {result.stderr}")
            run_entry['status'] = 'failed'
            run_entry['exit_code'] = result.returncode
            return run_entry
        
        job_id = result.stdout.strip()
        
        run_entry['status'] = 'running'
        run_entry['job_id'] = job_id
        run_entry['start_time'] = datetime.datetime.now().isoformat()
        
        logger.info(f"Submitted SLURM job {job_id} for run {run_entry['run_index']}")
        
        return run_entry


def get_executor(cmd_type: str) -> CommandExecutor:
    if cmd_type == 'python':
        return PythonExecutor()
    elif cmd_type == 'torchrun':
        return TorchrunExecutor()
    elif cmd_type == 'slurm':
        return SlurmExecutor()
    else:
        raise ValueError(f"Unsupported command type: {cmd_type}")


def poll_slurm_jobs(sweep_dir: Path, active_jobs: Dict[str, RunEntry]):
    for job_id, run_entry in list(active_jobs.items()):
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h'],
            capture_output=True,
            text=True
        )
        
        if not result.stdout:
            sacct_result = subprocess.run(
                ['sacct', '-j', job_id, '--format=ExitCode', '--noheader'],
                capture_output=True,
                text=True
            )
            
            exit_code = -1
            if sacct_result.stdout:
                exit_code_str = sacct_result.stdout.strip().split(':')[0]
                try:
                    exit_code = int(exit_code_str)
                except ValueError:
                    exit_code = -1
            
            run_entry['status'] = 'success' if exit_code == 0 else 'failed'
            run_entry['exit_code'] = exit_code
            run_entry['end_time'] = datetime.datetime.now().isoformat()
            update_manifest_entry(sweep_dir, run_entry['run_id'], run_entry)
            
            logger.info(f"SLURM job {job_id} completed with exit code {exit_code}")
            del active_jobs[job_id]


def execute_slurm_parallel(
    commands: List[Tuple[str, RunEntry]],
    sweep_dir: Path,
    max_workers: Optional[int] = None
):
    executor = SlurmExecutor()
    
    if max_workers is None:
        logger.info("Submitting all SLURM jobs (unlimited parallelism)")
        active_jobs = {}
        for cmd, run_entry in commands:
            updated_entry = executor.execute(cmd, sweep_dir, run_entry)
            update_manifest_entry(sweep_dir, updated_entry['run_id'], updated_entry)
            if updated_entry.get('job_id'):
                active_jobs[updated_entry['job_id']] = updated_entry
        
        while active_jobs:
            poll_slurm_jobs(sweep_dir, active_jobs)
            time.sleep(30)
    else:
        logger.info(f"Submitting SLURM jobs with max_workers={max_workers}")
        job_queue = queue.Queue()
        for item in commands:
            job_queue.put(item)
        
        active_jobs = {}
        while not job_queue.empty() or active_jobs:
            while len(active_jobs) < max_workers and not job_queue.empty():
                cmd, run_entry = job_queue.get()
                updated_entry = executor.execute(cmd, sweep_dir, run_entry)
                update_manifest_entry(sweep_dir, updated_entry['run_id'], updated_entry)
                if updated_entry.get('job_id'):
                    active_jobs[updated_entry['job_id']] = updated_entry
            
            poll_slurm_jobs(sweep_dir, active_jobs)
            time.sleep(30)


def format_run_display(run_entry: RunEntry, total_runs: int) -> str:
    params = run_entry['parameters']
    exclude_keys = {'config', 'device'}
    param_str = ', '.join(
        f"{k}={v}" for k, v in params.items() 
        if k not in exclude_keys
    )
    
    if not param_str:
        param_str = "default params"
    
    return f"Run {run_entry['run_index']}/{total_runs}: {param_str}"


def print_run_start(run_entry: RunEntry, sweep_dir: Path, total_runs: int):
    print(f"\n{'='*70}")
    print(format_run_display(run_entry, total_runs))
    print(f"  Command: {run_entry['command']}")
    print(f"  Logs: {sweep_dir}/logs/run_{run_entry['run_index']}.{{stdout,stderr}}.txt")
    print(f"\n  To follow progress, run:")
    print(f"  tail -f {sweep_dir}/logs/run_{run_entry['run_index']}.stdout.txt")
    print(f"{'='*70}\n")


def execute_with_retry(
    command: str,
    sweep_dir: Path,
    run_entry: RunEntry,
    executor: CommandExecutor,
    max_retries: int = 0
) -> RunEntry:
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"Retrying {run_entry['run_id']} (attempt {attempt + 1}/{max_retries + 1})")
        
        result_entry = executor.execute(command, sweep_dir, run_entry)
        
        if result_entry['status'] == 'success':
            return result_entry
        
        update_manifest_entry(sweep_dir, result_entry['run_id'], result_entry)
    
    return result_entry


def execute_runs(
    config: Dict[str, Any],
    runs: List[RunEntry],
    sweep_dir: Path
):
    command_config = config['command']
    execution_config = config.get('execution', {})
    
    cmd_type = command_config.get('type', 'python')
    mode = execution_config.get('mode', 'sequential')
    max_workers = execution_config.get('max_workers')
    max_retries = execution_config.get('max_retries', 0)
    
    total_runs = len(runs)
    
    if cmd_type == 'slurm':
        commands = [(run['command'], run) for run in runs]
        execute_slurm_parallel(commands, sweep_dir, max_workers)
        return
    
    executor = get_executor(cmd_type)
    
    if mode == 'sequential' or max_workers == 1:
        logger.info("Executing runs sequentially")
        for run_entry in runs:
            print_run_start(run_entry, sweep_dir, total_runs)
            result_entry = execute_with_retry(
                run_entry['command'],
                sweep_dir,
                run_entry,
                executor,
                max_retries
            )
            update_manifest_entry(sweep_dir, result_entry['run_id'], result_entry)
    
    elif mode == 'parallel':
        if max_workers is None:
            max_workers = len(runs)
        logger.info(f"Executing runs in parallel (max_workers={max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as pool_executor:
            futures = {}
            for run_entry in runs:
                print_run_start(run_entry, sweep_dir, total_runs)
                future = pool_executor.submit(
                    execute_with_retry,
                    run_entry['command'],
                    sweep_dir,
                    run_entry,
                    executor,
                    max_retries
                )
                futures[future] = run_entry
            
            for future in as_completed(futures):
                result_entry = future.result()
                update_manifest_entry(sweep_dir, result_entry['run_id'], result_entry)
    
    else:
        raise ValueError(f"Unsupported execution mode: {mode}")


def multi_run(config: Dict[str, Any]) -> Dict[str, Any]:
    name = config.get('name', 'unnamed_sweep')
    logger.info(f"Starting multi-run sweep: {name}")
    
    sweep_dir = create_sweep_directory(config)
    
    rm = ReproducibilityManager(output_dir=str(sweep_dir), is_main_process=True)
    logger.info("Initialized reproducibility tracking for sweep")

    with rm:
        
        runs = []

        param_combinations = expand_parameters(config)
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        command_config = config['command']
        runs = []
        for i, params in enumerate(param_combinations):
            command = build_command(command_config, params)
            run_entry = RunEntry(
                run_id=f"run_{i+1}",
                run_index=i+1,
                parameters=params,
                command=command,
                status='pending'
            )
            runs.append(run_entry)
        
        if 'raw_commands' in config:
            raw_commands = config['raw_commands']
            logger.info(f"Using {len(raw_commands)} raw commands")
            for i, cmd in enumerate(raw_commands):
                run_entry = RunEntry(
                    run_id=f"run_{i+1}",
                    run_index=i+1,
                    parameters={},
                    command=cmd,
                    status='pending'
                )
                runs.append(run_entry)
        
        write_manifest_skeleton(sweep_dir, runs)
        
        execute_runs(config, runs, sweep_dir)
        
        manifest_path = sweep_dir / 'runs_manifest.jsonl'
        successful = 0
        failed = 0
        
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['status'] == 'success':
                    successful += 1
                elif entry['status'] == 'failed':
                    failed += 1
        
        summary = {
            'name': name,
            'sweep_dir': str(sweep_dir),
            'total_runs': len(runs),
            'successful': successful,
            'failed': failed
        }
        
        logger.info(f"Multi-run complete: {successful}/{len(runs)} successful")
        
        return summary


def format_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        f"Multi-run sweep '{summary['name']}' completed",
        f"Sweep directory: {summary['sweep_dir']}",
        f"Total runs: {summary['total_runs']}",
        f"Successful: {summary['successful']}",
        f"Failed: {summary['failed']}",
        "=" * 70,
    ]
    return "\n" + "\n".join(lines)


def validate_config(config: dict) -> list[str]:
    if 'raw_commands' in config:
        return []
    
    required_fields = ['command']
    missing_fields = [field for field in required_fields if field not in config]
    
    if 'parameters' not in config and 'raw_commands' not in config:
        missing_fields.append('parameters')
    
    return missing_fields


def execute_multi_run(config: dict) -> int:
    
    missing_fields = validate_config(config)
    if missing_fields:
        logger.error(f"Configuration is missing required fields: {missing_fields}")
        return 1
    
    try:
        summary = multi_run(config)
        print(format_summary(summary))
        return 1 if summary['failed'] > 0 else 0
    
    except Exception as e:
        logger.error(f"Multi-run failed with error: {e}", exc_info=True)
        return 1


def _main_impl():
    """Internal implementation for the multi-run command."""
    parser = argparse.ArgumentParser(
        description="Run multiple experiments with different parameter configurations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    config = compose_config(parser)
    exit_code = execute_multi_run(config)
    sys.exit(exit_code)


class Command:
    """Multi-run command for hyperparameter sweeps."""
    
    name = "multi-run"
    description = "Run hyperparameter sweeps across multiple configurations"
    
    @staticmethod
    def main():
        _main_impl()


if __name__ == "__main__":
    _main_impl()

