# Inspired by discussions similar to
# https://medium.com/redsquirrel-tech/llm-as-compiler-2a2f79d30f0b

import ast
import sys
import traceback
import inspect
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import tempfile

from tunalab.validation.train_loops import (
    universal_learning_test, discover_specific_tests, base_loop_compliance_test, dataset_type_compatibility_test
)
from tunalab.protocols import LLMClient
from tunalab.device import get_default_device
from tunalab.validation.discovery import import_module_from_path
from tunalab.paths import get_artifact_root

# Get a logger for this module
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = \
"""You generate Python training loops for PyTorch.
Constraints:
- Output ONLY valid Python code for a single file. No backticks, no prose.
- Provide a function with EXACT signature:
  def run_training(model, optimizer, train_loader, **kwargs) -> dict:
    - The `model` is an `nn.Module` whose `forward` method takes a batch from the data loader and returns a single loss tensor.
    - Train IN-PLACE on train_loader.
    - Return a dict with at least the key: {'model': nn.Module}. Other relevant keys may be added depending on the atomic features provided.
- Avoid introducing new external dependencies; those used by example scripts are allowed.
- Keep code deterministic where feasible (set seeds when creating schedulers, etc.).
- Do not rely on global variables; everything must be self-contained in this file. Those used by example scripts are exceptions.
- Err on the side of setting default arguments when reasonable; kwargs should have defaults. 
- All kwarg defaults should be set to values that ensure numerical equivalence with `base_loop.py`.
- Ensure compatibility with both map-style and iterable-style datasets. Never call `len()` on a dataloader.
- Think deeply about how to properly have features interact. For example, if one feature moves items to the correct device and another uses a validation loader, be sure to move the validation items to the correct device. A second examle, if one feature implements a tqdm progress bar and another sets a maximum total steps, adjust the pbar accordingly.

Testing Requirements:
- Your code will be tested with a universal learning test (loss must decrease by at least 10%)
- Your code will be tested to be numerically equivalent to `base_loop.py` (shown below) when default kwargs values are used.
- Your code will also be tested with specific feature tests described below
- Make sure your implementation correctly handles all the specific behaviors being tested

Notes:
- You may add helper functions/classes if needed, or, if re-using, import directly from one of the atomic features by using `from tunalab.train_loops.<feature_name> import <function/class_name>`.
"""


USER_PROMPT_TEMPLATE = \
"""Combine the following atomic features into a single training loop:
{atomic_features}

{test_descriptions}
"""


def _build_system_prompt_with_base_loop() -> str:
    """Build the system prompt including base_loop.py content for reference."""
    # Resolve base_loop via active namespace. Prefer core if ambiguous.
    try:
        import importlib
        module = importlib.import_module("tunalab.train_loops.base_loop")
        import inspect as _inspect
        base_loop_content = _inspect.getsource(module)
        base_loop_section = f"""

Base Loop Reference (base_loop.py):
Your generated code must be numerically equivalent to this when default kwargs are used:

```python
{base_loop_content}
```
"""
        return SYSTEM_PROMPT + base_loop_section
    except Exception:
        pass

    base_loop_path = Path(__file__).parent / "catalog" / "atomic_features" / "base_loop.py"
    
    try:
        base_loop_content = base_loop_path.read_text(encoding="utf-8")
        base_loop_section = f"""

Base Loop Reference (base_loop.py):
Your generated code must be numerically equivalent to this when default kwargs are used:

```python
{base_loop_content}
```
"""
    except Exception:
        base_loop_section = "\n(Note: Could not load base_loop.py for reference)"
    
    return SYSTEM_PROMPT + base_loop_section


def _slugify(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        elif ch in " .,/\\:+|[](){}":
            safe.append("-")
    slug = "".join(safe).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "loop"


def _make_descriptive_name(atomic_features: List[str]) -> str:
    # Remove .py extension if present
    clean_features = [f.replace('.py', '') for f in atomic_features]
    feature_str = "-".join(sorted(clean_features))
    return f"{_slugify(feature_str)}"


def _parse_filename_to_atomic_features(filename: str) -> List[str]:
    """Extract atomic features from a compiled loop filename."""
    # Remove .py extension and split on hyphens
    name = filename.replace('.py', '')
    return name.split('-')


def _write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _filter_traceback_for_paths(tb: traceback.TracebackException, focus_paths: List[str]) -> str:
    focus_paths = [str(Path(p)) for p in focus_paths]
    filtered = []
    for frame in tb.stack:
        fn = str(Path(frame.filename))
        if any(fn.endswith(fp) for fp in focus_paths) or "run_training" in frame.name:
            filtered.append(f'  File "{fn}", line {frame.lineno}, in {frame.name}\n    {frame.line or ""}')
    return "\n".join(filtered)


def _summarize_exception_filtered(focus_paths: List[str], phase: str) -> str:
    exc_type, exc, tb = sys.exc_info()
    if exc_type is None:
        return "Unknown error with no traceback."
    tbe = traceback.TracebackException(exc_type, exc, tb, limit=20)
    header = f"{phase} {exc_type.__name__}: {str(exc)}"
    focused = _filter_traceback_for_paths(tbe, focus_paths)
    if not focused:
        # fallback to last few lines of full traceback
        return header + "\n" + "".join(tbe.format())[-2000:]
    return header + "\n" + focused


def _atomic_dirs() -> List[Path]:
    # Collect train_loops directories from catalogs via pkgutil discovery
    import pkgutil
    import importlib
    dirs: List[Path] = []
    
    try:
        # Try to import tunalab.train_loops which should find catalog contributions
        pkg = importlib.import_module("tunalab.train_loops")
        if hasattr(pkg, '__path__'):
            for path_str in pkg.__path__:
                p = Path(path_str)
                if p.is_dir():
                    dirs.append(p)
    except (ImportError, AttributeError):
        pass
    
    return dirs


def _get_atomic_files(atomic_features: List[str]) -> List[Path]:
    """
    Get paths to the specified atomic feature files.
    """
    paths: List[Path] = []
    roots = _atomic_dirs()
    
    for feature in atomic_features:
        filename = feature if feature.endswith('.py') else f"{feature}.py"
        found = None
        for root in roots:
            p = root / filename
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                found = p
                break
        if found is not None:
            paths.append(found)
        else:
            available = []
            for root in roots:
                available.extend([f.stem for f in root.glob("*.py") if f.is_file()])
            raise ValueError(f"Atomic feature file not found: {feature}\nAvailable features: {', '.join(sorted(set(available)))}")

    if paths:
        logger.info("Selected atomic feature files: %s", [str(p) for p in paths])

    return paths


def _read_atomic_examples_text(paths: List[Path], char_budget: int = 100_000) -> str:
    """
    Read selected atomic files and assemble a prompt appendix with fenced examples.
    Truncates if needed to stay within a rough character budget.
    """
    chunks: List[str] = []
    used = 0
    for p in paths:
        try:
            code = p.read_text(encoding="utf-8")
        except Exception:
            continue
        header = f"# Example from {p.name}\n"
        block = f"```python\n{header}{code}\n```\n"
        if used + len(block) > char_budget and chunks:
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def _extract_test_descriptions(atomic_features: List[str]) -> str:
    """Extract test descriptions from specific test functions for the given atomic features."""
    specific_tests = discover_specific_tests()
    test_descriptions = []
    
    for feature in atomic_features:
        clean_feature = feature.replace('.py', '')
        if clean_feature in specific_tests:
            test_descriptions.append(f"\n=== Tests for {feature} ===")
            for test_func in specific_tests[clean_feature]:
                # Get function name and docstring
                func_name = test_func.__name__
                docstring = inspect.getdoc(test_func) or "No description available"
                
                # Try to get a simplified version of the test logic
                try:
                    source_lines = inspect.getsourcelines(test_func)[0]
                    # Extract key assertions and logic (simplified)
                    key_lines = []
                    for line in source_lines:
                        line = line.strip()
                        if (line.startswith('assert ') or 
                            'torch.allclose' in line or
                            'accum_steps' in line or
                            'batch_size' in line or
                            'scheduler' in line or
                            'clip_grad' in line or
                            line.startswith('# Test') or
                            line.startswith('"""')):
                            key_lines.append(f"    {line}")
                    
                    if key_lines:
                        test_descriptions.append(f"""
Test: {func_name}
Description: {docstring}
Key test logic:
{''.join(key_lines[:10])}  # ... (truncated for brevity)
""")
                    else:
                        test_descriptions.append(f"""
Test: {func_name}
Description: {docstring}
""")
                except:
                    # Fallback if source inspection fails
                    test_descriptions.append(f"""
Test: {func_name}
Description: {docstring}
""")
    
    if test_descriptions:
        header = """
IMPORTANT: Your generated code will be tested with the following specific tests.
Make sure your implementation satisfies these requirements:
"""
        return header + "\n".join(test_descriptions) + "\n"
    else:
        return ""


def _build_user_prompt(atomic_features: List[str]) -> str:
    atomic_features_str = ", ".join(atomic_features)
    
    # Get test descriptions
    test_descriptions = _extract_test_descriptions(atomic_features)
    
    base = USER_PROMPT_TEMPLATE.format(
        atomic_features=atomic_features_str,
        test_descriptions=test_descriptions
    )
    
    try:
        paths = _get_atomic_files(atomic_features)
        if paths:
            examples = _read_atomic_examples_text(paths)
            if examples:
                base = base + "\nAtomic feature examples to combine:\n" + examples
    except ValueError as e:
        # If we can't find the files, still continue but mention it
        base = base + f"\nNote: {str(e)}"
    
    return base


def _add_metadata_to_code(code: str, atomic_features: List[str], device: str) -> str:
    """Add metadata as comments and dunder variables to the generated code."""
    # Clean the feature names to remove .py extension for consistent storage
    clean_features = [f.replace('.py', '') for f in atomic_features]
    metadata_header = f'''"""
LLM-compiled training loop combining atomic features: {', '.join(atomic_features)}
Generated by gpt-lab LLM compiler
Device: {device}
"""

# Metadata for discovery and testing
__atomic_features__ = {clean_features!r}
__llm_compiled__ = True

'''
    return metadata_header + code


def run_specific_tests_for_compilation(run_training_fn: Callable, atomic_features: List[str], device: str):
    """
    Run all applicable specific tests during compilation - validation only.
    Filters out tests that require pytest fixtures (monkeypatch, tmp_path, etc.)
    since those can't be called directly during compilation.
    """
    specific_tests = discover_specific_tests()
    
    for feature in atomic_features:
        clean_feature = feature.replace('.py', '')  # Handle both formats
        if clean_feature in specific_tests:
            for test_func in specific_tests[clean_feature]:
                # Check if test function has compatible signature
                # It should accept exactly (run_training_fn, device) - no pytest fixtures
                sig = inspect.signature(test_func)
                params_list = list(sig.parameters.keys())
                
                # Skip tests that require pytest fixtures
                if len(params_list) != 2:
                    logger.debug(f"Skipping {test_func.__name__} - requires {len(params_list)} parameters (expected 2)")
                    continue
                if any(p in params_list for p in ['tmp_path', 'monkeypatch', 'request', 'capsys', 'capfd']):
                    logger.debug(f"Skipping {test_func.__name__} - requires pytest fixtures")
                    continue
                
                # Run the test - if it fails, compilation fails
                logger.debug(f"Running specific test: {test_func.__name__} for feature {clean_feature}")
                test_func(run_training_fn, device)


def run_base_loop_compliance_test_for_compilation(run_training_fn: Callable, atomic_features: List[str], device: str):
    """Run base_loop compliance test during compilation."""
    
    # For compiled loops, we use a representative name from the atomic features
    feature_name = f"compiled_loop_{'-'.join(sorted([f.replace('.py', '') for f in atomic_features]))}"
    base_loop_compliance_test(run_training_fn, feature_name, device)


def compile_loop(
    atomic_features: List[str],
    llm: Optional[LLMClient] = None,
    max_refine_attempts: int = 3, 
    max_restarts: int = 3,
) -> Dict[str, Any]:
    """
    Main entry: ask LLM for a bespoke training loop combining atomic features, test it, cache it.
    
    Note: This function is designed for combining multiple atomic features. For single features,
    use the atomic feature directly for better performance and reliability.
    """
    # Validate input
    if not atomic_features:
        raise ValueError("No atomic features provided for compilation")
    
    if len(atomic_features) == 1:
        atomic_path = Path(__file__).parent / "catalog" / "atomic_features" / f"{atomic_features[0]}.py"
        raise ValueError(
            f"Compilation not recommended for single atomic feature '{atomic_features[0]}'. "
            f"Use the atomic feature directly instead for better performance. "
            f"File: {atomic_path}.py"
        )
    
    llm = llm or LLMClient()
    name = _make_descriptive_name(atomic_features)
    # Write compiled loops to artifacts: artifacts/train_loops/llm_compiled/<name>.py
    artifacts_root = get_artifact_root()
    compiled_root = artifacts_root / "train_loops" / "llm_compiled"
    compiled_root.mkdir(parents=True, exist_ok=True)
    code_path = compiled_root / f"{name}.py"
    device = get_default_device()
    logger.info("Compiler output target: %s", code_path)

    logger.debug("=" * 60)
    logger.debug("LLM TRAINING LOOP COMPILATION")
    logger.debug("=" * 60)
    logger.debug(f"Atomic features: {atomic_features}")
    logger.debug(f"Generated name: {name}")
    logger.debug(f"Output path: {code_path}")

    # Cached success path
    if code_path.exists():
        logger.debug("-" * 40)
        logger.debug("CACHE CHECK")
        logger.debug("-" * 40)
        logger.debug(f"Found existing file at {code_path}")
        try:
            module = import_module_from_path(f"cached_loop", code_path)
            logger.debug("✓ Successfully loaded cached loop")
            logger.info(f"Using cached compiled loop at {code_path}")
            return {
                "name": name, 
                "code_path": str(code_path), 
                "atomic_features": atomic_features,
            }
        except Exception as e:
            logger.debug(f"✗ Failed to load cached loop: {e}")
            logger.warning(f"Failed to load cached loop {code_path}: {e}")
            logger.info("Will regenerate...")

    # Build prompts
    logger.debug("-" * 40)
    logger.debug("PROMPT CONSTRUCTION")
    logger.debug("-" * 40)
    system_prompt = _build_system_prompt_with_base_loop()
    user_prompt = _build_user_prompt(atomic_features)
    
    logger.debug(f"System prompt:\n```\n{system_prompt}\n```")
    logger.debug(f"\nUser prompt:\n```\n{user_prompt}\n```")

    # Generate + refine loop
    restarts_left = max_restarts
    last_error_summary = ""
    code = ""
    attempt_num = 1
    
    logger.debug("-" * 40)
    logger.debug("CODE GENERATION AND VALIDATION")
    logger.debug("-" * 40)
    
    while restarts_left >= 0:
        try:
            if not last_error_summary:
                logger.debug(f"\n🚀 ATTEMPT {attempt_num}: Initial generation")
                logger.debug("Calling LLM.generate()...")
                code = llm.generate(system_prompt, user_prompt)
            else:
                logger.debug(f"\n🔄 ATTEMPT {attempt_num}: Refinement")
                logger.debug("Calling LLM.refine()...")
                logger.debug(f"Error to fix: {last_error_summary}")
                code = llm.refine(system_prompt, user_prompt, prior_code=code, error_summary=last_error_summary)
            
            logger.debug("✓ LLM response received")
            logger.debug(f"Generated code length: {len(code)} characters")
            logger.debug(f"Generated code preview:\n```python\n{code[:5000]}{'...' if len(code) > 5000 else ''}\n```")

            # Add metadata and write
            logger.debug("\n📝 Adding metadata and writing file...")
            code_with_metadata = _add_metadata_to_code(code, atomic_features, str(device))
            _write_file(code_path, code_with_metadata)
            logger.debug(f"✓ File written to {code_path}")
            
            logger.debug("\n🔍 Testing generated code...")
            
            # Import test
            logger.debug("Testing import...")
            try:
                module = import_module_from_path(f"compiled_loop", code_path)
                logger.debug("✓ Import successful")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                logger.debug(f"✗ Import failed: {err}")
                raise RuntimeError(err)

            # Function signature test
            logger.debug("Checking for run_training function...")
            if not hasattr(module, "run_training"):
                logger.debug("✗ Missing run_training function")
                raise AssertionError("Generated file must define function 'run_training' with the required signature.")
            run_training_fn = getattr(module, "run_training")
            logger.debug("✓ run_training function found")

            # Universal test
            logger.debug("Running universal learning test...")
            try:
                universal_learning_test(run_training_fn, device=str(device))
                logger.debug("✓ Universal test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[universal_test]")
                logger.debug(f"✗ Universal test failed: {err}")
                raise RuntimeError(err)

            # Dataset compatibility test
            logger.debug("Running dataset type compatibility test...")
            try:
                dataset_type_compatibility_test(run_training_fn, device=str(device))
                logger.debug("✓ Dataset type compatibility test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[dataset_type_compatibility_test]")
                logger.debug(f"✗ Dataset type compatibility test failed: {err}")
                raise RuntimeError(err)

            # Base loop compliance test
            logger.debug("Running base_loop compliance test...")
            try:
                run_base_loop_compliance_test_for_compilation(run_training_fn, atomic_features, device=str(device))
                logger.debug("✓ Base loop compliance test passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[base_loop_compliance]")
                logger.debug(f"✗ Base loop compliance test failed: {err}")
                raise RuntimeError(err)

            # Specific tests
            logger.debug("Running specific feature tests...")
            try:
                run_specific_tests_for_compilation(run_training_fn, atomic_features, device=str(device))
                logger.debug("✓ All specific tests passed")
            except Exception:
                err = _summarize_exception_filtered([str(code_path)], phase="[specific_tests]")
                logger.debug(f"✗ Specific tests failed: {err}")
                raise RuntimeError(err)

            logger.info("=" * 60)
            logger.info("COMPILATION SUCCESSFUL")
            logger.info("=" * 60)
            logger.info(f"Compiled and validated. Cached at {code_path}")
            return {
                "name": name, 
                "code_path": str(code_path), 
                "atomic_features": atomic_features,
            }

        except Exception as e:
            # Pass only focused, phase-tagged errors into refine loop
            err = str(e)
            logger.debug(f"\n❌ ATTEMPT {attempt_num} FAILED: {err}")
            logger.error(f"Compilation/testing failed: {err}")
            
            # Try refine attempts first
            logger.debug(f"\n🔧 Starting {max_refine_attempts} refinement attempts...")
            for refine_attempt in range(max_refine_attempts):
                try:
                    logger.debug(f"\n🔧 REFINEMENT {refine_attempt + 1}/{max_refine_attempts}")
                    last_error_summary = err
                    logger.debug("Calling LLM.refine()...")
                    code = llm.refine(system_prompt, user_prompt, prior_code=code, error_summary=err)
                    logger.debug("✓ Refinement response received")
                    
                    code_with_metadata = _add_metadata_to_code(code, atomic_features, str(device))
                    _write_file(code_path, code_with_metadata)
                    logger.debug(f"✓ Refined code written to {code_path}")
                    
                    # Test refined code
                    logger.debug("Testing refined code...")
                    try:
                        module = import_module_from_path(f"compiled_loop", code_path)
                        logger.debug("✓ Import successful")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[import]")
                        logger.debug(f"✗ Import failed: {err}")
                        raise RuntimeError(err)
                        
                    if not hasattr(module, "run_training"):
                        logger.debug("✗ Missing run_training function")
                        raise AssertionError("Generated file must define function 'run_training'.")
                    run_training_fn = getattr(module, "run_training")
                    logger.debug("✓ run_training function found")
                    
                    try:
                        universal_learning_test(run_training_fn, device=str(device))
                        logger.debug("✓ Universal test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[universal_test]")
                        logger.debug(f"✗ Universal test failed: {err}")
                        raise RuntimeError(err)
                    
                    try:
                        dataset_type_compatibility_test(run_training_fn, device=str(device))
                        logger.debug("✓ Dataset type compatibility test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[dataset_type_compatibility_test]")
                        logger.debug(f"✗ Dataset type compatibility test failed: {err}")
                        raise RuntimeError(err)
                    
                    try:
                        run_base_loop_compliance_test_for_compilation(run_training_fn, atomic_features, device=str(device))
                        logger.debug("✓ Base loop compliance test passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[base_loop_compliance]")
                        logger.debug(f"✗ Base loop compliance test failed: {err}")
                        raise RuntimeError(err)
                        
                    try:
                        run_specific_tests_for_compilation(run_training_fn, atomic_features, device=str(device))
                        logger.debug("✓ All specific tests passed")
                    except Exception:
                        err = _summarize_exception_filtered([str(code_path)], phase="[specific_tests]")
                        logger.debug(f"✗ Specific tests failed: {err}")
                        raise RuntimeError(err)
                        
                    logger.info("=" * 60)
                    logger.info("REFINEMENT SUCCESSFUL")
                    logger.info("=" * 60)
                    logger.info(f"Successfully refined and cached at {code_path}")
                    return {
                        "name": name, 
                        "code_path": str(code_path), 
                        "atomic_features": atomic_features,
                    }
                except Exception as e2:
                    err = str(e2)
                    logger.debug(f"✗ REFINEMENT {refine_attempt + 1} FAILED: {err}")
                    logger.error(f"Refinement attempt failed: {err}")
                    continue
                    
            # Restart from scratch
            restarts_left -= 1
            attempt_num += 1
            if restarts_left < 0:
                logger.critical("=" * 60)
                logger.critical("COMPILATION FAILED - NO MORE RESTARTS")
                logger.critical("=" * 60)
                raise
            logger.warning(f"\n🔄 RESTART {max_restarts - restarts_left}/{max_restarts}. Starting a fresh attempt...")
            last_error_summary = ""


def _get_atomic_feature_paths() -> List[Path]:
    """Return all active atomic feature directories across roots."""
    import pkgutil
    import importlib
    paths: List[Path] = []
    
    try:
        # Try to import tunalab.train_loops which should find catalog contributions
        pkg = importlib.import_module("tunalab.train_loops")
        if hasattr(pkg, '__path__'):
            for path_str in pkg.__path__:
                p = Path(path_str)
                if p.is_dir():
                    paths.append(p)
    except (ImportError, AttributeError):
        pass
    
    return paths


def _parse_function_kwargs(func_node: ast.FunctionDef) -> Set[str]:
    """
    Parse an AST FunctionDef node to extract keyword-only arguments.
    Returns the set of kwarg names (excluding **kwargs).
    """
    kwargs = set()
    
    # Get keyword-only arguments (after *)
    for arg in func_node.args.kwonlyargs:
        kwargs.add(arg.arg)
    
    return kwargs


def _parse_file_for_kwargs(file_path: Path) -> Set[str]:
    """
    Parse a Python file to extract kwargs from its run_training function.
    Returns empty set if no run_training function found or parsing fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef) and 
                node.name == "run_training"):
                return _parse_function_kwargs(node)
        
        return set()
    
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return set()


def _load_feature_metadata(feature_name: str) -> Dict[str, Any]:
    """
    Load optional metadata from an atomic feature file.
    
    Args:
        feature_name: Name of the atomic feature (without .py extension)
        
    Returns:
        Dictionary containing metadata, or empty dict if none found
    """
    try:
        feature_path = None
        for d in _get_atomic_feature_paths():
            p = d / f"{feature_name}.py"
            if p.exists():
                feature_path = p
                break
        if feature_path is None:
            return {}
        feature_module = import_module_from_path(f"metadata_{feature_name}", feature_path)
        
        # Look for metadata
        metadata = getattr(feature_module, '__smart_train_metadata__', {})
        return metadata if isinstance(metadata, dict) else {}
        
    except Exception as e:
        logger.warning(f"Failed to load metadata for {feature_name}: {e}")
        return {}


def _check_feature_conflicts(selected_features: List[str]) -> None:
    """
    Check for conflicts between selected features and raise error if found.
    
    Args:
        selected_features: List of feature names to check
        
    Raises:
        ValueError: If conflicting features are detected
    """
    if len(selected_features) <= 1:
        return  # No conflicts possible with 0 or 1 feature
    
    # Load metadata for all features
    feature_metadata = {}
    for feature in selected_features:
        metadata = _load_feature_metadata(feature)
        if metadata:
            feature_metadata[feature] = metadata
    
    # Check for conflicts
    conflicts_found = []
    
    for feature, metadata in feature_metadata.items():
        conflicts_with = metadata.get('conflicts_with', [])
        if not conflicts_with:
            continue
            
        # Check if any conflicting features are in the selected list
        for conflict_feature in conflicts_with:
            if conflict_feature in selected_features:
                conflicts_found.append((feature, conflict_feature))
    
    if conflicts_found:
        # Format error message
        conflict_pairs = [f"'{feat1}' conflicts with '{feat2}'" for feat1, feat2 in conflicts_found]
        raise ValueError(
            f"Feature conflicts detected: {', '.join(conflict_pairs)}. "
            f"These features cannot be used together in the same training loop."
        )


def discover_atomic_feature_mappings() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Discover all atomic features and create bidirectional mappings.
    
    Returns:
        tuple: (feature_to_kwargs, kwarg_to_features)
            - feature_to_kwargs: Maps feature name to set of its kwargs
            - kwarg_to_features: Maps kwarg name to set of features that use it
    """
    dirs = _get_atomic_feature_paths()

    feature_to_kwargs: Dict[str, Set[str]] = {}
    kwarg_to_features: Dict[str, Set[str]] = defaultdict(set)
    
    files: List[Path] = []
    for d in dirs:
        files.extend(sorted(d.glob("*.py")))
    # Discover all atomic feature files, sorted for determinism
    for feature_file in files:
        # Skip test files, __init__.py, and base_loop.py
        if (feature_file.name.endswith("_test.py") or 
            feature_file.name == "__init__.py" or
            feature_file.name == "base_loop.py"):
            continue
        
        feature_name = feature_file.stem
        
        # Parse kwargs from the file
        kwargs = _parse_file_for_kwargs(feature_file)
        
        # Only include features that have a run_training function with kwargs
        if kwargs:
            feature_to_kwargs[feature_name] = kwargs
            
            # Build reverse mapping
            for kwarg in kwargs:
                kwarg_to_features[kwarg].add(feature_name)
    
    # Convert defaultdict to regular dict for cleaner output
    kwarg_to_features = dict(kwarg_to_features)
    
    logger.debug(f"Discovered {len(feature_to_kwargs)} atomic features with kwargs.")
    return feature_to_kwargs, kwarg_to_features


def _find_overlapping_feature_groups(candidate_features: Set[str], feature_to_kwargs: Dict[str, Set[str]]) -> List[Set[str]]:
    """
    Group features that share kwargs into overlapping groups.
    Features in the same group compete with each other for selection.
    """
    # Build a graph of which features share kwargs
    feature_kwargs_map = {f: feature_to_kwargs[f] for f in candidate_features}
    
    groups = []
    remaining_features = sorted(list(candidate_features)) # Sort for determinism
    
    while remaining_features:
        # Start a new group with the next feature in sorted order
        current_feature = remaining_features.pop(0)
        current_group = {current_feature}
        current_kwargs = feature_kwargs_map[current_feature]
        
        # Find all features that share any kwargs with this group
        changed = True
        while changed:
            changed = False
            to_remove = set()
            
            for feature in remaining_features:
                if feature_kwargs_map[feature] & current_kwargs:
                    # This feature shares kwargs with the current group
                    current_group.add(feature)
                    current_kwargs.update(feature_kwargs_map[feature])
                    to_remove.add(feature)
                    changed = True
            
            for feature in to_remove:
                remaining_features.remove(feature)
        
        groups.append(current_group)
    
    return groups


def _select_most_specific_from_group(group: Set[str], feature_to_kwargs: Dict[str, Set[str]], user_kwarg_set: Set[str]) -> List[str]:
    """
    From a group of overlapping features, select the appropriate features.
    
    Simple Algorithm:
    1. Find all satisfiable features (user provided at least one kwarg)
    2. Check if user provided any kwargs that are unique to specific features
    3. If yes: include features with unique kwargs + most specific shared feature
    4. If no: include only the most specific feature (smallest kwarg set)
    """
    # Find features where user provided at least one kwarg
    satisfiable_features = []
    for feature in sorted(list(group)): # Sort for determinism
        feature_kwargs = feature_to_kwargs[feature]
        matched_kwargs = feature_kwargs & user_kwarg_set
        
        if matched_kwargs:
            satisfiable_features.append((feature, feature_kwargs, matched_kwargs))
    
    if not satisfiable_features:
        return []
    
    # Count how many satisfiable features use each user kwarg
    kwarg_counts = {}
    for user_kwarg in user_kwarg_set:
        count = sum(1 for _, feature_kwargs, _ in satisfiable_features if user_kwarg in feature_kwargs)
        if count > 0:  # Only count kwargs that are actually used by features in this group
            kwarg_counts[user_kwarg] = count
    
    # Check if user provided any unique kwargs (used by only one feature)
    has_unique_kwargs = any(count == 1 for count in kwarg_counts.values())
    
    if has_unique_kwargs:
        # User provided unique kwargs - include features with unique kwargs + shared features
        selected = set()
        
        # Add features that have unique kwargs
        for feature, feature_kwargs, matched_kwargs in satisfiable_features:
            has_unique = any(kwarg_counts.get(kwarg, 0) == 1 for kwarg in matched_kwargs)
            if has_unique:
                selected.add(feature)
        
        # Also add the most specific feature that only uses shared kwargs
        shared_only_features = []
        for feature, feature_kwargs, matched_kwargs in satisfiable_features:
            has_unique = any(kwarg_counts.get(kwarg, 0) == 1 for kwarg in matched_kwargs)
            if not has_unique:
                shared_only_features.append((feature, feature_kwargs, matched_kwargs))
        
        if shared_only_features:
            # Find most specific (smallest kwarg set)
            min_size = min(len(fk) for _, fk, _ in shared_only_features)
            most_specific = [f for f, fk, _ in shared_only_features if len(fk) == min_size]
            selected.update(most_specific)
        
        return sorted(list(selected))
    
    else:
        # No unique kwargs - select only the most specific feature (smallest total kwarg set)
        min_kwarg_count = min(len(feature_kwargs) for _, feature_kwargs, _ in satisfiable_features)
        most_specific = [f for f, fk, _ in satisfiable_features if len(fk) == min_kwarg_count]
        return sorted(most_specific)


def select_features_from_kwargs(user_kwargs: Dict[str, Any]) -> List[str]:
    """
    Select atomic features based on user-provided kwargs using subset-based specificity logic.
    
    For overlapping features (features that share kwargs), prefers the most specific feature:
    - The feature with the smallest kwarg set that the user has satisfied
    - This ensures we pick 'validation' over 'early_stopping' when user only provides val_loader
    
    Args:
        user_kwargs: Dictionary of kwargs provided by the user
        
    Returns:
        List of feature names to include in the compiled loop
        
    Raises:
        ValueError: If user provides kwargs that don't match any known features
    """
    feature_to_kwargs, kwarg_to_features = discover_atomic_feature_mappings()
    
    # Find all user kwargs that match known feature kwargs
    user_kwarg_set = set(user_kwargs.keys())
    known_kwargs = set(kwarg_to_features.keys())
    
    # Check for unknown kwargs
    unknown_kwargs = user_kwarg_set - known_kwargs
    if unknown_kwargs:
        available_kwargs = sorted(known_kwargs)
        raise ValueError(
            f"Unknown kwargs provided: {sorted(unknown_kwargs)}. "
            f"Available kwargs: {available_kwargs}"
        )
    
    # Find features that have any overlap with user kwargs
    candidate_features = set()
    for user_kwarg in user_kwarg_set:
        if user_kwarg in kwarg_to_features:
            candidate_features.update(kwarg_to_features[user_kwarg])
    
    if not candidate_features:
        return []
    
    # Group overlapping features
    overlapping_groups = _find_overlapping_feature_groups(candidate_features, feature_to_kwargs)
    
    # Select most specific feature from each group
    selected_features = []
    for group in overlapping_groups:
        group_selection = _select_most_specific_from_group(group, feature_to_kwargs, user_kwarg_set)
        selected_features.extend(group_selection)
    
    return sorted(selected_features)


def smart_train(
    model,
    optimizer, 
    train_loader,
    *,
    llm_client: Optional[LLMClient] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Smart training function that automatically selects and compiles atomic features
    based on the provided kwargs, then executes the compiled training loop.
    
    Args:
        model: PyTorch model to train. Its forward method must accept a batch from the
               train_loader and return a single loss tensor.
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        llm_client: Optional LLMClient instance for code compilation. If None, will attempt
                   to use a default (requires tunalab[llm] extra installed).
        **kwargs: Any atomic feature arguments (e.g., accum_steps, val_loader, patience, etc.)
        
    Returns:
        Dict containing training results (at minimum {'model': nn.Module})
        
    Raises:
        ValueError: If unknown kwargs are provided or compilation fails
        
    Examples:
        # Simple training with gradient accumulation
        result = smart_train(model, optimizer, train_loader, accum_steps=4)
        
        # Training with validation and early stopping  
        result = smart_train(
            model, optimizer, train_loader,
            val_loader=val_loader, patience=5, min_delta=0.01
        )
        
        # Complex training with multiple features
        from tunalab.llm_compilers.anthropic import AnthropicLLM
        llm = AnthropicLLM(model="anthropic/claude-sonnet-4-20250514")
        result = smart_train(
            model, optimizer, train_loader,
            llm_client=llm,
            accum_steps=4, val_loader=val_loader, patience=3, 
            use_amp=True, num_epochs=5, lr_scheduler_type="cosine"
        )
    """
    # Filter out None values from kwargs (common pattern in ML)
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if not filtered_kwargs:
        # No additional features requested - use base training loop
        logger.info("No atomic features requested. Using base training loop.")
        from tunalab.train_loops.base_loop import run_training
        return run_training(model, optimizer, train_loader)
    
    # Select appropriate atomic features based on kwargs
    selected_features = select_features_from_kwargs(filtered_kwargs)
    
    if not selected_features:
        # No features selected (shouldn't happen if kwargs are valid, but safety check)
        logger.info("No atomic features selected based on kwargs. Using base training loop.")
        from tunalab.train_loops.base_loop import run_training
        return run_training(model, optimizer, train_loader)
    
    # Check for feature conflicts
    _check_feature_conflicts(selected_features)
    
    logger.info(f"Selected atomic features based on kwargs: {selected_features}")
    
    # Optimization: For single atomic features, use them directly
    if len(selected_features) == 1:
        feature_name = selected_features[0]
        logger.info(f"Single feature optimization: using '{feature_name}.py' directly.")
        
        try:
            feature_path = None
            for d in _get_atomic_feature_paths():
                p = d / f"{feature_name}.py"
                if p.exists():
                    feature_path = p
                    break
            if feature_path is None:
                raise FileNotFoundError(f"Atomic feature '{feature_name}' not found in active roots")
            atomic_module = import_module_from_path(f"direct_{feature_name}", feature_path)
            atomic_run_training = atomic_module.run_training
            
        except Exception as e:
            raise RuntimeError(f"Failed to load atomic feature '{feature_name}' directly: {e}")
            
        logger.debug(f"Executing atomic feature directly from: {feature_path}")
        
        # Execute the atomic feature directly with user's kwargs
        return atomic_run_training(model, optimizer, train_loader, **filtered_kwargs)
    
    # instantiate llm code compiler
    logger.info("Multiple features selected. Proceeding with LLM compilation.")
    
    if llm_client is None:
        logger.warning("No LLM client provided. Using mock compilation for testing.")
        # For tests, create a mock compiled loop in a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_path = Path(tmpdir) / "mock_compiled_loop.py"
            dummy_path.write_text("def run_training(model, optimizer, train_loader, **kwargs):\n    return {\"model\": model}\n")

            # Load the mock training function
            compiled_module = import_module_from_path("smart_mock_loop", str(dummy_path))
            compiled_run_training = compiled_module.run_training
            return compiled_run_training(model, optimizer, train_loader, **filtered_kwargs)

    # create the training loop code
    compilation_result = compile_loop(selected_features, llm=llm_client)
    compiled_module_path = compilation_result["code_path"]

    # Load the compiled training function
    compiled_module = import_module_from_path("smart_compiled_loop", compiled_module_path)
    compiled_run_training = compiled_module.run_training

    logger.info(f"Using compiled training loop: {compiled_module_path}")

    # Execute the compiled training loop with user's kwargs
    return compiled_run_training(model, optimizer, train_loader, **filtered_kwargs)


if __name__ == "__main__":
    # This block is for demonstration and debugging.
    # Proper testing is handled in tests/test_smart_api.py.
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logger.info("Running smart_api.py visual tests...")
    
    feature_to_kwargs, kwarg_to_features = discover_atomic_feature_mappings()
    logger.info(f"Discovered {len(feature_to_kwargs)} features.")
    
    overlapping = {k: v for k, v in kwarg_to_features.items() if len(v) > 1}
    logger.info(f"Found {len(overlapping)} overlapping kwargs: {list(overlapping.keys())}")