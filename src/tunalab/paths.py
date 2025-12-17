import os
from pathlib import Path

def get_repo_root() -> Path:
    """Find the repository root by walking up from CWD or this file."""
    # Check env override first
    env_root = os.getenv("TUNALAB_ROOT")
    if env_root:
        return Path(env_root).resolve()
    
    # Walk up from CWD looking for markers
    for path in [Path.cwd()] + list(Path.cwd().parents):
        if (path / ".tunalab_root").exists():
            return path
        if (path / "catalogs").is_dir() and (path / "src").is_dir():
            return path
    
    raise FileNotFoundError(
        "Could not find tunalab repository root. "
        "Ensure you are running from within the repository or set TUNALAB_ROOT environment variable."
    )


def get_artifact_root() -> Path:
    """
    Get artifact directory. Precedence:
    1) TUNALAB_ARTIFACT_ROOT env var
    2) Catalog artifacts/ (if CWD is within a catalog's benchmarks/)
    3) Experiment artifacts/ (if in an experiment directory)
    4) Standalone project artifacts/ (CWD/artifacts)
    5) Monorepo root artifacts/
    """
    env_path = os.getenv("TUNALAB_ARTIFACT_ROOT")
    if env_path:
        p = Path(env_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    # Check if we're inside a catalog's benchmarks/ directory structure
    # Walk up from CWD to see if we're under a benchmarks/ directory
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        # If this parent is named "benchmarks", check if its parent is a catalog
        if parent.name == "benchmarks":
            catalog_root = parent.parent
            # Verify it's a catalog: has src/tunalab/ and pyproject.toml
            if (catalog_root / "src" / "tunalab").exists() and \
               (catalog_root / "pyproject.toml").exists():
                catalog_artifacts = catalog_root / "artifacts"
                catalog_artifacts.mkdir(parents=True, exist_ok=True)
                return catalog_artifacts
    
    # Also check if CWD itself is a catalog root
    if (cwd / "benchmarks").exists() and \
       (cwd / "src" / "tunalab").exists() and \
       (cwd / "pyproject.toml").exists():
        catalog_artifacts = cwd / "artifacts"
        catalog_artifacts.mkdir(parents=True, exist_ok=True)
        return catalog_artifacts
    
    # Check if we're in an experiment (has tunalab/ folder or existing artifacts/)
    cwd_artifacts = cwd / "artifacts"
    if cwd_artifacts.exists() or (cwd / "tunalab").exists():
        cwd_artifacts.mkdir(parents=True, exist_ok=True)
        return cwd_artifacts
    
    # Try to find monorepo root, but gracefully fall back
    try:
        repo_root = get_repo_root()
        fallback = repo_root / "artifacts"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except FileNotFoundError:
        # Standalone project - just use CWD/artifacts
        fallback = cwd / "artifacts"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
