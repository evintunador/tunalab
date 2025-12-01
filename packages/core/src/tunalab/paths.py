# packages/tunalab-core/src/tunalab/paths.py
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
        if (path / "catalogs").is_dir() and (path / "packages").is_dir():
            return path
    
    # Fallback: assume we're in a package under packages/
    return Path(__file__).resolve().parents[4]  # packages/gpt-lab-core/src/tunalab/paths.py -> root


def get_artifact_root() -> Path:
    """
    Get artifact directory. Precedence:
    1) TUNALAB_ARTIFACT_ROOT env var
    2) CWD/artifacts/ (for experiments)
    3) Repo root artifacts/
    """
    env_path = os.getenv("TUNALAB_ARTIFACT_ROOT")
    if env_path:
        p = Path(env_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    # Check if CWD has a tunalab/ folder (we're in an experiment)
    cwd_artifacts = Path.cwd() / "tunalab" / "artifacts"
    if cwd_artifacts.parent.exists():
        cwd_artifacts.mkdir(parents=True, exist_ok=True)
        return cwd_artifacts
    
    # Fallback to repo root
    repo_root = get_repo_root()
    fallback = repo_root / "artifacts"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback