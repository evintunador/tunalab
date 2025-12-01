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
    2) CWD/artifacts/ (for experiments)
    3) Repo root artifacts/
    """
    env_path = os.getenv("TUNALAB_ARTIFACT_ROOT")
    if env_path:
        p = Path(env_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    # Check if CWD has a tunalab/ folder (we're in an experiment)
    cwd_artifacts = Path.cwd() / "artifacts"
    if cwd_artifacts.exists() or (Path.cwd() / "tunalab").exists():
        cwd_artifacts.mkdir(parents=True, exist_ok=True)
        return cwd_artifacts
    
    # Fallback to repo root (only works if in monorepo)
    try:
        repo_root = get_repo_root()
        fallback = repo_root / "artifacts"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except FileNotFoundError:
        # Not in monorepo, just use CWD/artifacts
        fallback = Path.cwd() / "artifacts"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
