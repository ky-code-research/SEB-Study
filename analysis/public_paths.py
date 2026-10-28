from __future__ import annotations

import os
from pathlib import Path

# Repository root. Override only when the analysis sources are embedded elsewhere.
ROOT = Path(os.environ.get("PROUD_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()

def authorized_data_path(env_name: str, default_name: str) -> Path:
    """Resolve a restricted local input without embedding a machine-specific path."""
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else ROOT / "data" / default_name

