from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)

    # Minimal fallback for environments without PyYAML is intentionally absent.
    # This project expects PyYAML in real deployment; demo checks will reveal it.
    raise RuntimeError("PyYAML is required: pip install pyyaml")
