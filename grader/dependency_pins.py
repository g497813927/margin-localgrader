"""Read an exact runtime pin from a checked-in requirements file."""
from __future__ import annotations

from pathlib import Path
import re


def pinned_version(path: Path, package: str) -> str:
    prefix = f"{package}=="
    pins = [line[len(prefix):].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(prefix)]
    if len(pins) != 1 or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", pins[0]):
        raise ValueError(f"Expected one exact {package} pin in {path.name}.")
    return pins[0]
