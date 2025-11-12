"""Version information for COLA package."""
import os
from pathlib import Path

__version__ = "0.1.0"

# Try to read version from VERSION file in project root
version_file = Path(__file__).parent.parent / "VERSION"
if version_file.exists():
    __version__ = version_file.read_text().strip()

__version_info__ = tuple(int(x) for x in __version__.split("."))

