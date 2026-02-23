"""
refracpo: Physical optics analysis toolkit for refractive optical systems.

Modules
-------
coordinate
    Coordinate system utilities for the full model, including coordinate-frame definitions
    and transforms. Use:
        from refracpo.coordinate import coord_sys, global_coord
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version
from typing import Any

__all__ = ["__version__", "coordinate"]

try:
    __version__ = _version("refracpo")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    """Lazy import selected submodules to keep top-level import fast."""
    if name == "coordinate":
        obj = import_module(".coordinate", __name__)
        globals()[name] = obj  # cache module object
        return obj
    raise AttributeError(f"module 'refracpo' has no attribute {name!r}")