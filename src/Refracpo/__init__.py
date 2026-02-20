"""Physical optics analysis toolkit for refractive optical systems.

Enables modeling of anti-reflection (AR) coating properties throughout the full physical-optics propagation chain.
"""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version as _version
from typing import Any

__all__ = ["__version__", "coordinate", "BeamDecoder"]

try:
    __version__ = _version("refracpo")
except PackageNotFoundError:
    __version__ = "0.0.0"

def __getattr__(name: str) -> Any:
    if name == "coordinate":
        from .coordinate import fresnel_propagate as obj
        globals()[name] = obj
        return obj
    if name == "BeamDecoder":
        from .models.net import BeamDecoder as obj
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'refracpo' has no attribute {name!r}")