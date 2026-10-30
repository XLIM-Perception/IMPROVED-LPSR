"""Public API for multi-frame license plate super-resolution."""
from ._version import __version__
from .enhancer import LPSREnhancer

__all__ = ["LPSREnhancer", "__version__"]
