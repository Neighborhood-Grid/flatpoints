"""Flatten a point cloud into square images: one pixel per point, neighbour pixels hold neighbour points."""
from .main import Compressor, Flat, Resampler

__all__ = ["Flat", "Resampler", "Compressor"]
__version__ = "0.1.0"
