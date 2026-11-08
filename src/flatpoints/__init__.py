"""Flatten a point cloud into square images: one pixel per point, neighbour pixels hold neighbour points."""
from .datasets import datasets, load
from .examples import exemple_compression, exemple_flat, exemple_sampler
from .main import Compressor, Flat, Resampler

__all__ = ["Flat", "Resampler", "Compressor", "load", "datasets",
           "exemple_flat", "exemple_sampler", "exemple_compression"]
__version__ = "0.1.0"
