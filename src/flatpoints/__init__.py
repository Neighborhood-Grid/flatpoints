"""Arrange a point cloud on square grids: neighbours on the grid are neighbours on the surface."""
from .main import Corners, Flat, Resampler
from .viz import plot, plot_resampler, imshow

__all__ = ["Flat", "Corners", "Resampler", "plot", "plot_resampler", "imshow"]
__version__ = "0.1.0"
