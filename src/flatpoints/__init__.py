"""Arrange a point cloud on square grids: neighbours on the grid are neighbours on the surface."""
from .api import Flat
from .viz import plot, imshow

__all__ = ["Flat", "plot", "imshow"]
__version__ = "0.1.0"
