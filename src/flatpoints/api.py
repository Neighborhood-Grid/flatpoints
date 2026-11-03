"""User-facing class of flatpoints: `Flat`.

`Flat` takes points sampled on a surface and assigns each of them to a cell of a square
grid, so that neighbouring cells hold points that are close *on the surface*.
Grid-based tools (convolutions, images, ...) can then be applied to the cloud.

A closed surface (e.g. an animal, a sphere) cannot be flattened in one piece: it is cut
in two faces, "left" and "right", by a plane perpendicular to a chosen `lateral` axis
(x, y or z), each face laid out on its own grid (see `core.layout`). An open surface
(e.g. a terrain) fits on a single grid, the face "main" (single_face=True).
"""
import numpy as np

from .core import layout
from .utils import axes, fill_empty, split


class Flat:
    """A point cloud arranged on square grids, neighbours on the grid being neighbours
    on the surface.

    Args:
        points: (N, 3) array of xyz coordinates.
        single_face: False (default) for a closed surface, cut in two faces "left" and
            "right". True for an open surface: one face "main".
        lateral: "x", "y" or "z", the axis across which the surface is cut (default x).
        up: "x", "y" or "z", the vertical axis (default z); v follows it.
        verbose: print progress.

    Attributes:
        faces: names of the faces, ("left", "right") or ("main",).
        n: side of the grids.
        indexes: {face: (n, n) int array}, the index in `points` of each cell's point.
        isomap: {face: (n, n, 2) float array}, the flat (u, v) coordinates of each
            cell's point, in [0, 1].

        The grids read like images: u increases from left to right, v from bottom to top
        (row 0 is the top).

    Example:
        flat = Flat(points)
        idx = flat.indexes["left"]     # (n, n)
        uv = flat.isomap["left"]       # (n, n, 2)
        xyz = points[idx]              # (n, n, 3)
    """

    def __init__(self, points, single_face=False, lateral="x", up="z", verbose=False):
        self.points = np.asarray(points, float)
        self._lateral, self._up = lateral, up
        self.faces = ("main",) if single_face else ("left", "right")
        self.indexes, self.isomap = {}, {}

        face_of, ab = split(self.points, single_face, *axes(lateral, up))
        self._n = int(np.ceil(np.sqrt(np.bincount(face_of).max())))
        for f, name in enumerate(self.faces):
            idx = np.flatnonzero(face_of == f)
            if verbose:
                print(f"face {name}: {len(idx)} points on {self._n} x {self._n}")
            cells, uv = layout(self.points[idx], ab[idx], self._n, verbose)
            cells = fill_empty(cells)  # indices into idx; empty -> corner landmark
            self.indexes[name] = idx[cells]
            self.isomap[name] = uv[cells]