"""User-facing function of flatpoints: `flatten`.

`flatten` takes points sampled on a surface and assigns each of them to a cell of a
square grid, so that neighbouring cells hold points that are close *on the surface*.
Grid-based tools (convolutions, images, ...) can then be applied to the cloud.

A closed surface (e.g. an animal, a sphere) cannot be flattened in one piece: it is cut
in two faces, "left" and "right", by a plane perpendicular to a chosen `lateral` axis
(x, y or z), each face laid out on its own grid (see `core.layout`). An open surface
(e.g. a terrain) fits on a single grid (single_face=True).
"""
import numpy as np

from .core import layout
from .utils import axes, fill_empty, split


def flatten(point_cloud, verbose=False, single_face=False, 
            lateral="x", up="z"):
    """Arrange a point cloud on square grids, neighbours on the grid being neighbours on
    the surface.

    Args:
        point_cloud: (N, 3) array of xyz coordinates.
        verbose: print progress.
        single_face: False (default) for a closed surface: the cloud is cut in two faces of
            equal size, "left" and "right", each laid out on its own grid. True for an open
            surface (e.g. a terrain): all the points on one grid.
        lateral: "x", "y" or "z", the lateral axis, default x. The lateral axis will be
            flattened.
        up: "x", "y" or "z", the vertical axis, default z. To orient the (u, v) projection:
            v will follow the vertical axis.

    Returns:
        A dict with two entries, laid out on the same n x n grid:
            "indexes": (n, n) int array, the index of each cell's point in `point_cloud`.
            "isomap": (n, n, 2) float array, the flat coordinates (u, v) of each
                cell's point, in [0, 1] (smoothed if smoothing=True).
        With single_face=False, each entry is itself a dict {"left": ..., "right": ...},
        one grid per lateral face.

        The grids read like images: u increases from left to right, v from bottom to top
        (row 0 is the top).

    Example:
        out = flatten(points)
        idx = out["indexes"]["left"]          # (n, n)
        uv = out["isomap"]["left"]            # (n, n, 2)
        xyz = points[idx]                     # (n, n, 3)
    """
    l, u = axes(lateral, up)
    P = np.asarray(point_cloud, float)
    face, ab = split(P, single_face, l, u)
    n = int(np.ceil(np.sqrt(np.bincount(face).max())))
    n_faces = 1 if single_face else 2

    indexes, coordinates = [], []
    for f in range(n_faces):
        idx = np.flatnonzero(face == f)
        if verbose:
            print(f"face {f + 1}/{n_faces}: {len(idx)} points on {n} x {n}")
        cells, uv = layout(P[idx], ab[idx], n, verbose)
        cells = fill_empty(cells)  # cells: indices into idx; empty -> corner landmark
        indexes.append(idx[cells])
        coordinates.append(uv[cells])

    if single_face:
        return {"indexes": indexes[0], "isomap": coordinates[0]}
    return {"indexes": {"left": indexes[0], "right": indexes[1]},
            "isomap": {"left": coordinates[0], "right": coordinates[1]}}
