"""Core of flatpoints: the image of one face (a chart of the cut surface).

    1. rectangular conformal map of the mesh of the face onto the unit square (`conformal_uv`);
    2. the image: the points sorted on an n x n grid by their conformal uv (`image_cells`).
"""
import numpy as np

from .rcm import boundary_loop, rectangular_conformal_map
from .utils import grid_sort


# === Conformal uv and image ===

def conformal_uv(V, F, ab):
    """(nv, 2) conformal map of the disk mesh (V, F) onto the unit square.

    Args:
        ab: (nv, 2) horizontal and up coordinates of the vertices: the corners of the
            square go to the boundary vertices closest to the corners of the silhouette.
    """
    loop = boundary_loop(F)
    x = (ab[loop] - ab[loop].mean(0)) / np.maximum(ab[loop].std(0), 1e-12)
    corner = [np.argmax(sa * x[:, 0] + sb * x[:, 1])  # position on the loop
              for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # the corners must follow each other along the loop, in one direction or the other
    turn = np.argsort((np.array(corner) - corner[0]) % len(loop))
    ordered = np.array_equal(turn, [0, 1, 2, 3]) or np.array_equal(turn, [0, 3, 2, 1])
    if not ordered or len(set(corner)) < 4:
        corner = corner[0] + np.arange(4) * len(loop) // 4  # odd silhouette: quarters
    return rectangular_conformal_map(V, F, loop[np.mod(corner, len(loop))])


def image_cells(uv, n, verbose=False):
    """(n, n) index of the point of each pixel (-1: empty), from the uv of the points.
    Image layout: u along the columns, v up the rows."""
    return grid_sort(uv, n, verbose).reshape(n, n).T[::-1]
