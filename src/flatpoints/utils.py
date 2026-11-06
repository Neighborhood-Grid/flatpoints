"""Helpers of flatpoints: axes, grid sorting, landmarks, padding, containers, resampling seeds.

Everything here is plumbing around `core` and `mesh`: no geometry.
"""
import gridpoints
import numpy as np
from typing import NamedTuple
from pathlib import Path

# Landmark positions (row, col), as fractions of the grid side: 4 corners, 4 edge middles,
# centre, and the centres of the 4 quarter squares (13 in total).
LANDMARKS = [(a, b) for a in (0, 0.5, 1) for b in (0, 0.5, 1)] + [
    (a, b) for a in (0.25, 0.75) for b in (0.25, 0.75)
]


# === Grids ====

def axes(normal_axis):
    """Indices (0, 1 or 2) of the normal, horizontal and up axes. Up (followed by v, the
    rows of the images) is z, or y if the normal axis is z; horizontal is the third one."""
    if normal_axis not in ("x", "y", "z"):
        raise ValueError('normal_axis must be "x", "y" or "z"')
    normal = "xyz".index(normal_axis)
    up = 1 if normal == 2 else 2
    return normal, 3 - normal - up, up


def grid_sort(X, n, verbose=False):
    """Assign N <= n*n 2D points to the cells of an n x n grid, with `gridpoints`.

    Returns:
        (n*n,) for each cell (row-major order), the index of its point in X, or -1 if empty.
    """
    N = len(X)
    X = gridpoints.pad(np.c_[X], n * n, value="inf")
    order = np.asarray(gridpoints.argsort(X, (n, n), verbose=int(verbose), n_iter = 100))
    return np.where(order < N, order, -1)


def landmarks(real, n):
    """Cells of the LANDMARKS on an n x n grid.

    Args:
        real: (n*n,) True for cells holding a point.

    Returns:
        Sorted unique cell indices (row-major). A landmark falling on an empty cell is moved
        towards the grid centre until it reaches a non-empty one.
    """
    real = real.reshape(n, n)
    cells = [i * n + j for i, j in (walk_to_real(real, a, b) for a, b in LANDMARKS)]
    return np.unique(cells)


def walk_to_real(real, a, b):
    """(row, col) of the cell at fractions (a, b) of an n x n grid: if it is empty, walk
    from it (t = 0) towards the centre (t = 1) until a non-empty cell.

    Args:
        real: (n, n) True for cells holding a point.
    """
    n = len(real)
    m = n - 1
    for t in np.linspace(0, 1, n):
        i, j = (int(round(x * m + t * (m / 2 - x * m))) for x in (a, b))
        if real[i, j]:
            break
    return i, j


def fill_empty(cells):
    """Final grid with its empty cells filled from the corners.

    The grid is cut into four quadrants. Each empty cell takes the point of its quadrant's
    corner: the first non-empty cell met when walking from that corner towards the centre
    (see `walk_to_real`).

    Args:
        cells: (n, n) final grid (image layout), index of each cell's point, -1 for empty
            cells.

    Returns:
        (n, n) the same grid, with no -1 left.
    """
    n = len(cells)
    real = cells >= 0
    out = cells.copy()
    i, j = np.nonzero(~real)
    # quadrant of each empty cell: 0 = top / left half, 1 = bottom / right half
    qi, qj = (2 * i > n - 1).astype(int), (2 * j > n - 1).astype(int)
    for a in (0, 1):
        for b in (0, 1):
            sel = (qi == a) & (qj == b)
            if sel.any():
                out[i[sel], j[sel]] = cells[walk_to_real(real, a, b)]
    return out


class Mesh(NamedTuple):
    """Triangle mesh of the surface, cut into faces."""
    vertices: np.ndarray   # (V, 3) xyz
    triangles: np.ndarray  # (T, 3) vertex indices
    face: np.ndarray       # (T,) face of each triangle: 0, 1, or -1 if in no face


class Chart(NamedTuple):
    """One face of the mesh, a topological disk, with the conformal uv of its vertices."""
    vertices: np.ndarray      # (Vf,) indices in Mesh.vertices
    triangles: np.ndarray     # (Tf, 3) indices in Chart.vertices
    uv_conformal: np.ndarray  # (Vf, 2) rectangular conformal map onto [0, 1]^2


class Corners(NamedTuple):
    """Padding of the grids: copy points[idxs[k]] into pixel pixels[k]."""
    idxs: np.ndarray    # (Npadding,) index in the point cloud of the point to copy
    pixels: np.ndarray  # (Npadding, 3) (face, row, col) where to copy it

# === Resampling ====

SEED = Path(__file__).resolve().parents[2] / "data" / "seed.npy"  # (2**17, 2) hyperuniform

# A new point is on the surface if it is closer to the input cloud than ON_SURFACE times
# the local spacing of the cloud (mean distance of the nearest input point to its 6
# neighbours). Bilinear interpolation across a tear of the images (two pixels side by side
# from far apart parts of the surface) gives points in the air, far from the cloud.
ON_SURFACE = 1.5


def _window(seed, n, m, keep):
    """The `n` points of `seed` closest to the origin (square windows [0, side)^2) that
    `keep`, as (row, col) on an m x m grid: [0, side)^2 is rescaled to [0, m - 1]^2.

    Args:
        keep: (k, 2) row, col -> (k,) True for the points to keep.

    Returns:
        (n, 2) row, col.
    """
    order = np.argsort(seed.max(1))
    x, d = seed[order], seed.max(1)[order]
    k = n
    while True:
        if k > len(x):
            raise ValueError(f"the seed is too small for {n} points on this surface")
        side = (d[k - 1] + d[k]) / 2 if k < len(x) else 1.0
        rc = x[:k] / side * (m - 1)
        ok = keep(rc)
        if ok.sum() >= n:
            return rc[ok][:n]
        k += n - ok.sum()


def _seeds(seed, names):
    """{face name: (M, 2) points in [0, 1)^2} from the `seed` argument of `Resampler`."""
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        shifts = np.random.default_rng(seed).random((len(names), 2))
        hyperuniform = np.load(SEED).astype(float)
        return {name: (hyperuniform + s) % 1.0 for name, s in zip(names, shifts)}
    if isinstance(seed, dict):
        if set(seed) != set(names):
            raise ValueError(f"seed: expected the keys {list(names)}, got {list(seed)}")
        return {name: _unit_square(seed[name]) for name in names}
    return {name: _unit_square(seed) for name in names}

def _unit_square(cloud):
    """`cloud` as an (M, 2) float array, checked to lie in [0, 1)^2."""
    try:
        cloud = np.asarray(cloud, float)
    except (TypeError, ValueError):
        cloud = None
    if cloud is None or cloud.ndim != 2 or cloud.shape[1] != 2 or not (
            (cloud >= 0).all() and (cloud < 1).all()):
        raise ValueError("seed: an int, an (M, 2) array in [0, 1)^2, or a dict of them")
    return cloud
