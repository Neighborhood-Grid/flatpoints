"""Helpers of flatpoints: axes, faces, grid sorting, landmarks, orientation, smoothing.

Everything here is plumbing around the algorithm of `core`: no geodesics, no flattening.
"""
import gridpoints
import numpy as np

# Landmark positions (row, col), as fractions of the grid side: 4 corners, 4 edge middles,
# centre, and the centres of the 4 quarter squares (13 in total).
LANDMARKS = [(a, b) for a in (0, 0.5, 1) for b in (0, 0.5, 1)] + [
    (a, b) for a in (0.25, 0.75) for b in (0.25, 0.75)
]

# Number of random points used to fit the orientation of (u, v).
ROTATION_SAMPLES = 50

# Number of 3D nearest neighbours used to smooth (u, v) (smoothing=True).
KNN = 16


def axes(lateral, up):
    """Indices (0, 1 or 2) of the lateral and up axes; up=None picks the default (the
    last non-lateral axis in xyz order)."""
    if lateral not in ("x", "y", "z"):
        raise ValueError('lateral must be "x", "y" or "z"')
    l = "xyz".index(lateral)
    if up is None:
        return l, 1 if l == 2 else 2
    if up not in ("x", "y", "z") or up == lateral:
        raise ValueError('up must be "x", "y" or "z", different from lateral')
    return l, "xyz".index(up)


def split(P, single_face, lateral, up):
    """Assign each point to a face, and keep its two non-lateral coordinates.

    Args:
        lateral, up: indices (0, 1 or 2) of the lateral and up axes.

    Returns:
        face: (N,) face of each point: 0 (left, lateral coordinate <= median) or 1
            (right); all 0 if single_face.
        ab: (N, 2) the two other coordinates: the horizontal axis (the one that is
            neither lateral nor up), then the up axis. u will follow the first, v the
            second.
    """
    ab = P[:, [3 - lateral - up, up]]
    if single_face:
        return np.zeros(len(P), int), ab
    t = P[:, lateral]
    return (t > np.median(t)).astype(int), ab


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
    """(row, col) of the landmark at fractions (a, b) of an n x n grid: if its cell is
    empty, walk from it (t = 0) towards the centre (t = 1) until a non-empty cell.

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
    """Final grid with its empty cells filled from the corner landmarks.

    The grid is cut into four quadrants. Each empty cell takes the point of the landmark
    of its quadrant's corner: the first non-empty cell met when walking from that corner
    towards the centre (as in `landmarks`).

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


def orient(uv, ab, n_samples=ROTATION_SAMPLES):
    """Rotate (u, v) so that u follows ab[:, 0] and v follows ab[:, 1] as closely as
    possible (orthogonal Procrustes, fitted on a few random points to keep it fast).

    A reflection is allowed, to undo a mirrored flattening.

    Returns:
        (M, 2) the rotated (u, v), each axis scaled to [0, 1].
    """
    rng = np.random.default_rng(0)  # fixed seed: same result on every run
    s = rng.choice(len(uv), min(n_samples, len(uv)), replace=False)
    A = uv[s] - uv[s].mean(0)
    B = ab[s] - ab[s].mean(0)
    B /= np.maximum(B.std(0), 1e-12)  # both target axes weigh the same
    # Orthogonal R maximising the correlation between A @ R and B
    U, _, Vt = np.linalg.svd(A.T @ B)
    uv = uv @ (U @ Vt)
    return _unit(uv)


def _unit(X):
    """Each column of X scaled to [0, 1]."""
    return (X - X.min(0)) / np.maximum(np.ptp(X, 0), 1e-12)
