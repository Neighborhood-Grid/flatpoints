"""Core of flatpoints: lay out the points of one face on a square grid.

Pipeline (`layout`):
    1. Naive grid: sort the points on the grid by their two non-lateral coordinates (e.g.
       y, z). Fast, but distorted where the surface is steep.
    2. Geodesics: approximate distances along the surface, from a few landmark points,
       using a graph that links nearby cells of the naive grid.
    3. Flattening: compute 2D coordinates (u, v) that best preserve these distances
       (landmark Isomap).
    4. Orientation: rotate (u, v) so that v follows the `up` axis and u the remaining one,
       so that the result keeps the orientation of the object (e.g. feet down).
    5. Final grid: sort the points on the grid again, this time by (u, v).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .utils import grid_sort, landmarks, orient

# Grid graph used for geodesics: each cell is linked to the cells at most RADIUS rows and
# columns away.
RADIUS = 4
# Penalty on long edges (relative to the typical edge length), so that shortest paths
# follow the surface instead of jumping across gaps between non-adjacent parts.
EPS = 0.1


def layout(P, ab, n, verbose=False, rotation=True):
    """Lay out M <= n*n points on an n x n grid (steps 1-6 above).

    Args:
        P: (M, 3) xyz.
        ab: (M, 2) the two non-lateral coordinates.

    Returns:
        cells: (n, n) index of each cell's point in P, -1 for empty cells; image layout
            (u along columns, v up the rows).
        uv: (M, 2) flat coordinates of each point, in [0, 1].
    """
    # (row, col) of each cell, scaled to [0, 1]; cells are numbered row-major
    pos = np.c_[np.divmod(np.arange(n * n), n)] / (n - 1)

    # 1. Naive grid, from the non-lateral coordinates
    slot = grid_sort(ab, n, verbose)  # cell -> point, -1 if empty
    real = slot >= 0
    cell = np.empty(len(P), int)  # point -> cell (inverse of slot)
    cell[slot[real]] = np.flatnonzero(real)
    grid = np.zeros((n * n, 3))
    grid[real] = P[slot[real]]

    # 2. Geodesic distances from the landmarks to every point, (L, M)
    marks = slot[landmarks(real, n)]
    D = geodesics(grid.reshape(n, n, 3), real, cell[marks])[:, cell]

    # 3. Flattening, 4. Orientation
    uv = isomap(D, marks, pos[cell])
    if rotation:
        uv = orient(uv, ab)

    # 5. Final grid, u along the rows and v along the columns; transposed and flipped to
    # the image layout (u along the columns, v up the rows)
    return grid_sort(uv, n, verbose).reshape(n, n).T[::-1], uv


def geodesics(grid, real, sources, radius=RADIUS, eps=EPS):
    """Approximate geodesic distances from `sources`, on a graph built from the grid.

    Each non-empty cell is linked to the non-empty cells at most `radius` rows and columns
    away, with an edge weighted by the 3D distance between their points. Shortest paths
    are chosen with a cost that penalises long edges (see `EPS`), but their length is the
    sum of the true 3D distances.

    Args:
        grid: (n, n, 3) xyz of the point in each cell.
        real: (n*n,) True for cells holding a point.
        sources: (L,) cell indices to measure distances from.

    Returns:
        (L, n*n) distances from each source to each cell; inf for empty or unreachable cells.
    """
    n = len(grid)
    ids = np.arange(n * n).reshape(n, n)

    # Edges (i, j): every pair of cells offset by (di, dj), each pair counted once
    i, j = [], []
    for di in range(radius + 1):
        for dj in range(-radius, radius + 1):
            if di > 0 or dj > 0:
                i.append(ids[: n - di, max(0, -dj) : n - max(0, dj)].ravel())
                j.append(ids[di:, max(0, dj) : n - max(0, -dj)].ravel())
    i, j = np.concatenate(i), np.concatenate(j)
    keep = real[i] & real[j]  # edges between real cells
    i, j = i[keep], j[keep]

    # 3D length of each edge (kept > 0: scipy treats 0 as "no edge")
    xyz = grid.reshape(-1, 3)
    x = np.maximum(np.linalg.norm(xyz[i] - xyz[j], axis=1), 1e-12)
    h = np.median(x[(j - i == 1) | (j - i == n)])  # typical length, from direct neighbours
    graph = lambda w: coo_matrix((w, (i, j)), shape=(n * n, n * n)).tocsr()  # noqa: E731

    # Shortest paths for the penalised cost x + eps * x^2 / h (long edges cost more)...
    cost, pred = dijkstra(graph(x*(1 + eps * x / h)), directed=False, indices=sources,
                          return_predecessors=True)

    # ...but measured with their true length, the sum of x along each path. Each cell
    # starts with the length of the edge to its predecessor; pointer doubling then adds
    # up the edges back to the source in log2(n*n) steps.
    X = graph(x*(1 + eps * x / h))
    X = X + X.T
    rows = np.arange(len(sources))[:, None]
    up = np.where(pred >= 0, pred, np.arange(n * n))  # source/unreached: points to itself
    D = np.asarray(X[up.ravel(), np.tile(np.arange(n * n), len(sources))]).reshape(up.shape)
    for _ in range(int(np.log2(n * n)) + 1):
        D, up = D + D[rows, up], up[rows, up]
    D[~np.isfinite(cost)] = np.inf
    return D


def isomap(D, marks, pos):
    """2D coordinates (u, v) that best preserve geodesic distances (landmark Isomap).

    The landmarks are placed in 2D by classical MDS on their pairwise distances; every
    other point is then placed from its distances to the landmarks (triangulation). The
    result is rotated to match the naive grid layout `pos` and scaled to [0, 1].

    Args:
        D: (L, M) geodesic distances from L landmarks to M points.
        marks: (L,) indices of the landmarks among the M points.
        pos: (M, 2) position of each point in the naive grid, in [0, 1].

    Returns:
        (M, 2) (u, v) of each point, in [0, 1]. Points not reached by every landmark keep
        their `pos`.
    """
    ok = np.isfinite(D).all(0)  # points reached from every landmark
    D2 = np.where(ok, D, 0) ** 2

    # Classical MDS of the landmarks: top-2 eigenvectors of the double-centred distances
    L2 = (D2[:, marks] + D2[:, marks].T) / 2  # symmetrised
    J = np.eye(len(marks)) - 1 / len(marks)
    val, vec = np.linalg.eigh(-J @ L2 @ J / 2)
    # Triangulation of all the points from their distances to the landmarks
    Y = -(D2 - L2.mean(1, keepdims=True)).T @ (vec[:, -2:] / np.sqrt(np.maximum(val[-2:], 1e-12))) / 2

    # Rotation (or reflection) best aligning Y with `pos` (orthogonal Procrustes)
    Y -= Y[ok].mean(0)
    U, _, Vt = np.linalg.svd(Y[ok].T @ (pos[ok] - pos[ok].mean(0)))
    Y = Y @ U @ Vt
    Y = (Y - Y[ok].min(0)) / np.ptp(Y[ok], 0)  # each axis scaled to [0, 1]
    Y[~ok] = pos[~ok]
    return Y
