"""Core of flatpoints: the uv and the image of one face (a chart of the cut surface).

Two independent flattenings of the same face (see Method.md):

    Geodesic uv with shortcuts (`geodesic_shortcut_uv`):
        1. temporary naive grid: the points sorted on an n x n grid by their two
           coordinates other than the normal axis (gridpoints);
        2. approximate geodesic distances along that grid, from a few landmarks;
        3. landmark Isomap of these distances, oriented as the object (feet down).

    Conformal uv and image (`conformal_uv`, `image_cells`):
        1. rectangular conformal map of the mesh of the face onto the unit square;
        2. the image: the points sorted on an n x n grid by their conformal uv (gridpoints).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .rcm import boundary_loop, rectangular_conformal_map
from .utils import grid_sort, landmarks

# Grid graph used for geodesics: each cell is linked to the cells at most RADIUS rows and
# columns away.
RADIUS = 4
# Penalty on long edges (relative to the typical edge length), so that shortest paths
# follow the surface instead of jumping across gaps between non-adjacent parts.
EPS = 0.1
# Number of random points used to fit the orientation of (u, v).
ROTATION_SAMPLES = 50


# === Geodesic uv with shortcuts ===

def geodesic_shortcut_uv(P, ab, n, verbose=False):
    """(M, 2) geodesic uv of the M points of a face, in [0, 1]^2.

    The geodesics are measured on a naive grid of the points (sorted by `ab`, their two
    coordinates other than the normal axis), which links points that are close in that
    projection: the distances take some shortcuts across folds of the surface, which keeps the uv close
    to the side view of the object, at the price of some distortion (see Method.md).
    """
    pos = np.c_[np.divmod(np.arange(n * n), n)] / (n - 1)  # cell (row, col) in [0, 1]

    # 1. Naive grid, from the coordinates other than the normal axis
    slot = grid_sort(ab, n, verbose)  # cell -> point, -1 if empty
    real = slot >= 0
    cell = np.empty(len(P), int)  # point -> cell
    cell[slot[real]] = np.flatnonzero(real)
    grid = np.zeros((n * n, 3))
    grid[real] = P[slot[real]]

    # 2. Geodesic distances from the landmarks to every point, (L, M)
    marks = slot[landmarks(real, n)]
    D = geodesics(grid.reshape(n, n, 3), real, cell[marks])[:, cell]

    # 3. Flattening, oriented as the object
    return orient(isomap(D, marks, pos[cell]), ab)


def geodesics(grid, real, sources, radius=RADIUS, eps=EPS):
    """Approximate geodesic distances from `sources`, on a graph built from the grid.

    Each non-empty cell is linked to the non-empty cells at most `radius` rows and columns
    away, by an edge of 3D length x weighted x + eps * x^2 / h (h: typical edge length).
    The penalty makes the long edges, across gaps between non-adjacent parts of the
    surface, cost more: shortest paths and their lengths both use these weights.

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

    W = graph(x * (1 + eps * x / h))
    cost, pred = dijkstra(W, directed=False, indices=sources, return_predecessors=True)

    # Length of each path: each cell starts with the weight of the edge to its
    # predecessor; pointer doubling then adds up the edges back to the source in
    # log2(n*n) steps.
    X = W + W.T
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
    U, _, Vt = np.linalg.svd(A.T @ B)
    uv = uv @ (U @ Vt)
    return (uv - uv.min(0)) / np.maximum(np.ptp(uv, 0), 1e-12)


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
