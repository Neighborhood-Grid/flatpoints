"""`Resampler(..., method="ot")`: the seed, then Lloyd iterations of optimal transport.

Each new point receives N / M input points (exact masses), by optimal transport between
the points, with the true cost ||x - y||^2 in 3D, then moves to the barycentre of its input
points. No mesh is used (it is only a reconstruction, it can miss thin parts): at the end,
each barycentre (slightly inside the surface where it curves) is projected on the tangent
plane of its nearest input point, fitted on the input points around it.

The images make the transport sparse: an input point may only go to its K nearest new
points on the images (N K pairs, instead of N M), and no farther than REACH times the
spacing of the new points in 3D (across a tear, a candidate can be near on the image and
far on the surface).

The transport is solved by its dual (a power diagram): input point j goes to the
candidate i minimising ||x_j - y_i||^2 - w_i, and the weight w_i rises if cell i has too
few points, falls if it has too many. numpy only; the weights are kept from one Lloyd
iteration to the next, so a few dual steps suffice. Cells hold N / M points up to a few.
"""
import numpy as np
from scipy.spatial import cKDTree

K = 8            # candidate new points of an input point, nearest on the images
REACH = 2.0      # ... and closer than REACH x the spacing of the new points in 3D
DUAL_STEPS = 30  # dual steps per Lloyd iteration
DUAL_RATE = 1.2  # dual step, x the typical cost / (N / M)
K_PLANE = 10     # input points fitting a tangent plane


def lloyd_ot(flat, points, pixels, n_iter=10):
    """`n_iter` Lloyd iterations of sparse optimal transport from the new `points` (M, 3),
    at `pixels` (M, 3) float (face, row, col) on the images of `flat`. Faces do not change.
    Returns (points (M, 3), pixels (M, 3) float (face, row, col))."""
    P, ip = flat.points, flat.pixels
    points = np.asarray(points, float).copy()
    pixels = np.asarray(pixels, float).copy()
    face = pixels[:, 0].astype(int)
    weights = np.zeros(len(points))
    for _ in range(n_iter):
        reach2 = REACH ** 2 * np.median(cKDTree(points).query(points, k=2)[0][:, 1]) ** 2
        for f in range(flat.n_faces):
            J, I = np.flatnonzero(ip[:, 0] == f), np.flatnonzero(face == f)
            if len(I) == 0:
                continue
            k = min(K, len(I))
            _, nb = cKDTree(pixels[I, 1:]).query(ip[J, 1:].astype(float), k=k)
            nb = nb.reshape(len(J), k)
            cost = ((P[J, None, :] - points[I][nb]) ** 2).sum(-1)
            far = cost > reach2
            far[np.arange(len(J)), cost.argmin(1)] = False  # always one candidate
            cost[far] = np.inf

            # transport: dual ascent on the weights of the cells
            share = len(J) / len(I)
            rate = DUAL_RATE * np.median(cost.min(1)) / share
            w = weights[I]
            for _ in range(DUAL_STEPS):
                owner = nb[np.arange(len(J)), np.argmin(cost - w[nb], 1)]
                count = np.bincount(owner, minlength=len(I))
                w += rate * (share - count)
            weights[I] = w

            # Lloyd: barycentre of each cell (an empty cell keeps its point)
            full = np.flatnonzero(count)
            for x, out, cols in ((P[J], points, slice(None)),
                                 (ip[J, 1:].astype(float), pixels, slice(1, 3))):
                sums = np.stack([np.bincount(owner, c, len(I)) for c in x.T], 1)
                out[I[full], cols] = sums[full] / count[full, None]
    return tangent_projection(P, points), pixels


def tangent_projection(P, Q):
    """`Q` (M, 3) projected on the tangent plane of their nearest point of `P`, fitted
    (PCA) on its K_PLANE nearest points."""
    tree = cKDTree(P)
    _, i = tree.query(Q)
    _, around = tree.query(P[i], k=min(K_PLANE, len(P)))
    X = P[around] - P[around].mean(1, keepdims=True)
    normal = np.linalg.eigh(np.einsum("mki,mkj->mij", X, X))[1][:, :, 0]
    return Q - ((Q - P[i]) * normal).sum(1, keepdims=True) * normal
