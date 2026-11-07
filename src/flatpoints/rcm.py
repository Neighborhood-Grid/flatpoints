"""Rectangular conformal map of a disk-like triangle mesh.

Python port of the MATLAB code https://github.com/garyptchoi/rectangular-conformal-map,
Copyright (c) 2015-2024 Gary Pui-Tung Choi, licensed under the Apache License 2.0 (see
LICENSE-rectangular-conformal-map). Modified from the original: translated to
Python/NumPy/SciPy, vectorised, Tutte fallback also used when the harmonic map folds, and
the optimal height (step 3) left out, since flatpoints rescales the rectangle to the unit
square anyway.

Reference:
    T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal
    Mappings of Point Clouds", SIAM Journal on Imaging Sciences, 9(4), 1922-1962, 2016.

Steps:
    1. Disk: harmonic map (cotangent Laplacian) of the mesh onto the unit disk, the
       boundary placed by arc length.
    2. Square: generalised Laplace equation built from the Beltrami coefficient of the
       disk map, with x = 0 / 1 on the left / right sides and y = 0 / 1 on the bottom / top
       sides, the sides being the boundary arcs between the four corners.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized


def rectangular_conformal_map(v, f, corners):
    """(nv, 2) conformal map of the disk mesh (v, f) onto the unit square.

    Args:
        v: (nv, 3) vertices.
        f: (nf, 3) triangles of a mesh with the topology of a disk (one boundary loop, no
            boundary vertex in a single triangle).
        corners: 4 boundary vertices sent to (0, 0), (1, 0), (1, 1), (0, 1).
    """
    loop = boundary_loop(f)
    where = np.full(len(v), -1)
    where[loop] = np.arange(len(loop))
    if (where[corners] < 0).any():
        raise ValueError("the corners must lie on the boundary")
    loop = np.roll(loop, -where[corners[0]])
    where[loop] = np.arange(len(loop))
    i1, i2, i3 = where[corners[1:]]
    if not i1 < i2 < i3:  # the corners turn the other way round: reverse the loop
        loop = np.r_[loop[0], loop[:0:-1]]
        where[loop] = np.arange(len(loop))
        i1, i2, i3 = where[corners[1:]]

    disk = disk_map(v, f, loop)

    # Square: bottom = loop[0..i1], right = loop[i1..i2], top = loop[i2..i3], left = rest
    A = generalized_laplacian(disk, f, beltrami_coefficient(disk, f, v))
    left_right = np.r_[loop[i3:], loop[0], loop[i1:i2 + 1]]
    bottom_top = np.r_[loop[:i1 + 1], loop[i2:i3 + 1]]
    x = _dirichlet(A, left_right, loop[i1:i2 + 1], len(v))()
    y = _dirichlet(A, bottom_top, loop[i2:i3 + 1], len(v))()
    return np.c_[x, y]


def disk_map(v, f, loop):
    """(nv, 2) harmonic map onto the unit disk, the boundary `loop` placed by arc length.
    Falls back to the Tutte map (always bijective) if the harmonic map folds."""
    arc = np.linalg.norm(v[loop] - v[np.roll(loop, -1)], axis=1)
    theta = 2 * np.pi * np.r_[0, np.cumsum(arc)[:-1]] / arc.sum()
    target = np.c_[np.cos(theta), np.sin(theta)]
    for laplacian in (cotangent_laplacian(v, f), tutte_laplacian(len(v), f)):
        disk = _dirichlet(laplacian, loop, None, len(v))(target)
        if np.isfinite(disk).all() and _orientation_consistent(disk, f):
            break
    return disk


def boundary_loop(f):
    """Vertices of the boundary of a disk mesh, in order. Raises ValueError if the boundary
    is not a single simple loop."""
    he = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]].astype(np.int64)
    n = he.max() + 1
    bd = he[~np.isin(he[:, 0] * n + he[:, 1], he[:, 1] * n + he[:, 0])]
    if len(bd) == 0:
        raise ValueError("the mesh has no boundary")
    nxt = np.full(n, -1)
    nxt[bd[:, 0]] = bd[:, 1]
    if len(np.unique(bd[:, 0])) != len(bd):
        raise ValueError("pinched boundary (a vertex is met twice)")
    loop = [bd[0, 0]]
    for _ in range(len(bd) - 1):
        loop.append(nxt[loop[-1]])
    if nxt[loop[-1]] != loop[0] or len(set(loop)) != len(bd):
        raise ValueError("the boundary has several loops")
    return np.array(loop)


def cotangent_laplacian(v, f):
    """(nv, nv) cotangent Laplacian (negative semi-definite)."""
    n = len(v)
    L = 0
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        # cotangent of the angle at c, weight of the edge (a, b)
        e1, e2 = v[f[:, a]] - v[f[:, c]], v[f[:, b]] - v[f[:, c]]
        cot = (e1 * e2).sum(1) / np.maximum(np.linalg.norm(np.cross(e1, e2), axis=1), 1e-30)
        L = L + coo_matrix((cot / 2, (f[:, a], f[:, b])), shape=(n, n))
    L = L + L.T
    return (L - diags(np.asarray(L.sum(1)).ravel())).tocsr()


def tutte_laplacian(n, f):
    """(n, n) graph Laplacian of the mesh edges, uniform weights."""
    e = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    W = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    W = ((W + W.T) > 0).astype(float)
    return W - diags(np.asarray(W.sum(1)).ravel())


def beltrami_coefficient(domain, f, image):
    """(nf,) complex Beltrami coefficient of the map domain (2D) -> image (2D or 3D),
    piecewise linear on the triangles."""
    p = domain[f]  # (nf, 3, 2)
    e = p[:, [2, 0, 1]] - p[:, [1, 2, 0]]  # edge opposite to each vertex
    area2 = e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]
    # gradient of the hat function of each vertex: rot90(opposite edge) / (2 area)
    gx, gy = e[..., 1] / area2[:, None], -e[..., 0] / area2[:, None]  # (nf, 3)
    q = image[f]  # (nf, 3, d)
    du, dv = np.einsum("fk,fkd->fd", gx, q), np.einsum("fk,fkd->fd", gy, q)
    E, G, F = (du * du).sum(1), (dv * dv).sum(1), (du * dv).sum(1)
    return (E - G + 2j * F) / (E + G + 2 * np.sqrt(np.maximum(E * G - F ** 2, 0)))


def generalized_laplacian(v, f, mu):
    """(nv, nv) Laplacian of the metric of Beltrami coefficient `mu` on the 2D mesh v."""
    mu = np.where(np.abs(mu) < 0.999, mu, 0.999 * mu / np.abs(mu))  # keep it invertible
    m2 = np.abs(mu) ** 2
    a = (1 - 2 * mu.real + m2) / (1 - m2)
    b = -2 * mu.imag / (1 - m2)
    g = (1 + 2 * mu.real + m2) / (1 - m2)
    p = v[f]
    rot = p[:, [2, 0, 1]] - p[:, [1, 2, 0]]  # opposite edges, (nf, 3, 2)
    ux, uy = rot[..., 1], -rot[..., 0]  # rotated by 90 degrees
    area2 = np.abs(rot[:, 0, 0] * rot[:, 1, 1] - rot[:, 0, 1] * rot[:, 1, 0])
    n = len(v)
    A = 0
    for i in range(3):
        for j in range(3):
            w = (a * ux[:, i] * ux[:, j] + b * (ux[:, i] * uy[:, j] + ux[:, j] * uy[:, i])
                 + g * uy[:, i] * uy[:, j]) / area2
            A = A + coo_matrix((-w, (f[:, i], f[:, j])), shape=(n, n))
    return A.tocsr()


def _dirichlet(M, fixed, ones, n):
    """Solver of M x = b with x fixed on the vertices `fixed`. With `ones`, the returned
    function takes no argument and solves for x = 1 on `ones`, 0 on the rest of `fixed`;
    otherwise it takes the values on `fixed` (k, d) and returns x (n, d)."""
    free = np.ones(n)
    free[fixed] = 0
    solve = factorized((diags(free) @ M + diags(1 - free)).tocsc())
    if ones is not None:
        b = np.zeros(n)
        b[ones] = 1
        return lambda: solve(b)

    def run(values):
        b = np.zeros((n, values.shape[1]))
        b[fixed] = values
        return np.stack([solve(c) for c in b.T], 1)
    return run


def _orientation_consistent(uv, f):
    """True if no triangle of the 2D map is flipped or degenerate."""
    d1, d2 = uv[f[:, 1]] - uv[f[:, 0]], uv[f[:, 2]] - uv[f[:, 0]]
    s = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return (s > 0).all() or (s < 0).all()
