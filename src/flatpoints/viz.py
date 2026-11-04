"""Plots of a `Flat`: `plot` shows the surface in 3D and in (u, v), `imshow` the grids.

Two colourings: RGB = normalised x, y, z, and a checkerboard on the grid cells. A good
flattening gives smooth colours and regular, undistorted squares.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter, label, median_filter

CHECKER = np.array([[0.15, 0.25, 0.45], [0.98, 0.75, 0.45]])  # dark blue, light orange
JUMP, ALPHA_RATE = 3, 0.5  # opacity ALPHA when a side is JUMP times the median side around it
ALPHA_MAX = 0.5
TAU = (JUMP - 1) / np.sqrt(-np.log(ALPHA_RATE))  # width of the fade, see `_alpha`
LIGHT = 1.0  # strength of the background quads in (u, v): 0 = white, 1 = full colour


def plot(flat, squares=8):
    """First column: all the faces in 3D. Then one column per face in (u, v).
    First row coloured by RGB, second row by a checkerboard of `squares` x `squares`."""
    grids = _grids(flat)  # {face: (n, n) index of each pixel's point}
    st = max(1, flat.n // 110)  # downsampling for faster plot
    s, k = slice(None, None, st), len(grids)
    lo, hi = flat.points.min(0), flat.points.max(0)
    norm = (flat.points - lo) / np.where(hi > lo, hi - lo, 1)  # x, y, z in [0, 1]
    xyz = {f: flat.points[g[s, s]] for f, g in grids.items()}
    uv = {f: flat.uv[g[s, s]] for f, g in grids.items()}
    rgb = {f: norm[g[s, s]] for f, g in grids.items()}
    m = len(next(iter(uv.values())))  # side of the drawn grids
    # Opacity on the full grid: a drawn quad takes the lowest of the quads it covers
    alpha = {f: np.minimum.reduce([a[i::st, j::st][:m - 1, :m - 1] for i in range(st)
                                   for j in range(st)])
             for f, a in ((f, _alpha(flat.uv[g])) for f, g in grids.items())}
    proj = _camera(flat.lateral, flat.up)
    fig = plt.figure(figsize=(4 + 3 * k, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 1 + k, width_ratios=[1.5] + [1] * k)
    size = (300 / m) ** 2  # marker area (pt^2): about half a grid step, the quads show between
    for row, name in enumerate(("RGB", "checkerboard")):
        col = {f: _checker(m, squares) if row else rgb[f] for f in grids}  # per cell
        ax = fig.add_subplot(grid[row, 0], title=f"xyz, {name}")
        _draw3d(ax, [(xyz[f] @ proj.T, col[f], alpha[f]) for f in grids])
        for c, f in enumerate(grids, start=1):
            ax = fig.add_subplot(grid[row, c], aspect="equal", xlim=(0, 1), ylim=(0, 1),
                                 xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel="u", ylabel="v",
                                 title=f"{f + ': ' if f else ''}uv, {name}")
            light = 1 - LIGHT * (1 - _quads(col[f]).mean(2))  # quads, washed towards white
            _add(ax, _quads(uv[f]).reshape(-1, 4, 2), _rgba(light, alpha[f]).reshape(-1, 4))
            ax.scatter(*uv[f].reshape(-1, 2).T, c=col[f].reshape(-1, 3), s=size,
                       linewidths=0, rasterized=True)
    return fig


def imshow(xyz, sigma=0.5):
    """An xyz grid (n, n, 3), e.g. `flat.to_image(points)[0]`, as an RGB image: normalised
    x, y, z, smoothed by a Gaussian of `sigma` pixels."""
    points = xyz.reshape(-1, 3)
    fig, ax = plt.subplots()
    lo, hi = points.min(0)[None, None], points.max(0)[None, None]
    ax.imshow(gaussian_filter((xyz - lo)/(hi - lo), (sigma, sigma, 0)), interpolation="nearest")
    ax.set_axis_off()
    return fig


def _grids(flat):
    """{face: (n, n) index of each pixel's point}, empty pixels filled from the corners;
    faces "left", "right", or "" if single_face."""
    idx = flat.to_image(np.arange(len(flat.points)))
    return {"": idx[0]} if flat.single_face else {"left": idx[0], "right": idx[1]}


def _checker(n, squares):
    """(n, n, 3) colours of the cells of an n x n grid: `squares` x `squares` squares."""
    i, j = np.indices((n, n)) // int(np.ceil(n / squares))
    return CHECKER[(i + j) % 2]


def _rgba(rgb, alpha):
    return np.concatenate([rgb, alpha[..., None]], -1)


def _quads(grid):
    """(n-1, n-1, 4, d) the quads between neighbouring cells of an (n, n, d) grid."""
    return np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2)


def _add(ax, polys, rgba):
    """Draw the polygons; edges in the face colour, except on see-through ones (no seams)."""
    ax.add_collection(PolyCollection(polys, facecolors=rgba, linewidths=0.3,
                                     edgecolors=np.where(rgba[:, 3:] < 1, 0, rgba)))


def _alpha(uv):
    """(n-1, n-1) opacity of the quads, from r = longest side of the quad / median side
    around it (11 x 11 cells): exp(-((r - 1) / TAU)^2) for r > 1, else 1. Flat near r = 1
    (ordinary irregularities stay opaque), ALPHA at r = JUMP, ~0 beyond. A jump is a seam
    of several sides: areas of fewer than 6 quads with r > JUMP (local defects) stay opaque."""
    d, r = [L / np.maximum(median_filter(L, 11), 1e-12) for L in
            (np.linalg.norm(np.diff(uv, axis=a), axis=-1) for a in (0, 1))]
    r = np.maximum.reduce([d[:, :-1], d[:, 1:], r[:-1], r[1:]])
    areas = label(r > JUMP)[0]
    small = (areas > 0) & (np.bincount(areas.ravel())[areas] < 6)
    return np.where(small, ALPHA_MAX, ALPHA_MAX*np.exp(-(np.maximum(r - 1, 0) / TAU) ** 2))


def _camera(lateral, up, elev=15, azim=30):
    """(3, 3) rotation from xyz to the camera axes (right, up, towards the viewer): a side
    view from the `lateral` axis, with the remaining axis to the right (as u in the 2D
    panels), turned by `azim` degrees around `up` and raised by `elev` degrees."""
    e = np.eye(3)
    l, u = "xyz".index(lateral), "xyz".index(up)
    side = e[l] * np.sign(np.cross(e[u], e[l]) @ e[3 - l - u])
    az, el = np.radians(azim), np.radians(elev)
    eye = np.cos(el) * (np.cos(az) * side + np.sin(az) * np.cross(e[u], side)) \
        + np.sin(el) * e[u]
    right = np.cross(e[u], eye) / np.linalg.norm(np.cross(e[u], eye))
    return np.stack([right, np.cross(eye, right), eye])


def _draw3d(ax, faces):
    """Draw the faces [(projected xyz (n, n, 3), cell colours (n, n, 3), `_alpha`)], shaded:
    the quads in light colours, and on each cell a small disc (an octagon facing the camera)
    in full colour. Quads and discs are sorted together back to front; a disc is moved
    one grid step towards the viewer, so it covers the quads of its own surface but not a
    surface in front of it."""
    light = np.array([-0.3, 0.5, 1]) / np.linalg.norm([-0.3, 0.5, 1])
    step = np.median(np.concatenate([np.linalg.norm(np.diff(g, axis=a), axis=-1).ravel()
                                     for g, _, _ in faces for a in (0, 1)]))
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    disc = 0.4 * step * np.stack([np.cos(t), np.sin(t)], -1)  # diameter: half a step
    polys, rgba, depth = [], [], []
    for g, col, a in faces:
        q = _quads(g)
        n = np.cross(q[..., 2, :] - q[..., 0, :], q[..., 3, :] - q[..., 1, :])
        shade = 0.55 + 0.45 * abs(n @ light) / np.maximum(np.linalg.norm(n, axis=-1), 1e-12)
        pale = 1 - LIGHT * (1 - _quads(col).mean(2))
        polys += list(q[..., :2].reshape(-1, 4, 2))
        rgba.append(_rgba(pale * shade[..., None], a).reshape(-1, 4))
        depth.append(q[..., 2].mean(-1).ravel())
        s = np.pad(shade, 1, mode="edge")  # shade of a cell: mean of the quads around it
        s = (s[:-1, :-1] + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:]) / 4
        polys += list(g[..., None, :2].reshape(-1, 1, 2) + disc)
        rgba.append(_rgba(col * s[..., None], np.ones(s.shape)).reshape(-1, 4))
        depth.append(g[..., 2].ravel() + step)
    order = np.argsort(np.concatenate(depth), kind="stable")  # far first
    _add(ax, [polys[i] for i in order], np.concatenate(rgba)[order])
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
