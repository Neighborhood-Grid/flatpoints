"""Plots of a `Flat`: `plot` shows the surface in 3D and in (u, v), `imshow` the grids.

Two colourings: RGB = normalised x, y, z, and a checkerboard on the grid cells. A good
flattening gives smooth colours and regular, undistorted squares.

The surface is drawn as the quads between neighbouring cells; the 3D view is projected and
sorted back to front (painter's algorithm), with a simple shading. A quad fades out the
more likely the grid jumps there: the longer its longest side in (u, v) compared with the
sides around it. Opaque for regular sides, ALPHA at JUMP times longer, ~0 beyond.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter, label, median_filter

CHECKER = np.array([[0.15, 0.25, 0.45], [0.98, 0.75, 0.45]])  # dark blue, light orange
JUMP, ALPHA = 5, 0.5  # opacity ALPHA when a side is JUMP times the median side around it
TAU = (JUMP - 1) / np.sqrt(-np.log(ALPHA))  # width of the fade, see `_alpha`


def plot(flat, squares=8):
    """First column: all the faces in 3D. Then one column per face in (u, v).
    First row coloured by RGB, second row by a checkerboard of `squares` x `squares`."""
    st = max(1, flat._n // 100)  # downsampling for faster plot
    s, k = slice(None, None, st), len(flat.faces)
    xyz = {f: flat.points[flat.indexes[f][s, s]] for f in flat.faces}
    uv = {f: flat.isomap[f][s, s] for f in flat.faces}
    rgb = {f: c[s, s] for f, c in _rgb(flat).items()}
    m = len(next(iter(uv.values())))  # side of the drawn grids
    # Opacity on the full grid: a drawn quad takes the lowest of the quads it covers
    alpha = {f: np.minimum.reduce([a[i::st, j::st][:m - 1, :m - 1] for i in range(st)
                                   for j in range(st)])
             for f, a in ((f, _alpha(flat.isomap[f])) for f in flat.faces)}
    proj = _camera(flat._lateral, flat._up)
    fig = plt.figure(figsize=(4 + 3 * k, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 1 + k, width_ratios=[1.5] + [1] * k)
    for row, name in enumerate(("RGB", "checkerboard")):
        col = {f: _checker(m, squares)[:-1, :-1] if row else _quads(rgb[f]).mean(2)
               for f in flat.faces}
        ax = fig.add_subplot(grid[row, 0], title=f"xyz, {name}")
        _draw3d(ax, [(xyz[f] @ proj.T, col[f], alpha[f]) for f in flat.faces])
        for c, f in enumerate(flat.faces, start=1):
            ax = fig.add_subplot(grid[row, c], aspect="equal", xlim=(0, 1), ylim=(0, 1),
                                 xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel="u", ylabel="v",
                                 title=f"{f}: uv, {name}")
            _add(ax, _quads(uv[f]).reshape(-1, 4, 2), _rgba(col[f], alpha[f]).reshape(-1, 4))
    return fig


def imshow(flat, colors="rgb", squares=4, sigma=2):
    """The grids as images, one per face: colors "rgb" (normalised x, y, z, smoothed by a
    Gaussian of `sigma` cells) or "checkerboard"."""
    fig, axes = plt.subplots(1, len(flat.faces), figsize=(4 * len(flat.faces), 4.4),
                             squeeze=False, layout="constrained")
    for ax, (f, im) in zip(axes[0], _rgb(flat).items()):
        ax.imshow(gaussian_filter(im, (sigma, sigma, 0)) if colors == "rgb"
                  else _checker(flat._n, squares), interpolation="nearest")
        ax.set_title(f"{f}: grid, {colors}")
        ax.set_axis_off()
    return fig


def _rgb(flat):
    """{face: (n, n, 3)} normalised x, y, z, on a common scale for all the faces."""
    lo, hi = flat.points.min(0), flat.points.max(0)
    return {f: (flat.points[flat.indexes[f]] - lo) / np.where(hi > lo, hi - lo, 1)
            for f in flat.faces}


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
    return np.where(small, 1.0, np.exp(-(np.maximum(r - 1, 0) / TAU) ** 2))


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
    """Draw the faces [(projected xyz (n, n, 3), quad colours, `_alpha`)] back to front,
    shaded."""
    light = np.array([-0.3, 0.5, 1]) / np.linalg.norm([-0.3, 0.5, 1])
    q = np.concatenate([_quads(g).reshape(-1, 4, 3) for g, _, _ in faces])
    c = np.concatenate([_rgba(c, a).reshape(-1, 4) for _, c, a in faces])
    n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    c[:, :3] *= (0.55 + 0.45 * abs(n @ light) / np.maximum(np.linalg.norm(n, axis=1), 1e-12))[:, None]
    order = np.argsort(q[:, :, 2].mean(1))  # far first
    _add(ax, q[order, :, :2], c[order])
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()