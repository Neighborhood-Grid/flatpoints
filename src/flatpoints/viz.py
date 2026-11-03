"""Plots of a `Flat`, drawn in 2D: projection, occlusion and shading are done here.

`plot` shows the surface in 3D and in (u, v), `imshow` shows the grids as images. Two
colourings: RGB = normalised x, y, z, and a checkerboard on the grid cells with black
borders. A good flattening gives smooth colours and regular, undistorted squares.

The surface is drawn as the quads between neighbouring cells of the grid, sorted back to
front in 3D (painter's algorithm) and shaded. Quads that leave the surface (the grid links
two parts close in space but not on the surface) are drawn in light grey.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# Checkerboard (dark blue, light orange), its borders, and the quads off the surface.
CHECKER = np.array([[0.15, 0.25, 0.45], [0.98, 0.75, 0.45]])
BORDER, TORN = np.zeros(3), np.full(3, 0.92)
# Width of the borders: in points (2D panels), relative to the object size (3D panels).
LINE_2D, LINE_3D = 1.2, 0.006
# 3D view: elevation, and azimuth offset from a pure side view (degrees).
ELEV, AZIM = 15, 30
AMBIENT = 0.55  # shading: brightness of a surface seen edge-on (1 facing the light)
# A quad leaves the surface if its centre or the middle of one of its sides is further
# than OFF x the typical spacing between points from all the points of the cloud.
OFF = 4


def plot(flat, squares=4):
    """First column: all the faces in 3D. Then one column per face in (u, v).
    First row coloured by RGB, second row by checkerboard.

    Args:
        squares: number of checkerboard squares along each side of the grid.
    """
    k = len(flat.faces)
    xyz = {f: flat.points[flat.indexes[f]] for f in flat.faces}
    on = _on_surface(flat, xyz, squares)
    proj = _camera(flat._lateral, flat._up)
    fig = plt.figure(figsize=(4 + 3 * k, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 1 + k, width_ratios=[1.5] + [1] * k)
    for row, label in enumerate(("RGB", "checkerboard")):
        colors = {f: _quad_colors(flat, f, label, squares) for f in flat.faces}
        checker = label == "checkerboard"

        ax = fig.add_subplot(grid[row, 0])
        _draw3d(ax, [(xyz[f], colors[f], on[f]) for f in flat.faces], proj, squares,
                checker)
        ax.set_title(f"xyz, {label}")

        for col, f in enumerate(flat.faces, start=1):
            ax = fig.add_subplot(grid[row, col])
            quads, (keep, seg_keep) = _quads(flat.isomap[f]), on[f]
            c = np.where(keep[..., None], colors[f], TORN).reshape(-1, 3)
            ax.add_collection(PolyCollection(quads.reshape(-1, 4, 2), facecolors=c,
                                             edgecolors=c, linewidths=0.3))
            if checker:
                ax.add_collection(LineCollection(_segments(flat.isomap[f], squares)[seg_keep],
                                                 colors=[BORDER], linewidths=LINE_2D,
                                                 capstyle="round"))
            ax.set(aspect="equal", xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1],
                   yticks=[0, 0.5, 1], xlabel="u", ylabel="v", title=f"{f}: uv, {label}")
    return fig


def imshow(flat, colors="rgb", squares=4, sigma=2):
    """The grids as images, one per face.

    Args:
        colors: "rgb" (normalised x, y, z of each cell's point) or "checkerboard".
        sigma: Gaussian smoothing of the RGB image, in cells (0: none).
    """
    if colors == "rgb":
        images = {f: gaussian_filter(img, (sigma, sigma, 0), mode="nearest")
                  for f, img in _rgb(flat).items()}
    else:
        i, j = np.indices((flat._n, flat._n))
        tile = int(np.ceil(flat._n / squares))
        image = CHECKER[(i // tile + j // tile) % 2]
        edge = np.isin(np.arange(flat._n), _edges(flat._n, squares))
        image[edge[:, None] | edge[None, :]] = BORDER
        images = dict.fromkeys(flat.faces, image)
    fig, axes = plt.subplots(1, len(flat.faces), figsize=(4 * len(flat.faces), 4.4),
                             squeeze=False, layout="constrained")
    for ax, f in zip(axes[0], flat.faces):
        ax.imshow(images[f], interpolation="nearest")
        ax.set_title(f"{f}: grid, {colors}")
        ax.set_axis_off()
    return fig


def _rgb(flat):
    """{face: (n, n, 3)} normalised x, y, z, on a common scale for all the faces."""
    lo, hi = flat.points.min(0), flat.points.max(0)
    return {f: (flat.points[flat.indexes[f]] - lo) / np.where(hi > lo, hi - lo, 1)
            for f in flat.faces}


def _edges(n, squares):
    """Rows (or columns) on the edge of a checkerboard square: the first and the last of
    each square, e.g. n=20, squares=4: 0, 4, 5, 9, 10, 14, 15, 19."""
    tile = int(np.ceil(n / squares))
    r = np.arange(n)
    return r[(r % tile == 0) | (r % tile == tile - 1) | (r == n - 1)]


def _quads(grid):
    """The (n-1)^2 quads between neighbouring cells of an (n, n, d) grid of coordinates,
    as an (n-1, n-1, 4, d) array."""
    return np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2)


def _on_surface(flat, xyz, squares):
    """{face: (quads (n-1, n-1) bool, segments (S,) bool)}: True where the quad (or the
    checkerboard segment) stays on the surface, i.e. its centre and the middles of its
    sides are within OFF x the typical spacing of a point of the cloud."""
    tree = cKDTree(flat.points)
    off = OFF * np.median(tree.query(flat.points, 2)[0][:, 1])
    out = {}
    for f in flat.faces:
        q = _quads(xyz[f])
        probes = np.concatenate([q.mean(2, keepdims=True), (q + np.roll(q, 1, 2)) / 2], 2)
        quads = (tree.query(probes.reshape(-1, 3))[0] <= off).reshape(probes.shape[:3])
        seg = _segments(xyz[f], squares)
        t = np.linspace(0, 1, 5)[:, None, None]  # 5 probes along each segment
        probes = (seg[:, 0] + t * (seg[:, 1] - seg[:, 0])).reshape(-1, 3)
        out[f] = quads.all(-1), (tree.query(probes)[0] <= off).reshape(5, -1).all(0)
    return out


def _quad_colors(flat, f, label, squares):
    """(n-1, n-1, 3) colour of each quad: mean RGB of its corners, or its checkerboard
    square."""
    if label == "RGB":
        return _quads(_rgb(flat)[f]).mean(axis=2)
    i, j = np.indices((flat._n - 1, flat._n - 1))
    tile = int(np.ceil(flat._n / squares))
    return CHECKER[(i // tile + j // tile) % 2]


def _segments(grid, squares):
    """(S, 2, d) segments of the checkerboard borders, on an (n, n, d) grid of
    coordinates: the outline of the grid, and the lines between two squares (half-way
    between the last row of a square and the first of the next)."""
    n = len(grid)
    tile = int(np.ceil(n / squares))
    rows = [grid[0], grid[-1]] + [(grid[r - 1] + grid[r]) / 2 for r in range(tile, n, tile)]
    cols = [grid[:, 0], grid[:, -1]] + \
        [(grid[:, c - 1] + grid[:, c]) / 2 for c in range(tile, n, tile)]
    return np.concatenate([np.stack([line[:-1], line[1:]], axis=1) for line in rows + cols])


def _camera(lateral, up):
    """(3, 3) rotation from xyz to the camera axes (right, up, towards the viewer): a side
    view from the `lateral` axis, from the side where the remaining axis points right (as
    u does in the 2D panels), then turned by AZIM around `up` and raised by ELEV."""
    e = np.eye(3)
    l, u = "xyz".index(lateral), "xyz".index(up)
    h = e[3 - l - u]
    side = e[l] * np.sign(np.cross(e[u], e[l]) @ h)  # eye direction of the side view
    az, el = np.radians(AZIM), np.radians(ELEV)
    flat_eye = np.cos(az) * side + np.sin(az) * np.cross(e[u], side)
    towards = np.cos(el) * flat_eye + np.sin(el) * e[u]
    right = np.cross(e[u], towards)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(towards, right), towards])


def _draw3d(ax, faces, proj, squares, checker):
    """Draw the faces [(xyz (n, n, 3), quad colours (n-1, n-1, 3), `_on_surface` masks)]
    seen through `proj`, back to front, with the checkerboard borders as black ribbons."""
    polys, colors, depths = [], [], []
    extent = np.ptp(np.concatenate([g.reshape(-1, 3) for g, _, _ in faces]) @ proj.T, 0)
    light = np.array([-0.3, 0.5, 1]) / np.linalg.norm([-0.3, 0.5, 1])
    for grid, c, (keep, seg_keep) in faces:
        q, keep = _quads(grid @ proj.T).reshape(-1, 4, 3), keep.ravel()
        c = np.where(keep[:, None], c.reshape(-1, 3), TORN)
        normal = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        shade = AMBIENT + (1 - AMBIENT) * np.abs(normal @ light)
        polys.append(q[:, :, :2])
        colors.append(np.where(keep[:, None], c * shade[:, None], TORN))
        depths.append(q[:, :, 2].mean(1))
        if checker:  # ribbons, slightly in front of the surface they lie on
            seg = _segments(grid @ proj.T, squares)[seg_keep]
            a, b = seg[:, 0, :2], seg[:, 1, :2]
            d = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-12)
            w = LINE_3D * extent[:2].max() / 2
            a, b, side = a - w * d, b + w * d, w * np.stack([-d[:, 1], d[:, 0]], 1)
            polys.append(np.stack([a - side, b - side, b + side, a + side], axis=1))
            colors.append(np.tile(BORDER, (len(seg), 1)))
            depths.append(seg[:, :, 2].mean(1) + 0.01 * extent[2])
    order = np.argsort(np.concatenate(depths))  # far first
    polys, colors = np.concatenate(polys)[order], np.concatenate(colors)[order]
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=colors,
                                     linewidths=0.3))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
