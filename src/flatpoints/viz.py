"""Plots of the output of `flatten`.

`plot` shows the points in 3D and in (u, v); `imshow` shows the grids themselves, as
images. Both use the same two colourings: RGB = normalised x, y, z, and a checkerboard on
the grid cells.

`plot` draws four kinds of panels:
    1. the points in 3D, coloured by position (RGB = normalised x, y, z),
    2. the points in 3D, coloured by a checkerboard on their grid cell,
    3. the flat isomap (u, v), with the RGB colours of panel 1,
    4. the flat isomap (u, v), with the checkerboard colours of panel 2.
Layout: first row coloured by RGB, second row by checkerboard. The first column shows all
the faces together in 3D, then one column per face in (u, v).
A good flattening shows smooth colours in panel 3, and regular, undistorted squares in
panel 2 (neighbouring cells hold points close on the surface).
"""
import matplotlib.pyplot as plt
import numpy as np

# The two colours of the checkerboard (dark blue, light orange).
CHECKER = np.array([[0.15, 0.25, 0.45], [0.98, 0.75, 0.45]])
# Zoom of the 3D panels.
ZOOM = 1.2
# 3D view: elevation, and azimuth offset from a pure side view (degrees).
ELEV, AZIM = 15, 30


def plot(result, points, lateral="x", up=None, squares=4):
    """Plot the output of `flatten`: 3D panels with all the faces, 2D panels per face.

    Args:
        result: dict returned by `flatten` (one or two faces).
        points: (N, 3) the point cloud given to `flatten`.
        lateral: the `lateral` axis given to `flatten`. The 3D view looks at the object
            from that side, slightly oblique.
        up: the `up` axis given to `flatten` (same default), shown vertically in the 3D
            panels. In the 2D panels v is vertical, and follows `up` (rotation=True).
        squares: number of checkerboard squares along each side of the grid.

    Returns:
        The matplotlib Figure.
    """
    if up is None:  # same default as `flatten`: the last non-lateral axis
        up = "y" if lateral == "z" else "z"
    if lateral == up:
        raise ValueError("lateral and up must be different axes")
    points = np.asarray(points, float)
    faces = _faces(result)

    # Layout: row 0 coloured by RGB, row 1 by checkerboard. Column 0 is the 3D view of all
    # the faces together, then one column per face in (u, v).
    fig = plt.figure(figsize=(6 + 5 * len(faces), 10))
    grid = fig.add_gridspec(2, 1 + len(faces), width_ratios=[1.3] + [1] * len(faces))
    X_all, colors_all = [], [[], []]
    for f, (name, idx, uv) in enumerate(faces):
        real = idx >= 0
        X = points[idx[real]]
        n = len(idx)
        rgb, checker = (c[real] for c in _colors(idx, points, squares))
        X_all.append(X)

        prefix = f"{name}: " if name else ""
        for r, (colors, label) in enumerate([(rgb, "RGB"), (checker, "checkerboard")]):
            colors_all[r].append(colors)
            ax = fig.add_subplot(grid[r, 1 + f])
            _scatter2d(ax, uv[real], colors, n)
            ax.set_title(f"{prefix}uv, {label}")

    X = np.concatenate(X_all)
    for r, label in enumerate(["RGB", "checkerboard"]):
        ax = fig.add_subplot(grid[r, 0], projection="3d")
        _scatter3d(ax, X, np.concatenate(colors_all[r]), lateral, up)
        ax.set_title(f"xyz, {label}")
    fig.tight_layout()
    return fig


def imshow(result, points):
    """Show the grids of `flatten` as images, one per face, coloured by RGB
    (normalised x, y, z of each cell's point). Empty cells are left blank.

    Args:
        result: dict returned by `flatten` (one or two faces).
        points: (N, 3) the point cloud given to `flatten`.

    Returns:
        The matplotlib Figure.
    """
    points = np.asarray(points, float)
    faces = _faces(result)
    fig, axes = plt.subplots(1, len(faces), figsize=(5 * len(faces), 5), squeeze=False)
    lo, hi = points.min(0), points.max(0)
    for ax, (name, idx, _) in zip(axes[0], faces):
        # RGB normalised over the whole cloud, so all the faces share the same colours
        rgb = np.clip((points[idx] - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)
        # RGBA, empty cells fully transparent
        rgba = np.dstack([rgb, (idx >= 0).astype(float)])
        ax.imshow(rgba, interpolation="nearest")
        ax.set_title(f"{name}: grid, RGB" if name else "grid, RGB")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def _colors(idx, points, squares):
    """RGB and checkerboard colours of an (n, n) grid of indexes.

    Returns:
        rgb, checker: (n, n, 3) arrays, nan for empty cells. RGB is normalised over the
        whole cloud, so all the faces share the same colours.
    """
    lo, hi = points.min(0), points.max(0)
    rgb = np.clip((points[idx] - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)
    n = len(idx)
    tile = int(np.ceil(n / squares))  # side of one checkerboard square, in cells
    i, j = np.indices((n, n))
    checker = CHECKER[(i // tile + j // tile) % 2]
    rgb[idx < 0] = checker[idx < 0] = np.nan
    return rgb, checker


def _faces(result):
    """[(name, indexes, isomap)] for each face of a `flatten` output."""
    idx, uv = result["indexes"], result["isomap"]
    if isinstance(idx, dict):  # two faces: {"left": ..., "right": ...}
        return [(k, idx[k], uv[k]) for k in idx]
    return [("", idx, uv)]


def _scatter3d(ax, X, colors, lateral, up):
    """3D scatter of X (M, 3), at true scale, `up` vertical, seen from the `lateral` side."""
    l, u = "xyz".index(lateral), "xyz".index(up)
    h = 3 - l - u  # the remaining, horizontal axis
    # Plot axes: a cyclic permutation of xyz with `up` last (no mirroring)
    order = [(u + 1) % 3, (u + 2) % 3, u]
    X = X[:, order]
    ax.scatter(*X.T, c=colors, s=4, linewidths=0, depthshade=False)

    # Same scale on the three axes: limits fitted to the data, box shaped like the data
    lo, hi = X.min(0), X.max(0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    # zoom > 1 removes most of the empty margin matplotlib leaves around 3D axes
    ax.set_box_aspect(np.maximum(hi - lo, 1e-12), zoom=ZOOM)

    # Camera on the side of the lateral axis from which the horizontal axis h points to
    # the right, as u does in the 2D panels.
    e = np.eye(3)
    side = -np.sign(np.cross(e[l], e[u]) @ e[h])
    eye = (side * e[l])[order]  # camera direction, in plot axes
    ax.view_init(elev=ELEV, azim=np.degrees(np.arctan2(eye[1], eye[0])) + AZIM)
    ax.set_axis_off()


def _scatter2d(ax, Y, colors, n):
    """2D scatter of (u, v) isomap Y (M, 2), marker size matched to an n x n grid."""
    ax.scatter(*Y.T, c=colors, s=(150 / n) ** 2, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("u")
    ax.set_ylabel("v")