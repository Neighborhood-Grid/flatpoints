"""Plots of a `Flat`: `plot` shows the surface in 3D and in (u, v), `imshow` the grids.
Plot of a `Resampler`: `plot_resampler` shows the resampled points on the surface of the Flat.
Use them as methods: `flat.plot()`, `flat.imshow(data)` (see `main.Viz`), `sampler.plot()`;
`imshow` also works on its own, on any (n, n, *C) or (n_faces, n, n, *C) array.

Two colourings: RGB = normalised x, y, z, and a checkerboard on the grid cells. A good
flattening gives smooth colours and regular, undistorted squares.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter, label, median_filter
from scipy.spatial import cKDTree

from .utils import axes

CHECKER = np.array([[0.15, 0.25, 0.45], [0.98, 0.75, 0.45]])  # dark blue, light orange
JUMP, ALPHA_RATE = 3, 0.5  # opacity ALPHA when a side is JUMP times the median side around it
ALPHA_MAX = 0.5
TAU = (JUMP - 1) / np.sqrt(-np.log(ALPHA_RATE))  # width of the fade, see `_alpha`
LIGHT = 1.0  # strength of the background quads in (u, v): 0 = white, 1 = full colour
SUN = np.array([-0.3, 0.5, 1]) / np.linalg.norm([-0.3, 0.5, 1])  # light direction (camera)


def plot(flat, uv="geodesic_shortcut", squares=8, point_size=1.0):
    """First column: all the faces in 3D. Then one column per face in (u, v), `uv` being
    "geodesic_shortcut" (flat.uv_geodesic_shortcut) or "conformal" (flat.uv_conformal).
    First row coloured by RGB, second row by a checkerboard of `squares` x `squares`
    pixels. `point_size` scales the diameter of the points (1 = default)."""
    st = max(1, flat.n // 110)  # downsampling for faster plot
    grids = sampled_grids(flat, st, uv)
    k, m = len(grids), len(next(iter(grids.values()))["xyz"])  # m: side of the drawn grids
    proj = camera(flat.normal_axis)
    fig = plt.figure(figsize=(4 + 3 * k, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 1 + k, width_ratios=[1.5] + [1] * k)
    size = (300 * point_size / m) ** 2  # marker area (pt^2): about half a grid step
    for row, name in enumerate(("RGB", "checkerboard")):
        col = {f: _checker(m, squares) if row else g["rgb"] for f, g in grids.items()}
        ax = fig.add_subplot(grid[row, 0], title=f"xyz, {name}")
        _draw3d(ax, [(g["xyz"] @ proj.T, col[f], g["alpha"]) for f, g in grids.items()],
                point_size)
        for c, (f, g) in enumerate(grids.items(), start=1):
            ax = fig.add_subplot(grid[row, c], aspect="equal", xlim=(0, 1), ylim=(0, 1),
                                 xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel="u", ylabel="v",
                                 title=f"{f + ': ' if f else ''}{uv.split('_')[0]} uv")
            light = 1 - LIGHT * (1 - _quads(col[f]).mean(2))  # quads, washed towards white
            _add(ax, _quads(g["uv"]).reshape(-1, 4, 2), _rgba(light, g["alpha"]).reshape(-1, 4))
            ax.scatter(*g["uv"].reshape(-1, 2).T, c=col[f].reshape(-1, 3), s=size,
                       linewidths=0, rasterized=True)
    return fig


def plot_resampler(sampler, ax=None, point_size=1.0, light=0.45, radius=None):
    """Before / after, in 3D. Left: the input point cloud (black dots). Right: the
    resampled points (discs, RGB colours) on the mesh of the input cloud (quads of
    `sampler.flat`, washed towards white by `light`). With `ax`, only the right view,
    drawn in `ax`. `radius` of the discs: default from the mean area per point, so that
    two resamplers with as many points get the same discs. Returns the figure."""
    flat, points = sampler.flat, sampler.points
    proj = camera(flat.normal_axis)
    if ax is None:
        fig, (before, ax) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True,
                                         layout="constrained")
        p = flat.points @ proj.T
        before.scatter(p[:, 0], p[:, 1], s=1, c="k", linewidths=0)
        before.set_aspect("equal")
        before.set_axis_off()
        before.set_title(f"original ({len(p)} points)")
        ax.set_title(f"resampled ({len(points)} points)")
    st = max(1, flat.n // 110)
    if radius is None:
        radius = 0.19 * point_size * np.sqrt(surface_area(flat.points) / len(points))
    layers = []
    for f, g in enumerate(sampled_grids(flat, st).values()):
        # opaque mesh (hides the points behind it), see-through only across the seams
        quads, shade = quad_layer(g["xyz"] @ proj.T, g["rgb"], g["alpha"] / ALPHA_MAX, light)
        layers.append(quads)
        mine = points[sampler.faces == f]
        if len(mine):  # shade of a point: the one of the nearest quad
            _, near = cKDTree(_quads(g["xyz"]).mean(2).reshape(-1, 3)).query(mine)
            layers.append(disc_layer(mine @ proj.T, rgb(mine, flat.points),
                                     shade.ravel()[near], radius, push=radius, sides=10))
    render(ax, layers)
    return ax.figure


def imshow(image, sigma=0.5, cmap="viridis"):
    """Grids as images, one per face: `image` (n, n, *C) or (1, n, n, *C) for one face,
    (2, n, n, *C) for two (left, right), e.g. `flat.to_image(points)`. The C = prod(*C)
    channels are smoothed by a Gaussian of `sigma` pixels, then coloured with a range
    shared by the faces:
      C = 1: colormap `cmap`, with a colorbar;
      C = 2: bivariate colormap (one channel per axis of a colour square), with its key;
      C = 3: RGB = the channels normalised (e.g. x, y, z);
      C > 3: RGB = the 3 first principal components (PCA over all the pixels).
    NaN pixels are left transparent."""
    a = gaussian_filter(_faces(np.asarray(image, float)), (0, sigma, sigma, 0))
    k, c = len(a), a.shape[-1]
    side = {1: 0.07, 2: 0.3}.get(c)  # width of the colorbar / key, relative to a face
    fig = plt.figure(figsize=(4 * k + 4 * (side or 0) + 0.6 * (c <= 2), 4),
                     layout="constrained")
    grid = fig.add_gridspec(1, k + (c <= 2), width_ratios=[1] * k + [side] * (c <= 2))
    axes = [fig.add_subplot(grid[0, f]) for f in range(k)]
    names = ["left", "right"] if k == 2 else [""]
    if c == 1:
        v = a[..., 0]
        lo, hi = np.nanmin(v), np.nanmax(v)
        ims = [ax.imshow(v[f], cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
               for f, ax in enumerate(axes)]
        fig.colorbar(ims[0], cax=fig.add_subplot(grid[0, k]))
    else:
        if c == 2:
            colours = _bivariate(_unit(a))
        else:
            colours = _unit(a if c == 3 else _pca(a, 3))
        ok = np.isfinite(a).all(-1, keepdims=True)
        rgba = np.concatenate([np.where(ok, colours, 1), ok], -1)
        for f, ax in enumerate(axes):
            ax.imshow(rgba[f], interpolation="nearest")
        if c == 2:  # key: the colour square, axes in the units of the channels
            key = fig.add_subplot(grid[0, k], xlabel="channel 0", ylabel="channel 1")
            (x0, y0), (x1, y1) = np.nanmin(a, (0, 1, 2)), np.nanmax(a, (0, 1, 2))
            s = np.linspace(0, 1, 64)
            key.imshow(_bivariate(np.stack(np.meshgrid(s, s), -1)), origin="lower",
                       extent=(x0, x1, y0, y1), aspect="auto")
            key.set_box_aspect(1)
            key.set_xticks([x0, x1], [f"{x0:.3g}", f"{x1:.3g}"])
            key.set_yticks([y0, y1], [f"{y0:.3g}", f"{y1:.3g}"])
            key.tick_params(labelsize=7)
    for ax, name in zip(axes, names):
        ax.set_axis_off()
        ax.set_title(name + (" (PCA)" if c > 3 else ""))
    return fig


# ==== Building blocks ==================================================================

def sampled_grids(flat, st=1, uv="geodesic_shortcut"):
    """{face: {"xyz", "uv", "rgb": (m, m, .) grids, "alpha": (m-1, m-1)}}, every `st`-th
    cell of the (n, n) grids of `flat`, with its `uv` ("geodesic_shortcut" or
    "conformal"). A drawn quad takes the lowest opacity of the quads of the full grid it
    covers."""
    if uv not in ("geodesic_shortcut", "conformal"):
        raise ValueError('uv must be "geodesic_shortcut" or "conformal"')
    coords = getattr(flat, f"uv_{uv}")
    s, norm = slice(None, None, st), rgb(flat.points)
    out = {}
    for f, g in _grids(flat).items():
        a = _alpha(coords[g])
        m = len(g[s, s])
        out[f] = {"xyz": flat.points[g[s, s]], "uv": coords[g[s, s]], "rgb": norm[g[s, s]],
                  "alpha": np.minimum.reduce([a[i::st, j::st][:m - 1, :m - 1]
                                              for i in range(st) for j in range(st)])}
    return out


def rgb(points, reference=None):
    """(N, 3) colours: x, y, z normalised to [0, 1] by the bounding box of `reference`."""
    ref = points if reference is None else reference
    lo, hi = ref.min(0), ref.max(0)
    return np.clip((points - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)


def surface_area(points, k=8):
    """Area of the surface, estimated from points sampled uniformly on it: the disc reaching
    the k-th nearest neighbour of a point holds about k points, i.e. an area k * A / N."""
    d = cKDTree(points).query(points, k=k + 1)[0][:, k]
    return len(points) * np.mean(np.pi * d ** 2) / k


def camera(normal_axis, elev=15, azim=30):
    """(3, 3) rotation from xyz to the camera axes (right, up, towards the viewer): a view
    along `normal_axis`, with the up axis of the images up (z, or y if normal_axis is z)
    and the third axis to the right (as u in the 2D panels), turned by `azim` degrees
    around the up axis and raised by `elev` degrees."""
    e = np.eye(3)
    l, _, u = axes(normal_axis)
    side = e[l] * np.sign(np.cross(e[u], e[l]) @ e[3 - l - u])
    az, el = np.radians(azim), np.radians(elev)
    eye = np.cos(el) * (np.cos(az) * side + np.sin(az) * np.cross(e[u], side)) \
        + np.sin(el) * e[u]
    right = np.cross(e[u], eye) / np.linalg.norm(np.cross(e[u], eye))
    return np.stack([right, np.cross(eye, right), eye])


def quad_layer(g, col, alpha, light=LIGHT):
    """Layer of the quads of a projected grid `g` (n, n, 3), coloured by the mean of the
    cell colours `col` (n, n, 3) washed towards white (`light`: 0 = white, 1 = full
    colour), shaded by their orientation, opacity `alpha` (n-1, n-1).
    Returns (layer, shade (n-1, n-1))."""
    q = _quads(g)
    n = np.cross(q[..., 2, :] - q[..., 0, :], q[..., 3, :] - q[..., 1, :])
    shade = 0.55 + 0.45 * abs(n @ SUN) / np.maximum(np.linalg.norm(n, axis=-1), 1e-12)
    pale = 1 - light * (1 - _quads(col).mean(2))
    layer = (list(q[..., :2].reshape(-1, 4, 2)), _rgba(pale * shade[..., None], alpha)
             .reshape(-1, 4), q[..., 2].mean(-1).ravel())
    return layer, shade


def disc_layer(p, col, shade, radius, push=0.0, sides=8):
    """Layer of discs (polygons of `sides` sides, facing the camera) at the projected
    points `p` (m, 3), colours `col` (m, 3) times `shade` (m,), opaque. Each disc is moved
    `push` towards the viewer, to cover the quads of its own surface."""
    p, col = p.reshape(-1, 3), col.reshape(-1, 3)
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    disc = radius * np.stack([np.cos(t), np.sin(t)], -1)
    return (list(p[:, None, :2] + disc), _rgba(col * np.reshape(shade, (-1, 1)),
                                               np.ones(len(p))), p[:, 2] + push)


def render(ax, layers):
    """Draw the layers [(polygons, rgba, depth)], all sorted together, far first."""
    polys = [poly for layer in layers for poly in layer[0]]
    order = np.argsort(np.concatenate([layer[2] for layer in layers]), kind="stable")
    _add(ax, [polys[i] for i in order], np.concatenate([layer[1] for layer in layers])[order])
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw3d(ax, faces, point_size=1.0):
    """Draw the faces [(projected xyz (n, n, 3), cell colours (n, n, 3), `_alpha`)], shaded:
    the quads in light colours, and on each cell a small disc (an octagon facing the camera)
    in full colour. A disc is moved one grid step towards the viewer, so it covers the
    quads of its own surface but not a surface in front of it."""
    step = np.median(np.concatenate([np.linalg.norm(np.diff(g, axis=a), axis=-1).ravel()
                                     for g, _, _ in faces for a in (0, 1)]))
    layers = []
    for g, col, a in faces:
        quads, shade = quad_layer(g, col, a)
        s = np.pad(shade, 1, mode="edge")  # shade of a cell: mean of the quads around it
        s = (s[:-1, :-1] + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:]) / 4
        layers += [quads, disc_layer(g, col, s, 0.4 * point_size * step, push=step)]
    render(ax, layers)


def _grids(flat):
    """{face: (n, n) index of each pixel's point}, empty pixels filled from the corners;
    faces "left", "right", or "" if single_face."""
    idx = flat.to_image(np.arange(len(flat.points)))
    return {"": idx[0]} if flat.single_face else {"left": idx[0], "right": idx[1]}


def _checker(n, squares):
    """(n, n, 3) colours of the cells of an n x n grid: `squares` x `squares` squares."""
    i, j = np.indices((n, n)) // int(np.ceil(n / squares))
    return CHECKER[(i + j) % 2]


BIVARIATE = np.array([[[0.91, 0.91, 0.91], [0.35, 0.65, 0.85]],  # (0, 0) grey, (0, 1) blue
                      [[0.85, 0.45, 0.35], [0.25, 0.15, 0.40]]])  # (1, 0) red, (1, 1) purple


def _faces(a):
    """(F, n, n, C) from (n, n, *C) or (F, n, n, *C), F = 1 or 2."""
    if a.ndim >= 2 and a.shape[0] == a.shape[1]:
        a = a[None]
    elif not (a.ndim >= 3 and a.shape[0] in (1, 2) and a.shape[1] == a.shape[2]):
        raise ValueError(f"expected (n, n, *C), (1, n, n, *C) or (2, n, n, *C), got {a.shape}")
    return a.reshape(*a.shape[:3], -1)


def _unit(a):
    """Each channel (last axis) of `a` to [0, 1] by its range over all the pixels."""
    lo, hi = np.nanmin(a, tuple(range(a.ndim - 1))), np.nanmax(a, tuple(range(a.ndim - 1)))
    return np.clip((a - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)


def _bivariate(t):
    """(..., 3) colours of (..., 2) values in [0, 1]^2: bilinear between the 4 corners of
    BIVARIATE."""
    x, y = t[..., :1], t[..., 1:2]
    return ((1 - x) * ((1 - y) * BIVARIATE[0, 0] + y * BIVARIATE[0, 1])
            + x * ((1 - y) * BIVARIATE[1, 0] + y * BIVARIATE[1, 1]))


def _pca(a, k):
    """(..., k) projection of the channels (last axis) of `a` on their k first principal
    components, fitted on the finite pixels."""
    x = a.reshape(-1, a.shape[-1])
    ok = np.isfinite(x).all(1)
    mean = x[ok].mean(0)
    vt = np.linalg.svd(x[ok] - mean, full_matrices=False)[2][:k]
    vt *= np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])[:, None]  # stable signs
    return ((x - mean) @ vt.T).reshape(*a.shape[:-1], len(vt))


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
