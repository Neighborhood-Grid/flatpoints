"""Plots of a `Flat` and a `Resampler`, used as methods: `flat.plot()`,
`flat.imshow(data)`, `flat.imshow(data, animate=...)` (see `main.Viz`), `sampler.plot()`.
`imshow` also works on its own, on any (n, n, *C) or (n_faces, n, n, *C) array.

The surfaces are drawn as the quads between neighbouring pixels of the images, opaque and
shaded by their orientation, far first (occlusion). Colourings: RGB = normalised x, y, z,
or a checkerboard on the pixels; a good flattening gives smooth colours and regular squares.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .utils import axes

CHECKER = np.array([[0.15, 0.25, 0.45], [0.98, 0.75, 0.45]])  # dark blue, light orange
BIVARIATE = np.array([[[0.91, 0.91, 0.91], [0.35, 0.65, 0.85]],  # (0, 0) grey, (0, 1) blue
                      [[0.85, 0.45, 0.35], [0.25, 0.15, 0.40]]])  # (1, 0) red, (1, 1) purple
SUN = np.array([-0.3, 0.5, 1]) / np.linalg.norm([-0.3, 0.5, 1])  # light direction (camera)
# morph, per resolution: grid size, frames per second, inches per panel, supersampling,
# WebP quality and method
RESOLUTIONS = {"low": (40, 15, 3.0, 1, 75, 0), "high": (None, 25, 4.5, 2, 90, 4)}


# ==== Plots ============================================================================

def plot(flat, uv="geodesic_shortcut", squares=8, point_size=1.0):
    """First column: all the faces in 3D. Then one column per face in (u, v), `uv` being
    "geodesic_shortcut" (flat.uv_geodesic_shortcut) or "conformal" (flat.uv_conformal).
    First row coloured by RGB, second row by a checkerboard of `squares` x `squares`
    pixels. `point_size` scales the points (1 = default). Returns the figure."""
    if uv not in ("geodesic_shortcut", "conformal"):
        raise ValueError('uv must be "geodesic_shortcut" or "conformal"')
    grids = _grids(flat, getattr(flat, f"uv_{uv}"))
    k, m = len(grids), len(grids[0]["xyz"])  # m: side of the drawn grids
    proj = camera(flat.normal_axis)
    fig = plt.figure(figsize=(4 + 3 * k, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 1 + k, width_ratios=[1.5] + [1] * k)
    names = ["left: ", "right: "] if k == 2 else [""]
    for row, colouring in enumerate(("RGB", "checkerboard")):
        cols = [_checker(m, squares) if row else g["rgb"] for g in grids]
        # 3D: the quads, and on each pixel a disc in full colour, one grid step towards the
        # viewer so that it covers the quads of its own surface, not a surface in front
        p = [g["xyz"] @ proj.T for g in grids]
        step = np.median(np.concatenate([np.linalg.norm(np.diff(q, axis=a), axis=-1).ravel()
                                         for q in p for a in (0, 1)]))
        layers = []
        for q, col in zip(p, cols):
            quads, shade = quad_layer(q, col)
            s = np.pad(shade, 1, mode="edge")  # shade of a pixel: mean of the quads around
            s = (s[:-1, :-1] + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:]) / 4
            layers += [quads, disc_layer(q, col, s, 0.4 * point_size * step, push=step)]
        render(fig.add_subplot(grid[row, 0], title=f"xyz, {colouring}"), layers)
        for c, (g, col, name) in enumerate(zip(grids, cols, names), start=1):
            ax = fig.add_subplot(grid[row, c], aspect="equal", xlim=(0, 1), ylim=(0, 1),
                                 xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel="u", ylabel="v",
                                 title=f"{name}{uv.split('_')[0]} uv")
            _add(ax, _quads(g["uv"]).reshape(-1, 4, 2), _quads(col).mean(2).reshape(-1, 3))
            ax.scatter(*g["uv"].reshape(-1, 2).T, c=col.reshape(-1, 3),
                       s=(300 * point_size / m) ** 2, linewidths=0, rasterized=True)
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
        before.set(aspect="equal", title=f"original ({len(p)} points)")
        before.set_axis_off()
        ax.set_title(f"resampled ({len(points)} points)")
    if radius is None:
        radius = 0.19 * point_size * np.sqrt(surface_area(flat.points) / len(points))
    layers = []
    for f, g in enumerate(_grids(flat)):
        quads, shade = quad_layer(g["xyz"] @ proj.T, g["rgb"], light)
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
    NaN pixels are left transparent. Returns the figure."""
    a = gaussian_filter(_faces(np.asarray(image, float)), (0, sigma, sigma, 0))
    k, c = len(a), a.shape[-1]
    side = {1: 0.07, 2: 0.3}.get(c)  # width of the colorbar / key, relative to a face
    fig = plt.figure(figsize=(4 * k + 4 * (side or 0) + 0.6 * (c <= 2), 4),
                     layout="constrained")
    grid = fig.add_gridspec(1, k + (c <= 2), width_ratios=[1] * k + [side] * (c <= 2))
    rgba = _colour(a, cmap)
    for f, name in zip(range(k), ["left", "right"] if k == 2 else [""]):
        ax = fig.add_subplot(grid[0, f], title=name + (" (PCA)" if c > 3 else ""))
        ax.imshow(rgba[f], interpolation="nearest")
        ax.set_axis_off()
    lo, hi = np.nanmin(a, (0, 1, 2)), np.nanmax(a, (0, 1, 2))
    if c == 1:
        fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(lo[0], hi[0]), cmap),
                     cax=fig.add_subplot(grid[0, k]))
    elif c == 2:  # key: the colour square, axes in the units of the channels
        key = fig.add_subplot(grid[0, k], xlabel="channel 0", ylabel="channel 1")
        s = np.linspace(0, 1, 64)
        key.imshow(_bivariate(np.stack(np.meshgrid(s, s), -1)), origin="lower",
                   extent=(lo[0], hi[0], lo[1], hi[1]), aspect="auto")
        key.set_box_aspect(1)
        key.set_xticks([lo[0], hi[0]], [f"{lo[0]:.3g}", f"{hi[0]:.3g}"])
        key.set_yticks([lo[1], hi[1]], [f"{lo[1]:.3g}", f"{hi[1]:.3g}"])
        key.tick_params(labelsize=7)
    return fig


def morph(image, xyz, normal_axis="x", sigma=0.5, cmap="viridis", path=None,
          resolution="low", loops=5, seconds=1.2, pause=0.5):
    """Animation, one panel per face: the quads of the face's image fold from the flat
    image into the surface in 3D, and back. Colours as `imshow` (`image`, `sigma`,
    `cmap`), shaded as in `plot`; `xyz` (n_faces, n, n, 3): the point of each pixel, e.g.
    `flat.to_image(flat.points)`, seen as in `plot` (`normal_axis`), at the same scale in
    all the panels. A morph lasts `seconds`, each end is held `pause` seconds.

    `resolution`: "low" (a few seconds: the grids downsampled to about 40 x 40 pixels,
    smoothed first) or "high" (all the pixels, 25 frames per second, rendered at twice
    the size then reduced, no aliasing: up to a minute to write). Saved as an animated
    WebP (smooth gradients, light, shown by browsers, GitHub and VS Code) to `path`
    (".webp") if given, played `loops` times (there and back) then still. Returns the
    matplotlib animation (in a notebook: `IPython.display.HTML(anim.to_jshtml())`)."""
    from matplotlib.animation import FuncAnimation
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {list(RESOLUTIONS)}, got {resolution!r}")
    if path is not None and not str(path).lower().endswith(".webp"):
        raise ValueError(f"the animation is saved as an animated WebP: path must end with "
                         f".webp, got {path!r}")
    size, fps, inches, scale, quality, method = RESOLUTIONS[resolution]
    if resolution == "high":
        warnings.warn("high resolution animation activated, writing the webp will take up "
                      "to a minute...", stacklevel=3)
    a = _faces(np.asarray(image, float))
    k, n = len(a), a.shape[1]
    st = 1 if size is None else max(1, int(np.ceil(n / size)))  # every st-th pixel
    a = gaussian_filter(a, (0, max(sigma, st / 2), max(sigma, st / 2), 0))
    rgba = _colour(a, cmap)[:, ::st, ::st]
    end = (np.asarray(xyz, float).reshape(k, n, n, 3) @ camera(normal_axis).T)[:, ::st, ::st]
    m = end.shape[1]
    step = max(np.ptp(e[..., 1]) for e in end) / (m - 1)  # images as tall as the tallest face
    i, j = np.indices((m, m))
    flat = np.stack([j - (m - 1) / 2, (m - 1) / 2 - i, np.zeros((m, m))], -1) * step
    mids = [(e.reshape(-1, 3).min(0) + e.reshape(-1, 3).max(0)) / 2 for e in end]
    start = [flat + mid * [1, 1, 0] for mid in mids]  # each image centred on its face
    half = step + np.max([np.abs(np.concatenate([s, e]).reshape(-1, 3) - mid)[:, :2].max(0)
                          for s, e, mid in zip(start, end, mids)], 0)  # same window for all

    fig, panels = plt.subplots(1, k, figsize=(inches * k, inches * (half[1] / half[0] + 0.1)),
                               dpi=100 * scale, squeeze=False)
    fig.subplots_adjust(0, 0, 1, 0.92, wspace=0.02)
    collections = []
    for ax, mid, name in zip(panels[0], mids, ["left", "right"] if k == 2 else [""]):
        ax.set(xlim=(mid[0] - half[0], mid[0] + half[0]), title=name,
               ylim=(mid[1] - half[1], mid[1] + half[1]), aspect="equal")
        ax.set_axis_off()
        collections.append(ax.add_collection(PolyCollection([], linewidths=0.3)))

    def frame(t):
        for s, e, c, quads in zip(start, end, rgba, collections):
            layer, _ = quad_layer((1 - t) * s + t * e, c)
            order = np.argsort(layer[2], kind="stable")  # far first
            quads.set_verts([layer[0][o] for o in order])
            quads.set_facecolors(layer[1][order])
            quads.set_edgecolors(_edges(layer[1][order]))
        return collections

    hold, move = round(pause * fps), round(seconds * fps)
    ease = np.linspace(0, 1, move) ** 2 * (3 - 2 * np.linspace(0, 1, move))  # smoothstep
    times = np.concatenate([np.zeros(hold), ease, np.ones(hold), ease[::-1]])
    if path is not None:  # before the animation, which hides its artists from fig.draw
        _save_webp(fig, frame, times, path, fps, loops, scale, quality, method)
    anim = FuncAnimation(fig, frame, frames=times, interval=1000 / fps, blit=True)
    plt.close(fig)
    return anim


# ==== Building blocks ==================================================================

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


def quad_layer(g, col, light=1.0):
    """Layer (polygons, rgba, depth) of the quads of a projected grid `g` (n, n, 3),
    coloured by the mean of the cell colours `col` (n, n, 3 or 4) washed towards white
    (`light`: 0 = white, 1 = full colour), shaded by their orientation; opaque, unless
    `col` has an alpha channel (a quad takes the lowest of its cells).
    Returns (layer, shade (n-1, n-1))."""
    q, c = _quads(g), _quads(col)
    n = np.cross(q[..., 2, :] - q[..., 0, :], q[..., 3, :] - q[..., 1, :])
    shade = 0.55 + 0.45 * abs(n @ SUN) / np.maximum(np.linalg.norm(n, axis=-1), 1e-12)
    rgb = (1 - light * (1 - c[..., :3].mean(2))) * shade[..., None]
    alpha = c[..., 3].min(2) if col.shape[-1] == 4 else np.ones(shade.shape)
    layer = (list(q[..., :2].reshape(-1, 4, 2)), _rgba(rgb, alpha).reshape(-1, 4),
             q[..., 2].mean(-1).ravel())
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


def _grids(flat, uv=None):
    """Per face, every st-th pixel of the images (st: about 110 pixels per side at most):
    {"xyz", "rgb", "uv"} (m, m, .) grids of the points, their RGB and `uv` (N, 2)."""
    s = slice(None, None, max(1, flat.n // 110))
    norm = rgb(flat.points)
    return [{"xyz": flat.points[g[s, s]], "rgb": norm[g[s, s]],
             "uv": None if uv is None else uv[g[s, s]]}
            for g in flat.to_image(np.arange(len(flat.points)))]


def _save_webp(fig, frame, times, path, fps, loops=5, scale=1, quality=80, method=0):
    """Save the frames as an animated WebP, played `loops` times, then still. Frames
    rendered at `scale` times the size are reduced (supersampling: no aliasing).
    `quality` (0-100) and `method` (0 fast - 6 small) of the lossy WebP encoder."""
    from PIL import Image
    frames = []
    for t in times:
        frame(t)
        fig.canvas.draw()
        f = Image.fromarray(np.array(fig.canvas.buffer_rgba())[..., :3])
        frames.append(f.resize((f.width // scale, f.height // scale), Image.Resampling.LANCZOS)
                      if scale > 1 else f)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
                   loop=loops, quality=quality, method=method)


def _faces(a):
    """(F, n, n, C) from (n, n, *C) or (F, n, n, *C), F = 1 or 2."""
    if a.ndim >= 2 and a.shape[0] == a.shape[1]:
        a = a[None]
    elif not (a.ndim >= 3 and a.shape[0] in (1, 2) and a.shape[1] == a.shape[2]):
        raise ValueError(f"expected (n, n, *C), (1, n, n, *C) or (2, n, n, *C), got {a.shape}")
    return a.reshape(*a.shape[:3], -1)


def _colour(a, cmap="viridis"):
    """(F, n, n, 4) RGBA of the (F, n, n, C) channels, as in `imshow`; NaN: transparent."""
    c = a.shape[-1]
    if c == 1:
        colours = plt.get_cmap(cmap)(_unit(a)[..., 0])[..., :3]
    elif c == 2:
        colours = _bivariate(_unit(a))
    else:
        colours = _unit(a if c == 3 else _pca(a, 3))
    ok = np.isfinite(a).all(-1, keepdims=True)
    return np.concatenate([np.where(ok, colours, 1), ok], -1)


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


def _checker(n, squares):
    """(n, n, 3) colours of the cells of an n x n grid: `squares` x `squares` squares."""
    i, j = np.indices((n, n)) // int(np.ceil(n / squares))
    return CHECKER[(i + j) % 2]


def _quads(grid):
    """(n-1, n-1, 4, d) the quads between neighbouring cells of an (n, n, d) grid."""
    return np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2)


def _rgba(rgb, alpha):
    return np.concatenate([rgb, alpha[..., None]], -1)


def _edges(rgba):
    """Edge colours: the face colour (no hairline gaps), none on see-through polygons."""
    rgba = np.asarray(rgba, float)
    if rgba.shape[-1] == 3:
        return rgba
    return np.where(rgba[:, 3:] < 1, 0, rgba)


def _add(ax, polys, rgba):
    """Draw the polygons, edges in the face colour."""
    ax.add_collection(PolyCollection(polys, facecolors=rgba, edgecolors=_edges(rgba),
                                     linewidths=0.3))
