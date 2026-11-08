"""Ready-to-run examples on the point clouds of `fp.datasets()`, straight from the package:

    import flatpoints as fp
    flat = fp.exemple_flat("bunny")             # flattening: surface and images
    sampler = fp.exemple_sampler("bunny")       # resampling with a hyperuniform seed
    comp = fp.exemple_compression("bunny")      # compression of the xyz images

Each one prints what it computes, shows its figure (needs `flatpoints[viz]`,
`flatpoints[compress]` for the compression) and returns its result. `save="plots"`
also writes the figure in that folder (plots/, plots/resampling/, plots/compression/), and
`exemple_flat(..., animate=True)` the animation in plots/animation/.
"""
import tempfile
from pathlib import Path

import numpy as np

from .datasets import load
from .main import Compressor, Flat


def flatten(model):
    """`Flat` of the example point cloud `model`. A closed object (cow, bunny...) is cut in two
    faces, left and right, across the normal axis x. A terrain (everest) is an open surface:
    a single face, seen from above (normal axis z)."""
    points = load(model)
    everest = model == "everest"
    return Flat(points, single_face=everest, normal_axis="z" if everest else "x")


def exemple_flat(model="bunny", save=None, show=True, animate=False):
    """Flatten `model` and plot the surface in 3D with its images; with `animate`, also the
    animation of the images folding into the surface (saved as save/animation/<model>.webp
    if `save`). Returns the `Flat`."""
    flat = flatten(model)
    points = flat.points
    print("points  ", points.shape)               # (N, 3)
    print("pixels  ", flat.pixels.shape)          # (N, 3) (face, row, col), face 0 if single face
    print("images  ", (flat.n_faces, flat.n, flat.n))
    print("padding ", flat.corners.idxs.shape)    # (Npadding,) points repeated in the empty pixels
    if flat.faces is not None:
        print("faces   ", {f: len(i) for f, i in flat.faces.items()})

    # The point cloud as an image: (n_faces, n, n, 3), n_faces = 2 (left, right) or 1
    image = np.zeros((flat.n_faces, flat.n, flat.n, 3))
    image[tuple(flat.pixels.T)] = points                              # each point on its pixel
    image[tuple(flat.corners.pixels.T)] = points[flat.corners.idxs]   # fill the empty pixels
    assert np.array_equal(image, flat.to_image(points))               # the same, in one call

    # The flattening at a glance: the surface in 3D and its images, RGB and checkerboard
    fig = flat.plot()
    _finish(fig, save, f"{model}.png", show)
    # Any per-point data (N, *C) as images: flat.imshow(points[:, 2])

    # The same, animated: the images fold into the surface in 3D, and back
    if animate:
        path = None
        if save:
            path = Path(save) / "animation" / f"{model}.webp"
            path.parent.mkdir(parents=True, exist_ok=True)
        anim = flat.imshow(animate=path or True)  # resolution="high": cleaner, slower
        if path:
            print("saved    ", path)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        del anim
    return flat


def exemple_sampler(model="bunny", n_points=512, save=None, show=True):
    """Resample `model` with `n_points` new points, transported from a window of the
    hyperuniform seed. Returns the `Resampler`."""
    flat = flatten(model)
    sampler = flat.resample(n_points=n_points)
    print("flat     ", flat.n_faces, "faces of", flat.n, "x", flat.n)
    print("points   ", sampler.points.shape)   # (n_points, 3) the new points
    print("faces    ", sampler.faces.shape)    # (n_points,)   face of each new point
    print("pixels   ", sampler.pixels.shape)   # (n_points, 3) float (face, row, col)
    # Another, independent resampling, without recomputing the flat: flat.resample(n_points, seed=1)

    # The new points (discs) on the mesh of the full cloud (quads of the flat)
    fig = sampler.plot()
    _finish(fig, save and Path(save) / "resampling", f"{model}_resampled.png", show)
    return sampler


def exemple_compression(model="bunny", factor=0.8, save=None, show=True):
    """Compress `model`, compare the methods, and write / read a JPEG2000 file.
    Returns the `Compressor` of `factor`."""
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    from .compressor import available
    from .viz import camera

    flat = flatten(model)
    points = flat.points

    # `factor` of the storage saved (against float32 points), the most accurate method
    comp = flat.compress(factor=factor)
    print("method   ", comp.method)              # "dct", "wavelet" or "svd"
    print("saved    ", f"{comp.ratio:.1%}", f"({comp.bits / len(points):.1f} bits per point)")
    print("code     ", {k: getattr(v, "shape", v) for k, v in comp.code.items()})
    print("points   ", comp.points.shape)        # (N, 3) reconstructed, same order as points

    # Error of each point, in point spacings (mean distance to the nearest neighbour)
    spacing = cKDTree(points).query(points, k=2)[0][:, 1].mean()
    for method in available():
        c = flat.compress(factor, method=method)
        print(f"{method:8s} {factor:.0%}: RMS error {c.error / spacing:.3f} spacings")

    # Strong compressions (default method), coloured by the error of each point; then the
    # RMS error of each method against the storage saved
    factors = [0.9, 0.95, 0.98, 0.99]
    comps = [flat.compress(f) for f in factors]
    errors = [np.linalg.norm(c.points - points, axis=1) / spacing for c in comps]
    proj = camera(flat.normal_axis)
    fig, axs = plt.subplots(1, len(factors) + 2, figsize=(4 * (len(factors) + 2), 4),
                            layout="constrained")
    vmax = np.percentile(errors[-1], 95)
    panels = [("original", points, None)] + [
        (f"{c.ratio:.0%} saved, {c.method}", c.points, e) for c, e in zip(comps, errors)]
    for ax, (title, xyz, err) in zip(axs, panels):
        p = xyz @ proj.T
        order = np.argsort(p[:, 2])  # far first
        if err is None:
            ax.scatter(p[order, 0], p[order, 1], s=1, c="0.4")
        else:
            sc = ax.scatter(p[order, 0], p[order, 1], s=1, c=err[order], cmap="magma_r",
                            vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sc, ax=axs[-2], label="error (point spacings)")
    grid = np.linspace(0.5, 0.995, 30)
    for method in available():
        curve = [flat.compress(f, method=method) for f in grid]
        axs[-1].plot([100 * c.ratio for c in curve], [c.error / spacing for c in curve],
                     label=method)
    axs[-1].set_xlabel("storage saved (%)")
    axs[-1].set_ylabel("RMS error (point spacings)")
    axs[-1].set_yscale("log")
    axs[-1].set_xlim(50, 100)
    axs[-1].grid(alpha=0.3)
    axs[-1].legend()

    # A real file: the points in JPEG2000 (any JPEG2000 reader opens it), and back
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{model}.jp2"
        size = comp.save(path)
        print("file     ", f"{path.name}: {size} bytes ({8 * size / len(points):.1f} bits per point)")
        print("read     ", Compressor.read(path).shape)  # (N, 3)

    _finish(fig, save and Path(save) / "compression", f"{model}.png", show, dpi=100)
    return comp


def _finish(fig, folder, name, show, **kwargs):
    """Save `fig` as folder/name if `folder`, then show it if `show`."""
    import matplotlib.pyplot as plt

    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(folder) / name, **kwargs)
        print("saved    ", Path(folder) / name)
    if show:
        plt.show()


__all__ = ["exemple_flat", "exemple_sampler", "exemple_compression"]
