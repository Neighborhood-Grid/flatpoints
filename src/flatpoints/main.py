"""User-facing classes: `Flat` (point cloud -> images), `Resampler` (images -> new points
on the surface) and `Compressor` (image compression of the points, see compressor.py).
`Viz` adds the plots to `Flat`.
"""
from functools import cached_property

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from .core import conformal_uv, image_cells
from .mesh import TRIM, closest_on_mesh, disk_cut, given_mesh, hemisphere_cut, poisson_mesh, submesh
from .resampler import lloyd_ot
from .utils import axes, fill_empty, Chart, Corners, Mesh, _window, _seeds, ON_SURFACE


class Viz:
    """Plots of a `Flat`, as its methods: `flat.plot()` and `flat.imshow(data)`.

    Needs matplotlib (`pip install "flatpoints[viz]"`), imported only when a plot is drawn.
    The drawing itself is in `viz.py`."""

    def plot(self, squares=8, point_size=1.0):
        """The surface in 3D and its images, coloured by RGB = xyz and by a checkerboard
        of `squares` pixels: smooth colours and regular squares mean a good flattening.
        """
        from .viz import plot
        return plot(self, squares, point_size)

    def imshow(self, data=None, sigma=0.5, cmap="viridis", animate=False, resolution="low"):
        """`data` on the images: per-point (N, *C) or images (n_faces, n, n, *C),
        default the xyz of the points. See `viz.imshow` for the colouring, `viz.morph` for
        `animate` (True, or a .webp path) and `resolution` ("low" or "high").
        """
        from .viz import imshow, morph
        data = self.points if data is None else np.asarray(data)
        if len(data) != len(self.points):
            if data.shape[:3] != (self.n_faces, self.n, self.n):
                raise ValueError(f"data must be per point ({len(self.points)}, *C) or images "
                                 f"({self.n_faces}, {self.n}, {self.n}, *C), got {data.shape}")
        else:
            data = self.to_image(data)
        if animate is not False:
            return morph(data, self.to_image(self.points), self.normal_axis, sigma, cmap,
                         path=None if animate is True else animate, resolution=resolution)
        return imshow(data, sigma, cmap)


class Flat(Viz):
    """Flatten a 3D point cloud into one or two square images, one pixel per point.

    Neighbour pixels hold neighbour points, so per-point data (N, *C) becomes images
    (n_faces, n, n, *C). See Method.md for how.

    Args:
        points: (N, 3) points sampled on a surface.
        mesh: optional mesh of the surface, used instead of the Poisson reconstruction:
            (vertices (V, 3), triangles (F, 3)), or any object with `vertices` and
            `faces` (or `triangles`), e.g. trimesh, open3d. The points are projected on it.
        single_face: True for an open surface (one image), False for a closed one (two
            images, the surface cut in left and right halves).
        normal_axis: "x", "y" or "z". Orients the images (rows follow z, or y if
            normal_axis is "z") and, for a closed surface, sets the direction of the cut.
        verbose: print progress.

    Attributes:
        n, n_faces: side and number of the images.
        pixels: (N, 3) (face, row, col) of each point.
        faces: {"left": idx, "right": idx}, None if single_face.
        corners: the points copied into the empty pixels (see `to_image`).
        _mesh, _charts, _uv_conformal: intermediate steps, for inspection.

    Example:
        flat = Flat(points)
        image = flat.to_image(points)       # (2, n, n, 3)
        face, row, col = flat.pixels[42]
        new = flat.resample(512).points     # (512, 3)
    """

    def __init__(self, points, single_face=False, normal_axis="x", verbose=False, mesh=None):
        self.points = P = np.asarray(points, float)
        self.normal_axis, self.single_face = normal_axis, single_face
        normal, horiz, vert = axes(normal_axis)

        # 1. Mesh, cut into charts
        if mesh is not None:
            V, F = given_mesh(mesh)
        else:
            V, F = poisson_mesh(P, trim=TRIM if single_face else None)
        if single_face:
            side = disk_cut(V, F)
        else:
            side = hemisphere_cut(V, F, normal, np.median(P[:, normal]))
        self._mesh = Mesh(V, F, side)
        if verbose:
            print(f"mesh: {len(V)} vertices, {len(F)} triangles, "
                  f"{(side < 0).sum()} in no face")
        in_face = np.flatnonzero(side >= 0)
        tri, bary = closest_on_mesh(V, F[in_face], P)  # each point projected on the mesh
        tri = in_face[tri]
        face_of = side[tri]

        names = ("main",) if single_face else ("left", "right")
        self.n_faces = len(names)
        self.n = n = int(np.ceil(np.sqrt(np.bincount(face_of).max())))
        self.faces = None if single_face else {}
        self._charts = []
        self._uv_conformal = np.zeros((len(P), 2))
        self.pixels = np.zeros((len(P), 3), int)
        pad_idxs, pad_pixels = [], []

        for f, name in enumerate(names):
            idx = np.flatnonzero(face_of == f)
            if verbose:
                print(f"face {name}: {len(idx)} points on {n} x {n}")
            if not single_face:
                self.faces[name] = idx

            # 2. Conformal uv of the chart, interpolated at the points
            mine = side == f
            Ff, vertices = submesh(F, mine)
            Vf = V[vertices]
            uv = conformal_uv(Vf, Ff, Vf[:, [horiz, vert]])
            self._charts.append(Chart(vertices, Ff, uv))
            around = Ff[(np.cumsum(mine) - 1)[tri[idx]]]  # chart vertices around each point
            self._uv_conformal[idx] = np.einsum("nk,nkd->nd", bary[idx], uv[around])

            # 3. Image, from the conformal uv
            cells = image_cells(self._uv_conformal[idx], n, verbose)  # (n, n), -1: empty
            rows, cols = np.nonzero(cells >= 0)
            self.pixels[idx[cells[rows, cols]]] = np.c_[np.full(len(rows), f), rows, cols]
            rows, cols = np.nonzero(cells < 0)
            pad_idxs.append(idx[fill_empty(cells)[rows, cols]])
            pad_pixels.append(np.c_[np.full(len(rows), f), rows, cols])

        self.corners = Corners(np.concatenate(pad_idxs), np.concatenate(pad_pixels))

    def resample(self, n_points=512, seed=42, sigma=0.0, method="seed", n_iter=10):
        """`n_points` new points evenly spread on the surface, see `Resampler`."""
        return Resampler(self, n_points, seed, sigma, method, n_iter)

    def compress(self, factor=0.8, method="auto"):
        """Compress the points, `factor` (e.g. 0.8) of the storage saved, with `method`
        ("auto": the most accurate), see `Compressor`."""
        return Compressor(self, factor, method)

    @cached_property
    def _lifting(self):
        """Cached for `Resampler`: xyz images, mask of real pixels, k-d tree and local
        spacing of the points.
        """
        image = self.to_image(self.points)  # (n_faces, n, n, 3)
        valid = np.ones(image.shape[:3], bool)
        valid[tuple(self.corners.pixels.T)] = False
        tree = cKDTree(self.points)
        spacing = tree.query(self.points, k=7)[0][:, 1:].mean(1)
        return image, valid, tree, spacing

    def to_image(self, values):
        """Per-point `values` (N, ...) as images (n_faces, n, n, ...), empty pixels
        filled from `corners`. E.g. `to_image(points)`: the xyz images.
        """
        values = np.asarray(values)
        image = np.empty((self.n_faces, self.n, self.n) + values.shape[1:], values.dtype)
        image[tuple(self.pixels.T)] = values
        image[tuple(self.corners.pixels.T)] = values[self.corners.idxs]
        return image


class Resampler:
    """`n_points` new points evenly spread on the surface of a `Flat`
    (`flat.resample(n_points)`).

    A 2D seed is placed on the images and lifted to 3D by bilinear interpolation of the xyz
    images; points that fall off the surface (across a tear) are replaced by the next ones.

    Args:
        n_points: split evenly between the faces.
        seed: int (shift of the default hyperuniform seed), (M, 2) points in [0, 1)^2, or
            {face name: (M, 2)}.
        sigma: Gaussian smoothing of the images before lifting, in pixels.
        method: "seed" (the seed lifted to 3D) or "ot" (then `n_iter` Lloyd iterations of
            optimal transport onto the input cloud, see `resampler.py`).

    Attributes:
        points: (n_points, 3); faces: (n_points,); pixels: (n_points, 3) float
        (face, row, col).
    """

    def __init__(self, flat, n_points=512, seed=42, sigma=0.0, method="seed", n_iter=10):
        self.flat, self.n_points, self.sigma = flat, n_points, sigma
        names = ("main",) if flat.single_face else ("left", "right")
        if n_points < len(names):
            raise ValueError(f"n_points must be at least {len(names)}, one per face")
        seeds = _seeds(seed, names)
        counts = np.full(len(names), n_points // len(names))
        if n_points % len(names):  # odd: one face, drawn from the seed, gets one more
            rng = np.random.default_rng(seed if isinstance(seed, (int, np.integer)) else 0)
            counts[rng.integers(len(names))] += 1

        image, valid, tree, spacing = flat._lifting
        m = flat.n
        points, pixels = [], []
        for f, (name, n) in enumerate(zip(names, counts)):
            smooth = gaussian_filter(image[f], (sigma, sigma, 0), mode="nearest")
            lift = lambda rc: np.stack([map_coordinates(smooth[..., c], rc.T, order=1,  # noqa: E731
                                                        mode="nearest") for c in range(3)], -1)

            def keep(rc, f=f, lift=lift):
                """Not on a padding pixel, and on the surface (not in the air)."""
                pixel = valid[f][tuple(np.clip(np.rint(rc).astype(int), 0, m - 1).T)]
                d, i = tree.query(lift(rc))
                return pixel & (d <= ON_SURFACE * spacing[i])

            rc = _window(seeds[name], n, m, keep)
            points.append(lift(rc))
            pixels.append(np.c_[np.full(n, f), rc])
        self.points = np.concatenate(points)
        self.pixels = np.concatenate(pixels)
        self.faces = self.pixels[:, 0].astype(int)
        if method == "ot":
            self.points, self.pixels = lloyd_ot(flat, self.points, self.pixels, n_iter)
        elif method != "seed":
            raise ValueError('method must be "seed" or "ot"')

    def plot(self, ax=None, point_size=1.0, **kwargs):
        """Before (input cloud, left) / after (new points on its mesh, right), in 3D.
        With `ax`, only the after view, drawn in `ax`. See `viz.plot_resampler`."""
        from .viz import plot_resampler
        return plot_resampler(self, ax=ax, point_size=point_size, **kwargs)



class Compressor:
    """Compress the points of a `Flat` through its xyz images (`flat.compress(factor)`).

    The images are smooth (neighbour pixels hold neighbour points), so image compression
    applies. Methods (see `compressor.py`): "svd" (truncated SVD), "dct" (quantized DCT,
    as JPEG) and "wavelet" (quantized wavelets, as JPEG2000, needs PyWavelets); "auto"
    runs the available ones and keeps the most accurate.

    Args:
        factor: fraction of the storage saved, in [0, 1), against the points in float32
            (96 bits per point): 0.8 keeps 20%, about 19 bits per point.
        method: "auto", "svd", "dct" or "wavelet".

    Attributes:
        method: the method used.
        code: what is stored, arrays (n_faces, 3, ...): {"U", "S", "Vt"} for svd,
            {"coefficients", "step", ...} for dct and wavelet (integers, times step).
        bits: size of the code; ratio: storage actually saved, 1 - bits / (96 N).
        images: (n_faces, n, n, 3) the reconstructed xyz images.
        points: (N, 3) the reconstructed points, in the order of `flat.points`.
        error: RMS distance between the points and their reconstruction.

    `bits` is a theoretical bound for dct and wavelet (ideal entropy coder). `save` writes a
    real file, in JPEG2000, and `Compressor.read` reads it back.
    """

    def __init__(self, flat, factor=0.8, method="auto"):
        from . import compressor
        if not 0 <= factor < 1:
            raise ValueError("factor must be in [0, 1), e.g. 0.8 to save 80% of the storage")
        methods = compressor.available() if method == "auto" else [method]
        if method != "auto" and method not in compressor.METHODS:
            raise ValueError(f"method must be 'auto' or one of {list(compressor.METHODS)}")
        self.flat, self.factor = flat, factor
        raw = 32 * flat.points.size
        A = np.moveaxis(flat.to_image(flat.points), -1, 1)  # (n_faces, 3, n, n)
        shape = A.shape
        best = None
        for name in methods:
            code, rec, bits = compressor.METHODS[name](A.reshape(-1, *shape[2:]), (1 - factor) * raw)
            images = np.moveaxis(rec.reshape(shape), 1, -1)
            points = images[tuple(flat.pixels.T)]
            error = np.sqrt(((points - flat.points) ** 2).sum(1).mean())
            code = {k: v.reshape(shape[:2] + v.shape[1:]) if np.ndim(v) else v
                    for k, v in code.items()}  # per face and coordinate
            if best is None or error < best[-1]:
                best = (name, code, bits, images, points, error)
        self.method, self.code, self.bits, self.images, self.points, self.error = best
        self.ratio = 1 - self.bits / raw

    def save(self, path):
        """Write the points to `path` as a JPEG2000 file (16 bits, any JPEG2000 reader opens
        it), within the size budget of `factor`. The file holds the points, not their order.
        Returns its size in bytes."""
        from .compressor import to_jp2
        flat = self.flat
        n = flat.n
        padding = (flat.corners.pixels * [n * n, n, 1]).sum(1)
        data = to_jp2(flat.to_image(flat.points), padding, (1 - self.factor) * 32 * flat.points.size)
        with open(path, "wb") as file:
            file.write(data)
        return len(data)

    @staticmethod
    def read(path):
        """(N, 3) points of a file written by `save`, in the order of its pixels."""
        from .compressor import from_jp2
        with open(path, "rb") as file:
            return from_jp2(file.read())
