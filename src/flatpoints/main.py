"""User-facing classes of flatpoints: `Flat` and `Resampler`.

`Flat` takes points sampled on a surface and builds:
- a mesh of the surface, cut into one or two charts (topological disks);
- uv: 2D coordinates of the points, xyz -> uv, conformal (angle preserving, no overlap)
  or geodesic with shortcuts (close to distances, recognisable shape);
- images: a pixel for each point, idx -> (face, row, column), neighbour pixels holding
  neighbour points.

`Flat.resample` (a `Resampler`) uses the images of the flat to transport a 2D point set
(by default a hyperuniform seed) onto the surface: (row, col) -> xyz.

`Flat.plot` and `Flat.imshow` (from the `Viz` mixin) draw the flat, with matplotlib.
"""
from functools import cached_property

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from .core import conformal_uv, geodesic_shortcut_uv, image_cells
from .mesh import closest_on_mesh, hemisphere_cut, planar_mesh, poisson_mesh, submesh
from .utils import axes, fill_empty, Chart, Corners, Mesh, _window, _seeds, ON_SURFACE


class Viz:
    """Plots of a `Flat`, as its methods: `flat.plot()` and `flat.imshow(data)`.

    Needs matplotlib (`pip install -e ".[viz]"`), imported only when a plot is drawn.
    The drawing itself is in `viz.py`."""

    def plot(self, uv="geodesic_shortcut", squares=8, point_size=1.0):
        """The flattening at a glance. First column: the faces in 3D; then one column per
        face in (u, v), `uv` being "geodesic_shortcut" (`uv_geodesic_shortcut`) or
        "conformal" (`uv_conformal`). First row coloured by RGB = normalised x, y, z,
        second row by a checkerboard of `squares` x `squares` pixels: a good flattening
        gives smooth colours and regular squares. `point_size` scales the points.
        Returns the figure."""
        from .viz import plot
        return plot(self, uv, squares, point_size)

    def imshow(self, data=None, sigma=0.5, cmap="viridis"):
        """`data` on the images of the flat, one image per face. `data`: per-point values
        (N, *C), placed by `to_image`, or images already (n_faces, n, n, *C); default:
        the points, RGB = x, y, z. Smoothed by a Gaussian of `sigma` pixels, coloured by
        the number of channels C = prod(*C): 1 colormap `cmap`, 2 bivariate colormap,
        3 RGB, > 3 RGB of the 3 first principal components (see `viz.imshow`).
        Returns the figure."""
        from .viz import imshow
        data = self.points if data is None else np.asarray(data)
        if len(data) != len(self.points):
            if data.shape[:3] != (self.n_faces, self.n, self.n):
                raise ValueError(f"data must be per point ({len(self.points)}, *C) or images "
                                 f"({self.n_faces}, {self.n}, {self.n}, *C), got {data.shape}")
        else:
            data = self.to_image(data)
        return imshow(data, sigma, cmap)


class Flat(Viz):
    """Flatten a 3D point cloud: a mesh, two uv maps and one or two square images.

    Pipeline (see Method.md): the surface is meshed (Poisson reconstruction, or 2D
    Delaunay for an open surface) and cut into one or two topological disks, the faces
    (left and right of the median of the normal axis). Each point takes the face of its
    projection on the mesh. Then, per face, independently:
    - geodesic uv with shortcuts: temporary naive grid of the points (gridpoints on
      their two coordinates other than the normal axis), geodesics along it, landmark
      Isomap;
    - conformal uv: rectangular conformal map of the face's mesh, interpolated at the
      projections of the points; the image sorts the points by their conformal uv
      (gridpoints).

    Args:
        points: (N, 3) points sampled on a surface.
        single_face: one chart and one image for an open surface (e.g. a terrain),
            instead of two for a closed one.
        normal_axis: "x", "y" or "z", the axis the surface is seen along. A closed
            surface is cut across it, in two faces on either side of its median (left
            and right for "x"); an open surface is roughly normal to it (e.g. "z" for a
            terrain seen from above). v (the rows of the images, upwards) follows z, or y
            if normal_axis is "z"; u follows the third axis.
        verbose: print progress.

    Attributes:
        mesh: `Mesh` (vertices, triangles, face of each triangle) of the surface.
        charts: one `Chart` per face (vertices, triangles, uv_conformal of the vertices).
        uv_conformal: (N, 2) conformal uv of each point in [0, 1]^2, per face.
        uv_geodesic_shortcut: (N, 2) geodesic uv with shortcuts, in [0, 1]^2, per face.
        n: side of the images; n_faces: number of faces (1 or 2).
        normal_axis, single_face: the arguments.
        pixels: (N, 3) (face, row, col) of each point.
        faces: {"left": idx, "right": idx} points of each face, None if single_face.
        corners: padding of the empty pixels: copy points[corners.idxs] into
            corners.pixels (face, row, col).

    Methods:
        to_image(values): per-point values (N, ...) on the images, (n_faces, n, n, ...).
        resample(n_points): new points evenly spread on the surface, a `Resampler`.
        plot(uv): the surface in 3D and in (u, v) (from `Viz`, needs matplotlib).
        imshow(data): per-point data or images, one image per face (from `Viz`).

    Example:
        flat = Flat(points)
        image = flat.to_image(points)       # (2, n, n, 3)
        face, row, col = flat.pixels[42]
        uv = flat.uv_geodesic_shortcut      # (N, 2), or flat.uv_conformal
        V, T = flat.mesh.vertices, flat.mesh.triangles
        new = flat.resample(512).points     # (512, 3) new points on the surface
        flat.plot()                         # 3D and geodesic uv; flat.plot("conformal")
        flat.imshow()                       # the images, RGB = xyz
        flat.imshow(points[:, 2])           # any per-point data, here the height
    """

    def __init__(self, points, single_face=False, normal_axis="x", verbose=False):
        self.points = P = np.asarray(points, float)
        self.normal_axis, self.single_face = normal_axis, single_face
        normal, horiz, vert = axes(normal_axis)

        # 1. Mesh, cut into charts
        if single_face:
            V, F, side = planar_mesh(P, P[:, [horiz, vert]])
        else:
            V, F = poisson_mesh(P)
            side = hemisphere_cut(V, F, normal, np.median(P[:, normal]))
        self.mesh = Mesh(V, F, side)
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
        self.charts = []
        self.uv_conformal = np.zeros((len(P), 2))
        self.uv_geodesic_shortcut = np.zeros((len(P), 2))
        self.pixels = np.zeros((len(P), 3), int)
        pad_idxs, pad_pixels = [], []

        for f, name in enumerate(names):
            idx = np.flatnonzero(face_of == f)
            if verbose:
                print(f"face {name}: {len(idx)} points on {n} x {n}")
            if not single_face:
                self.faces[name] = idx

            # 2. Geodesic uv with shortcuts, from a temporary naive grid of the points
            ab = P[idx][:, [horiz, vert]]
            self.uv_geodesic_shortcut[idx] = geodesic_shortcut_uv(P[idx], ab, n, verbose)

            # 3. Conformal uv of the chart, interpolated at the points
            mine = side == f
            Ff, vertices = submesh(F, mine)
            Vf = V[vertices]
            uv = conformal_uv(Vf, Ff, Vf[:, [horiz, vert]])
            self.charts.append(Chart(vertices, Ff, uv))
            around = Ff[(np.cumsum(mine) - 1)[tri[idx]]]  # chart vertices around each point
            self.uv_conformal[idx] = np.einsum("nk,nkd->nd", bary[idx], uv[around])

            # 4. Image, from the conformal uv
            cells = image_cells(self.uv_conformal[idx], n, verbose)  # (n, n), -1: empty
            rows, cols = np.nonzero(cells >= 0)
            self.pixels[idx[cells[rows, cols]]] = np.c_[np.full(len(rows), f), rows, cols]
            rows, cols = np.nonzero(cells < 0)
            pad_idxs.append(idx[fill_empty(cells)[rows, cols]])
            pad_pixels.append(np.c_[np.full(len(rows), f), rows, cols])

        self.corners = Corners(np.concatenate(pad_idxs), np.concatenate(pad_pixels))

    def resample(self, n_points=512, seed=42, sigma=0.0):
        """`n_points` new points evenly spread on the surface (hyperuniform), see
        `Resampler`. Each call with another `seed` gives an independent resampling,
        without recomputing the flat."""
        return Resampler(self, n_points, seed, sigma)

    @cached_property
    def _lifting(self):
        """What every resampling of this flat needs, computed once: the xyz images, the
        mask of real (not padding) pixels, a k-d tree of the points and the local spacing
        of the cloud (mean distance of each point to its 6 nearest neighbours)."""
        image = self.to_image(self.points)  # (n_faces, n, n, 3)
        valid = np.ones(image.shape[:3], bool)
        valid[tuple(self.corners.pixels.T)] = False
        tree = cKDTree(self.points)
        spacing = tree.query(self.points, k=7)[0][:, 1:].mean(1)
        return image, valid, tree, spacing

    def to_image(self, values):
        """Place per-point `values` (N, ...) on the images, empty pixels filled from the
        corners: (n_faces, n, n, ...), image[0] being the left face (or the only one)
        and image[1] the right one.

        E.g. `to_image(points)` is the cloud as an xyz image, `to_image(np.arange(N))` the
        index of the point in each pixel.
        """
        values = np.asarray(values)
        image = np.empty((self.n_faces, self.n, self.n) + values.shape[1:], values.dtype)
        image[tuple(self.pixels.T)] = values
        image[tuple(self.corners.pixels.T)] = values[self.corners.idxs]
        return image


class Resampler:
    """Resample the surface of a `Flat`: `n_points` new points, evenly spread
    (hyperuniform). Build it with `flat.resample(n_points)`.

    The points of a 2D seed in the unit square are placed on the images of the flat and
    lifted to the surface (row, col -> xyz). Lifted points that fall off the surface
    (across a tear of the images) are skipped, and replaced by the next points of the seed.

    Args:
        flat: the `Flat` of the point cloud.
        n_points: total number of new points, split evenly between the faces; if it is
            odd, one face (drawn from `seed`) gets one more point.
        seed: one of
            - int (default 42): the hyperuniform seed data/seed.npy, randomly shifted;
              another integer gives an independent resampling;
            - (M, 2) points in [0, 1)^2, used for every face;
            - {"left": (M, 2), "right": (M, 2)}: points for each face.
        sigma: Gaussian smoothing of the images before lifting, in pixels.

    Attributes:
        flat: the `Flat`.
        points: (n_points, 3) the new points, face by face.
        faces: (n_points,) face of each new point.
        pixels: (n_points, 3) (face, row, col) of each new point, row and col as floats.

    Example:
        sampler = flat.resample(512)
        new = sampler.points  # (512, 3)
        sampler.plot()
    """

    def __init__(self, flat, n_points=512, seed=42, sigma=0.0):
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

    def plot(self, ax=None, point_size=1.0, **kwargs):
        """Before (input cloud, left) / after (new points on its mesh, right), in 3D.
        With `ax`, only the after view, drawn in `ax`. See `viz.plot_resampler`."""
        from .viz import plot_resampler
        return plot_resampler(self, ax=ax, point_size=point_size, **kwargs)
