"""User-facing classes of flatpoints: `Flat` and `Resampler`.

`Flat` takes points sampled on a surface and build 2 related "flattening" operations:
- isomaps: 2D parametrisation of the 3D surface, e.g. smooth flattening xyz -> uv
- images: multi index assignement of the points, raw linear index to pixel layout, e.g. idx -> (row, column)

`Resampler` uses the images of a `Flat` to transport a 2D point set (by default a
hyperuniform seed) onto the surface: (row, col) -> xyz.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .core import layout
from .utils import axes, fill_empty, split, Corners



class Flat:
    """Map a 3D point cloud onto one or two 2D grids.

    Two mappings are computed:
    - `uv`: continuous 2D coordinates for each point, defining its surface
      parametrisation.
    - `pixels`: discrete (face, row, col) coordinates, assigning each point
      to a unique grid cell.

    Args:
        points: (N, 3) array of 3D coordinates.
        single_face: If True, use a single grid. Otherwise, split the surface
            into left and right faces.
        lateral: Axis along which to split the surface ("x", "y" or "z").
        up: Vertical axis defining the v direction ("x", "y" or "z").
        verbose: Print progress information.

    Attributes:
        n: Side length of each square grid.
        n_faces: Number of grids (1 or 2).
        uv: (N, 2) continuous coordinates in [0, 1].
        pixels: (N, 3) integer coordinates (face, row, col) of each point.
        faces: Dictionary mapping face names to point indices, or None if
            single_face=True.
        corners: Padding information for empty grid cells. Contains:
            - idxs: Point indices to copy.
            - pixels: Destination (face, row, col) coordinates.

     Example: flatten raw (N, 3) point cloud
        flat = Flat(points)

        # Continuous surface coordinates
        uv = flat.uv  # (N, 2)

        # Discrete grid coordinates
        face, row, col = flat.pixels[42]

        # Convert any per-point data into images
        image = flat.to_image(data)  # (2, n, n, ...)
    """

    def __init__(self, points, single_face=False, lateral="x", up="z", verbose=False):
        self.points = np.asarray(points, float)
        if up == lateral:
            up = "y" if lateral == "z" else "z"
        self.lateral, self.up, self.single_face = lateral, up, single_face

        face_of, ab = split(self.points, single_face, *axes(lateral, up))
        names = ("main",) if single_face else ("left", "right")
        self.n_faces = len(names)
        self.n = n = int(np.ceil(np.sqrt(np.bincount(face_of).max())))

        N = len(self.points)
        self.uv = np.zeros((N, 2))
        self.pixels = np.zeros((N, 3), int)
        self.faces = None if single_face else {}
        corner_idxs, corner_pixels = [], []

        for f, name in enumerate(names):
            idx = np.flatnonzero(face_of == f)  # points of this face
            if verbose:
                print(f"face {name}: {len(idx)} points on {n} x {n}")
            cells, uv = layout(self.points[idx], ab[idx], n, verbose)  # (n, n), -1 = empty
            self.uv[idx] = uv
            if not single_face:
                self.faces[name] = idx

            rows, cols = np.nonzero(cells >= 0)
            self.pixels[idx[cells[rows, cols]]] = np.c_[np.full(len(rows), f), rows, cols]

            rows, cols = np.nonzero(cells < 0)
            corner_idxs.append(idx[fill_empty(cells)[rows, cols]])
            corner_pixels.append(np.c_[np.full(len(rows), f), rows, cols])

        self.corners = Corners(np.concatenate(corner_idxs), np.concatenate(corner_pixels))

    def to_image(self, values):
        """Place per-point `values` (N, ...) on the grids, empty pixels filled from the
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


SEED = Path(__file__).resolve().parents[2] / "data" / "seed.npy"  # (2**17, 2) hyperuniform


class Resampler:
    """Resample a surface: `n` new points per face, transported from a 2D point set.

    1. `Flat(points)`: each face is an (n_grid, n_grid) image, pixel -> xyz.
    2. A square window of the 2D `seed` (periodic on [0, 1)^2) holding `n` points, rescaled
       to the face's image: (row, col) in [0, n_grid - 1]^2. Points whose nearest pixel is
       a padding pixel (outside the surface, see `Flat.corners`) are skipped; the window
       grows until it holds `n` points on the surface.
    3. xyz of each point: bilinear interpolation of the image, smoothed by a Gaussian of
       `sigma` pixels.

    With the default seed, a stealthy hyperuniform point set of 2**17 points (data/seed.npy),
    the new points are evenly spread over the surface (blue noise), up to the area distortion
    of the flattening. Any (M, 2) array in [0, 1)^2 can be used instead, e.g. i.i.d. uniform
    points for comparison.

    Args:
        points: (N, 3) point cloud sampled on the surface.
        n: number of new points per face.
        seed: (M, 2) points in [0, 1)^2, or path to a .npy file; default: data/seed.npy.
        offset: (2,) corner of the window of the first face, in [0, 1)^2. Face f uses
            offset + f * (0.5, 0.5): the faces get disjoint windows of the seed.
        sigma: Gaussian smoothing of the images, in pixels (0: none).
        single_face, lateral, up, verbose: passed to `Flat`.

    Attributes:
        flat: the `Flat` of the input points.
        points: (n_faces * n, 3) the new points, face by face.
        faces: (n_faces * n,) face of each new point (0 = left or the only face, 1 = right).
        pixels: (n_faces * n, 3) float (face, row, col) of each new point on the images.
        windows: [(offset, side)] the window of the seed used for each face.

    Example:
        sampler = Resampler(points, n=256)
        new = sampler.points  # (512, 3)
        sampler.plot()
    """

    def __init__(self, points, n=256, seed=None, offset=(0.0, 0.0), sigma=0.25,
                 single_face=False, lateral="x", up="z", verbose=False):
        self.flat = flat = Flat(points, single_face, lateral, up, verbose)
        if seed is None or isinstance(seed, (str, Path)):
            seed = np.load(SEED if seed is None else seed)
        seed = np.asarray(seed, float) % 1.0
        self.n, self.sigma = n, sigma

        image = flat.to_image(flat.points)  # (n_faces, n_grid, n_grid, 3)
        valid = np.ones(image.shape[:3], bool)  # False on the padding pixels
        valid[tuple(flat.corners.pixels.T)] = False
        points, pixels, self.windows = [], [], []
        for f in range(flat.n_faces):
            o = (np.asarray(offset, float) + 0.5 * f) % 1.0
            rc, side = _window(seed, n, o, valid[f])
            smooth = gaussian_filter(image[f], (sigma, sigma, 0), mode="nearest")
            points.append(np.stack([map_coordinates(smooth[..., c], rc.T, order=1,
                                                    mode="nearest") for c in range(3)], -1))
            pixels.append(np.c_[np.full(n, f), rc])
            self.windows.append((o, side))
        self.points = np.concatenate(points)
        self.pixels = np.concatenate(pixels)
        self.faces = self.pixels[:, 0].astype(int)

    def plot(self, ax=None, point_size=1.0, **kwargs):
        """3D view, RGB colours: the new points (discs) on the mesh of the full point cloud
        (quads of `flat`). Draws in `ax` if given. See `viz.plot_resampler`."""
        from .viz import plot_resampler
        return plot_resampler(self, ax=ax, point_size=point_size, **kwargs)


def _window(seed, n, offset, valid):
    """The `n` points of `seed` closest to `offset` in the square (Chebyshev, towards +x,
    +y on the torus) whose nearest pixel is `valid`, as (row, col) on the (m, m) grid
    `valid`: the square [offset, offset + side)^2 is rescaled to [0, m - 1]^2.

    Returns:
        rc: (n, 2) row, col; side: side of the window in the seed.
    """
    x = (seed - offset) % 1.0
    order = np.argsort(x.max(1))
    x, d = x[order], x.max(1)[order]
    m, k = len(valid), n
    while True:
        if k > len(x):
            raise ValueError(f"the seed is too small for {n} points on this surface")
        side = (d[k - 1] + d[k]) / 2 if k < len(x) else 1.0
        rc = x[:k] / side * (m - 1)
        ok = valid[tuple(np.clip(np.rint(rc).astype(int), 0, m - 1).T)]
        if ok.sum() >= n:
            return rc[ok][:n], side
        k += n - ok.sum()
