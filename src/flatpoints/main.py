"""User-facing classes of flatpoints: `Flat` and `Resampler`.

`Flat` takes points sampled on a surface and build 2 related "flattening" operations:
- isomaps: 2D parametrisation of the 3D surface, e.g. smooth flattening xyz -> uv
- images: multi index assignement of the points, raw linear index to pixel layout, e.g. idx -> (row, column)

`Resampler` uses the images of a `Flat` to transport a 2D point set (by default a
hyperuniform seed) onto the surface: (row, col) -> xyz.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .core import layout
from .utils import axes, fill_empty, split, Corners, _window, _seeds



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


class Resampler:
    """Resample a surface: `n` new points per face, evenly spread (hyperuniform).

    The points of a 2D seed in the unit square are placed on the images of `Flat(points)`
    and lifted to the surface (row, col -> xyz).

    Args:
        points: (N, 3) point cloud sampled on the surface.
        n: number of new points per face.
        seed: one of
            - int (default 42): the hyperuniform seed data/seed.npy, randomly shifted;
              another integer gives an independent resampling;
            - (M, 2) points in [0, 1)^2, used for every face;
            - {"left": (M, 2), "right": (M, 2)}: points for each face
        sigma: Gaussian smoothing of the images before lifting, in pixels.
        single_face, lateral, up, verbose: passed to `Flat`.

    Attributes:
        flat: the `Flat` of `points`.
        points: (n_faces * n, 3) the new points, face by face.
        faces: (n_faces * n,) face of each new point.
        pixels: (n_faces * n, 3) (face, row, col) of each new point, row and col as floats.

    Example:
        sampler = Resampler(points, n=256)
        new = sampler.points  # (512, 3)
        sampler.plot()
    """

    def __init__(self, points, n=256, seed=42, sigma=0.25,
                 single_face=False, lateral="x", up="z", verbose=False):
        self.flat = flat = Flat(points, single_face, lateral, up, verbose)
        self.n, self.sigma = n, sigma
        names = ("main",) if single_face else ("left", "right")
        seeds = _seeds(seed, names)

        image = flat.to_image(flat.points)  # (n_faces, n_grid, n_grid, 3)
        valid = np.ones(image.shape[:3], bool)  # False on the padding pixels
        valid[tuple(flat.corners.pixels.T)] = False
        points, pixels = [], []
        for f, name in enumerate(names):
            rc, _ = _window(seeds[name], n, np.zeros(2), valid[f])
            smooth = gaussian_filter(image[f], (sigma, sigma, 0), mode="nearest")
            points.append(np.stack([map_coordinates(smooth[..., c], rc.T, order=1,
                                                    mode="nearest") for c in range(3)], -1))
            pixels.append(np.c_[np.full(n, f), rc])
        self.points = np.concatenate(points)
        self.pixels = np.concatenate(pixels)
        self.faces = self.pixels[:, 0].astype(int)

    def plot(self, ax=None, point_size=1.0, **kwargs):
        """3D view, RGB colours: the new points (discs) on the mesh of the full point cloud
        (quads of `flat`). Draws in `ax` if given. See `viz.plot_resampler`."""
        from .viz import plot_resampler
        return plot_resampler(self, ax=ax, point_size=point_size, **kwargs)
