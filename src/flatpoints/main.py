"""User-facing class of flatpoints: `Flat`.

`Flat` takes points sampled on a surface and build 2 related "flattening" operations:
- isomaps: 2D parametrisation of the 3D surface, e.g. smooth flattening xyz -> uv
- images: multi index assignement of the points, raw linear index to pixel layout, e.g. idx -> (row, column)
"""

import numpy as np

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
