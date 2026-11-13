# Method

flatpoints takes a point cloud sampled on a surface and returns square images:
each point of the cloud on its own pixel, neighbour pixels holding neighbour points.

This is the *geometry image* of Gu, Gortler & Hoppe (2002), with one difference: no
resampling. The pixels hold the input points themselves, so `flat.pixels` gives the exact
point <-> pixel correspondence.

The flattening pipeline: 

```
points ──► mesh, cut into disks topology ──► conformal map onto the square ──► grid assignment ──► image
```

The grid assignment (gridification) is the heart of the method. It matches the points to the
cells of a square grid, one to one (one point = one pixel), which gives a simple data
structure. By construction, it makes the final mapping roughly area-preserving: each pixel
covers the area of its point, the same for every point if the cloud is sampled uniformly.
Gridification is performed with the [gridpoints](https://pypi.org/project/gridpoints/) Python package.


## 1. Mesh and cut

A square is a topological disk, so the surface must first be cut into disks, the faces.

- **Mesh**: screened Poisson reconstruction (open3d) of the point cloud, from normals
  estimated and oriented on the points; only the largest connected piece is kept. Poisson
  always closes the surface, so for an open surface (`single_face=True`) the parts far
  from the points are trimmed.
- **Cut**: a closed surface is split into two faces, left and right of the median of the
  points along `normal_axis`; an open surface is a single face. Faces grow triangle by
  triangle and only accept triangles that keep them a disk, so handles and holes are cut
  open: their triangles belong to no face.
- **Points**: each point is projected onto the nearest triangle of a face, and takes that
  face.

## 2. Conformal map

Each face is mapped onto the unit square by a rectangular conformal map (Meng, Choi & Lui
2016). The four corners of the square go to the boundary vertices closest to the corners
of the face seen along `normal_axis`. Each point takes the conformal uv of its projection,
interpolated in its triangle.

## 3. Grid

The points of each face are assigned to the cells of an n x n grid, each point to its own
cell, by finding a spatially consistant assignement from the conformal uv coordinates of 
the points to cells of a grid, with one point matching one cell, up to a few padding cells
in the corners in case the number of points itself is not a square number, which are kept 
empty (because the grid has exactly n^2 cells) and filled  a posteriori with duplicate points.
The resulting one-to-one matching

```
point -> pixel = (face, row, column)
```

is the output of flatpoints.

## Limits

- **Cuts**: points on either side of a cut are neighbours on the surface but not in the
  images. This applies to the left / right cut of a closed surface, and to the extra cuts
  of handles and holes (e.g. a torus).
- **Tears**: the grid cannot follow the uv exactly, so a few adjacent pixels hold points
  several spacings apart (on the bunny, about 3% of the pairs of adjacent pixels are more
  than 3 local spacings apart, the local spacing being the mean distance of a point to its
  6 nearest neighbours).
- **Meshing**: the surface is unknown and reconstructed from the points, so the quality of
  the flattening depends on that of the Poisson mesh (sparse or noisy clouds, thin parts).

## References

X. Gu, S. J. Gortler and H. Hoppe, "Geometry Images", SIGGRAPH 2002.
[hhoppe.com/proj/gim](https://hhoppe.com/proj/gim/)

T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal
Mappings of Point Clouds", SIAM J. Imaging Sciences, 9(4), 2016.
