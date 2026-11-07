# Method

flatpoints takes a point cloud sampled on a surface and returns one or two square
images: each point of the cloud on its own pixel, neighbour pixels holding neighbour
points.

```
points ──► mesh + cut (open3D) ──► rectangular conformal map (garyptchoi)  ──► gridification (gridpoints) ──► image
```

## 0. The image

An image of the cloud is an assignment of its N points to the pixels of an n x n grid,
idx -> (row, column), with:

- **one point per pixel.** n is the smallest side with n² ≥ the number of points of the
  largest face, so each image holds every point of its face exactly once, and only a few
  pixels are left over;
- **neighbour pixels hold neighbour points.** Moving by one pixel moves by about one
  point spacing on the surface, in every direction and everywhere on the surface.

Any per-point feature (N, *C) is then an image (n, n, *C): 

```python
points = np.load("data/bunny_points.npy") #(N, 3)
feature = points[:, -1] #(N, *C) any feature, eg the z coordinate
feature_image = flatpoints.Flat(points).to_image(feature) #(2, n, n, *C), 2 square images = left and right faces
```

This is the *geometry image* of Gu, Gortler & Hoppe (2002): cut the surface into a
topological disk, map it onto a square, and store the geometry as a regular 2D array,
so that image tools apply directly to the surface. The main difference: a geometry image 
resamples a mesh on the
regular grid, while flatpoints starts from a point cloud and does not resample, the
pixels are the input points themselves, sorted on the grid. Nothing is interpolated, and
`pixels` gives the exact correspondence point <-> pixel. See section 4 for the full
comparison.

## 1. Mesh and cut

A square is a disk, so the surface must first be cut into topological disks, the faces.

- **Mesh.** A closed surface is reconstructed by screened Poisson (open3d) from the cloud
  and its estimated normals. An open surface (`single_face=True`, e.g. a terrain) is
  triangulated by a 2D Delaunay of the points in their two coordinates other than the
  normal axis, without the long thin triangles along the convex hull.
- **Cut.** A closed mesh is split in two faces, left and right of the median of the
  normal axis (`normal_axis`). Each face grows triangle by triangle from its extreme
  triangle, and accepts a triangle only if it stays a topological disk (one connected
  piece, one boundary loop, no pinched vertex). This works whatever the genus of the
  mesh: the few triangles of a handle that would close a loop stay out of both faces.
- **Points.** Each point is projected on the mesh; it belongs to the face of the
  triangle it falls in. Each face gets its own image.

The mesh is only a scaffold for the next step (`flat._mesh`, `flat._charts`).

## 2. Conformal map of each face

Per face, on the mesh:

1. **Rectangular conformal map** (Meng, Choi & Lui 2016, port of
   [garyptchoi/rectangular-conformal-map](https://github.com/garyptchoi/rectangular-conformal-map)):
   - harmonic map of the face onto the unit disk (cotangent Laplacian, boundary placed
     by arc length; Tutte map if the harmonic map folds);
   - disk to unit square: generalised Laplace equation from the Beltrami coefficient of
     the disk map, with the four corners of the square sent to the boundary vertices
     closest to the corners of the face's silhouette (so u is horizontal, v points up).

   The map preserves angles and is a bijection: no overlap, so points far apart on the
   surface stay far apart in the square. It does not preserve areas: protrusions (ears,
   legs) are strongly compressed.
2. **uv of the points.** Each point takes the uv of its projection on the mesh,
   interpolated with the barycentric coordinates of its triangle (`flat._uv_conformal`).

## 3. Image of each face

The points are sorted on the n x n grid by their conformal uv (gridpoints): close uv go
to close pixels, each pixel getting exactly one point. Rows follow v (upwards), columns
follow u.

This step also undoes the area distortion of the conformal map: in the uv, the points of
a compressed ear are packed in a small region, but on the grid each of them still gets a
full pixel. Since the points are uniform on the surface, every pixel covers about the
same area: the image is close to area preserving, while keeping the neighbourhoods of the
conformal map.

Empty pixels (n² is a little more than the number of points) are padded with the point of
the nearest image corner (`flat.corners`).

## Limits

- **Cuts.** Neighbour pixels hold neighbour points, but not the reverse: points on both
  sides of a cut (the seam between the two faces, or a handle left out of both) are close
  on the surface and far apart, or on different images.
- **Tears.** Where the sorting cannot follow the uv exactly, two adjacent pixels may hold
  points a few spacings apart (on the bunny, 2.8% of adjacent pixels jump more than 3
  point spacings in 3D).

## References

X. Gu, S. J. Gortler and H. Hoppe, "Geometry Images", ACM Transactions on Graphics
(SIGGRAPH), 21(3), 355–361, 2002. [hhoppe.com/proj/gim](https://hhoppe.com/proj/gim/)

T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal
Mappings of Point Clouds", SIAM Journal on Imaging Sciences, 9(4), 1922–1962, 2016.
