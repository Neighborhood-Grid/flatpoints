# Method

flatpoints takes a point cloud sampled on a surface and returns a mesh, two uv maps and
one or two square images. The surface is cut once; then two independent pipelines run
on each face.

```
points ──► mesh + cut ──┬──► naive grid (temporary) ──► geodesics ──► Isomap ──► uv_geodesic_shortcut
                        │
                        └──► rectangular conformal map ──► uv_conformal ──► gridpoints ──► image
```

## 1. Mesh and cut

- **Mesh.** A closed surface is reconstructed by screened Poisson (open3d) from the cloud
  and its estimated normals. An open surface (`single_face=True`, e.g. a terrain) is
  triangulated by a 2D Delaunay of the points in their two non-lateral coordinates,
  without the long thin triangles along the convex hull.
- **Cut.** The mesh is split in two faces, left and right of the median of the lateral
  axis. Each face grows triangle by triangle from its extreme triangle, and accepts a
  triangle only if it stays a topological disk (one connected piece, one boundary loop,
  no pinched vertex). This works whatever the genus of the mesh: the few triangles of a
  handle that would close a loop stay out of both faces.
- **Points.** Each point is projected on the mesh; it belongs to the face of the
  triangle it falls in. Both pipelines below use this same split.

## 2. Geodesic uv with shortcuts (`uv_geodesic_shortcut`)

Per face, on the points only (the mesh is not used):

1. **Temporary naive grid.** The points are sorted on an n x n grid by their two
   non-lateral coordinates (gridpoints), i.e. by their position in the side view. This
   grid only serves to build a neighbourhood graph and is then discarded.
2. **Geodesics.** Each cell is linked to the cells at most 4 rows and columns away, by
   an edge of 3D length x weighted x + 0.1 x²/h (h: typical edge length). Distances from
   13 landmarks (corners, edge middles, centre, quarter centres of the grid) are shortest
   paths in this graph.
3. **Flattening.** Landmark Isomap: the landmarks are placed by classical MDS of their
   pairwise distances, every point by triangulation from its distances to them. The
   result is rotated so that u follows the horizontal axis and v the up axis.

**Shortcuts.** Two points close in the side view are neighbours on the naive grid even
when they are far apart on the surface (e.g. two coils of the dragon's body). The penalty
on long edges limits these jumps but does not remove them: on the dragon, about 20% of
the distances are more than 20% shorter than the true geodesics (on the bunny, cow:
~0%).

These shortcuts are what keeps the uv recognisable. A curved surface has no flattening
that preserves distances (Gauss: an isometry preserves the Gaussian curvature, and the
plane has none). With the true geodesics, Isomap — a linear projection of the distance
matrix onto its two main axes — folds the surface onto itself where it is far from flat
(up to 28% of flipped triangles on a face of the dragon). The shortcuts pull the
distances towards those of the side view, which is a valid planar configuration: the uv
keeps the shape of the object, at the price of some topological truth — parts that
overlap in the side view may be glued together in the uv.

## 3. Conformal uv and image (`uv_conformal`, `to_image`)

Per face, on the mesh:

1. **Rectangular conformal map** (Meng, Choi & Lui 2016, port of
   [garyptchoi/rectangular-conformal-map](https://github.com/garyptchoi/rectangular-conformal-map)):
   - harmonic map of the face onto the unit disk (cotangent Laplacian, boundary placed
     by arc length; Tutte map if the harmonic map folds);
   - disk to unit square: generalised Laplace equation from the Beltrami coefficient of
     the disk map, with the four corners of the square sent to the boundary vertices
     closest to the corners of the face's silhouette (so u is horizontal, v points up).

   The map preserves angles and is a bijection: no overlap. It does not preserve areas:
   protrusions (ears, legs) are strongly compressed.
2. **uv of the points.** Each point takes the uv of its projection on the mesh,
   interpolated with the barycentric coordinates of its triangle.
3. **Image.** The points are sorted on an n x n grid by their conformal uv (gridpoints).
   One point per pixel: since the points are uniform on the surface, every pixel covers
   about the same area, whatever the area distortion of the conformal map. Empty pixels
   (n² is a little more than the number of points) are padded with the point of the
   nearest image corner.

Because the conformal uv never overlaps, neighbouring pixels hold neighbouring points
much more often than with an image sorted by the geodesic uv: on the bunny, 2.8% of
adjacent pixels jump more than 3 point spacings in 3D, against 6.7%.

## Reference

T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal
Mappings of Point Clouds", SIAM Journal on Imaging Sciences, 9(4), 1922–1962, 2016.
