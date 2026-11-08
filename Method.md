# Method

flatpoints takes a point cloud sampled on a surface and returns one or two square images: each point of the cloud on its own pixel, neighbour pixels holding neighbour points.

```
points ──► mesh, cut into disks ──► conformal map onto a square ──► sort onto the grid ──► image
```

This is the *geometry image* of Gu, Gortler & Hoppe (2002), with one difference: no
resampling. The pixels are the input points themselves, so `flat.pixels` gives the exact
point <-> pixel correspondence.

## 1. Mesh and cut

A square is a topological disk, so the surface must first be cut into disks, the faces.

- **Mesh**: mesh the point cloud using screened Poisson reconstruction (open3d) 
- **Cut**: a closed surface is split into two faces, left and right of the median along
  `normal_axis`; an open surface is a single face. Faces grow triangle by triangle and
  only accept triangles that keep them a disk, so handles and holes are left cut open.
- **Points**: each point is projected onto the mesh and takes the face of its triangle.

## 2. Conformal map

Each face is mapped onto the unit square by a rectangular conformal map (Meng, Choi & Lui
2016). Each point takes the uv_conformal of its projection on the mesh.

## 3. Grid

The points of each face are sorted onto an n x n grid by their uv_conformal (gridpoints). The resulting one to one matching
```
point -> pixel = face, row, column
```
is the final output of the flatpoints. Due to the one to one assignement, the flattening is by construction an area preserving transformation.

## Limits

- **Cuts**: points on either side of a cut are neighbours on the surface but not in the images. This apply to the left / right cut of the single_face = False pipeline, but also on further topological closedness of the surface (e.g. for a torus), which will also be cut.
- **Tears**: the grid cannot follow the uv exactly, so a few adjacent pixels hold points
  several spacings apart (2.7% beyond 3 spacings on the bunny).
- **Meshing**: since flatpoints operate on point cloud with unkown surface, the overal precision of the flattening will depend on the quality of the meshing performed by open3d.

## References

X. Gu, S. J. Gortler and H. Hoppe, "Geometry Images", SIGGRAPH 2002.
[hhoppe.com/proj/gim](https://hhoppe.com/proj/gim/)

T. W. Meng, G. P.-T. Choi and L. M. Lui, "TEMPO: Feature-Endowed Teichmüller Extremal
Mappings of Point Clouds", SIAM J. Imaging Sciences, 9(4), 2016.
