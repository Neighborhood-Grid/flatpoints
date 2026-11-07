# Point clouds

Each `*_points.npy` is a `(10000, 3)` array of points sampled uniformly (area-weighted) on the
surface of a classic 3D test mesh. `bunny_points.npy` is float64, the others float32.

```python
points = np.load("data/dragon_points.npy")
```

**These clouds are derived from third-party models and are not covered by the license of the
code.** Each one keeps the terms of its original model, listed below. Please credit the sources
if you use them.

| File | Model | Author / source | Terms |
|---|---|---|---|
| `spot_points.npy` | Spot | Keenan Crane, [Keenan's 3D Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/) | CC0 1.0 (public domain) |
| `bunny_points.npy` | Stanford Bunny | Stanford Computer Graphics Laboratory, [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) | Stanford terms (*) |
| `dragon_points.npy` | Stanford Dragon (original scan) | Stanford Computer Graphics Laboratory, same repository; mesh from [pyvista/vtk-data](https://github.com/pyvista/vtk-data) | Stanford terms (*) |
| `armadillo_points.npy` | Armadillo | Stanford Computer Graphics Laboratory, same repository | Stanford terms (*) |
| `xyzrgb_dragon_points.npy` | XYZ RGB Dragon | XYZ RGB Inc., distributed by the Stanford 3D Scanning Repository | Stanford terms (*) |
| `cow_points.npy` | Cow | Viewpoint Animation Engineering / Sun Microsystems | research and non-commercial use; copyright of the creator (**) |
| `horse_points.npy` | Horse | Cyberware, via the Georgia Tech Large Geometric Models Archive | research and non-commercial use; copyright of the creator (**) |
| `max_planck_points.npy` | Max Planck bust | Max-Planck-Institut für Informatik | research and non-commercial use; copyright of the creator (**) |
| `airplane_points.npy` | Airplane | sample file of the PLY format, shipped with [PyVista](https://github.com/pyvista/pyvista) | origin and license not documented |

Meshes for armadillo, xyzrgb_dragon, cow, horse, max_planck and spot were taken from
[alecjacobson/common-3d-test-models](https://github.com/alecjacobson/common-3d-test-models).

(*) Stanford 3D Scanning Repository: free for research, may be mirrored or redistributed for
free, images may be published in scholarly articles or books with credit to the Stanford
Computer Graphics Laboratory; no commercial use or use in a product for sale without Stanford's
permission.

(**) As listed in the [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/)
(Princeton): believed to be redistributable for research and non-commercial use, but remain
under copyright by their creators.

# Hyperuniform seed

`seed.npy` is a `(131072, 2)` float32 array: a stealthy hyperuniform point cloud of N = 2¹⁷
points on the unit square [0, 1)², periodic (it tiles the plane). It is the D = 2, l2 point
cloud `N17_D2.h5` of the
[Hyperuniform samples database](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers)
(release `dataset-v5.0.0`, [doi:10.5281/zenodo.23244509](https://doi.org/10.5281/zenodo.23244509)),
by R. Bardenet, R. Lachièze-Rey and A. de Cacqueray, MIT licence. `Resampler` cuts square
windows in it.
