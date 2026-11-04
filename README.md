# flatpoints

Turn a point cloud sampled on a surface into a square `(n, n)` grid, where neighbouring
pixels hold points that are close **on the surface**. You can then use 2D tools on it:
convolutions, SVD, image codecs…

![bunny](plots/bunny.png)
![dragon](plots/dragon.png)

## Install

```bash
pip install -e ".[viz]"
```

## Quickstart

Take a raw 3D point cloud (N, 3), and flatten it to 2D. \
Available: bunny, dragon, airplane, cow, everest...

```bash
python exemple.py --bunny
```

## Usage

```python
xyz = np.load("data/bunny_points.npy") #raw (N, 3) point cloud
data = xyz #(N, *C) data related to the point cloud, e.g. the points themselves

# ==== Build the flat object ============

flat = Flat(xyz) # Apply the flattening processing of the surface

# ==== Use it  ==========================

#I. Isomaps (projection of the points, latent 2D coordinates on the surface xyz -> uv):
uv = flat.uv #(N, 2)

#II. Pixels (learned multi index of the points: idx -> row, column)
i = 42 #point with index 42
face, row, col = flat.pixels[i] # face = 0 or 1 for left and right hemispheres (see topology section)

#III. Images (convert any data to an image using the pixel layout):
image = flat.to_image(data)    # (N, *C) to  (2, n, n, *C)
```

### Topology

Use `fp.Flat(points, single_face=True)` for an open surface (e.g. a terrain), \
to flatten it on a single map.

Use `fp.Flat(points, single_face=False)` for a closed surface (e.g. a dragon), \
to flatten left and right hemispheres independantly

See [exemple.py](exemple.py).

## Licence

- code: MIT
- data: see [data/README.md](data/README.md).
