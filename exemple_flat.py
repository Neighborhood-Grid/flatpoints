"""Usage: python exemple_flat.py --cow   (or any model of data/: --dragon, --bunny, --everest...)"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import src.flatpoints as fp

if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "cow"
    points = np.load(f"data/{model}_points.npy")  # (N, 3)

    # A closed object (cow, bunny...) is cut in two faces, left and right, across the
    # normal axis x. A terrain (everest) is an open surface: a single face, seen from above
    # (normal axis z).
    everest = model == "everest"
    flat = fp.Flat(points, single_face=everest, normal_axis="z" if everest else "x")

    print("points  ", points.shape)               # (N, 3)
    print("pixels  ", flat.pixels.shape)          # (N, 3)  (face, row, col), face 0 if single face
    print("images  ", (flat.n_faces, flat.n, flat.n))
    print("padding ", flat.corners.idxs.shape)    # (Npadding,) points repeated in the empty pixels (to fill the full n*n grid)
    if flat.faces is not None:
        print("faces   ", {f: len(i) for f, i in flat.faces.items()})

    # The flattening at a glance: the surface in 3D and its images, RGB and checkerboard
    flat.plot().savefig(f"plots/{model}.png")
    plt.show()

    # The point cloud as an image: (n_faces, n, n, 3), n_faces = 2 (left, right) or 1
    image = np.zeros((flat.n_faces, flat.n, flat.n, 3))
    image[tuple(flat.pixels.T)] = points                              # each point on its pixel
    image[tuple(flat.corners.pixels.T)] = points[flat.corners.idxs]   # fill the empty pixels
    assert np.array_equal(image, flat.to_image(points))               # the same, in one call

    # Any per-point data (N, *C) as images, one per face, coloured by its channels C:
    # default: the points, RGB = x, y, z
    #flat.imshow(points[:, 2])        # 1 channel: colormap, here the height
    #plt.show()
    # The same, animated (webp): the images fold into the surface in 3D, and back
    #flat.imshow(animate=f"plots/{model}.webp")  # resolution="high": cleaner, slower
