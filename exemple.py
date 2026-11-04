"""Usage: python exemple.py --cow   (or any model of data/: --dragon, --bunny, --everest...)"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import src.flatpoints as fp

if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "cow"
    points = np.load(f"data/{model}_points.npy")  # (N, 3)

    # A closed object (cow, bunny...) is cut in two faces, left and right, across `lateral`.
    # A terrain (everest) is an open surface: a single face, seen from above (lateral = z).
    everest = model == "everest"
    flat = fp.Flat(points, single_face=everest, lateral="z" if everest else "x")

    print("points  ", points.shape)               # (N, 3)
    print("uv      ", flat.uv.shape)              # (N, 2)  flat coordinates in [0, 1]
    print("pixels  ", flat.pixels.shape)          # (N, 3)  (face, row, col), face 0 if single face
    print("padding ", flat.corners.idxs.shape)    # (Npadding,) points repeated in the empty pixels (to fill the full n*n grid)
    if flat.faces is not None:
        print("faces   ", {f: len(i) for f, i in flat.faces.items()})

    # I. ISOMAPS: xyz -> uv
    fig = fp.plot(flat)
    fig.savefig(f"plots/{model}.png")
    plt.show()

    # II. Images: idx -> row, column
    # The point cloud as an image: (n_faces, n, n, 3), n_faces = 2 (left, right) or 1
    image = np.zeros((flat.n_faces, flat.n, flat.n, 3))
    image[tuple(flat.pixels.T)] = points                              # each point on its pixel
    image[tuple(flat.corners.pixels.T)] = points[flat.corners.idxs]   # fill the empty pixels
    assert np.array_equal(image, flat.to_image(points))               # the same, in one call
    #Uncomment to show it, left face (or the only one), rgb channels = xyz
    #fp.imshow(image[0])
    #plt.show()
