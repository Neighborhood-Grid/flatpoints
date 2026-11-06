"""Usage: python exemple_sampler.py --bunny   (or any model of data/: --dragon, --cow, --everest...)"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import src.flatpoints as fp

MODELS = ["airplane", "armadillo", "bunny", "cow", "dragon", "everest", "horse",
          "max_planck", "spot"]


def flatten(model):
    points = np.load(f"data/{model}_points.npy")  # (N, 3)
    everest = model == "everest"  # open surface: a single face, seen from above
    return fp.Flat(points, single_face=everest, lateral="z" if everest else "x")


if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "bunny"

    flat = flatten(model)
    # n_points new points in total, transported from a window of the hyperuniform seed
    sampler = flat.resample(n_points=512)
    print("flat     ", flat.n_faces, "faces of", flat.n, "x", flat.n)
    print("points   ", sampler.points.shape)   # (n_points, 3) the new points
    print("faces    ", sampler.faces.shape)    # (n_points,)   face of each new point
    print("pixels   ", sampler.pixels.shape)   # (n_points, 3) float (face, row, col)

    # Another, independent resampling: the flat is not recomputed
    other = flat.resample(n_points=512, seed=1)

    # The new points (discs) on the mesh of the full cloud (quads of the flat)
    fig = sampler.plot()
    fig.savefig(f"plots_resampling/{model}_resampled.png")
    plt.show()
