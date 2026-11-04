"""Usage: python exemple_sampler.py --bunny   (or any model of data/: --dragon, --cow, --everest...)
       python exemple_sampler.py --all     (every model, hyperuniform vs i.i.d. random points)"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import src.flatpoints as fp

MODELS = ["airplane", "armadillo", "bunny", "cow", "dragon", "everest", "horse",
          "max_planck", "spot"]


def resample(model, n=256, seed=None):
    points = np.load(f"data/{model}_points.npy")  # (N, 3)
    everest = model == "everest"  # open surface: a single face, seen from above
    return fp.Resampler(points, n=n, seed=seed, single_face=everest,
                        lateral="z" if everest else "x")


if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "bunny"

    if model != "all":
        # n new points per face, transported from a window of the hyperuniform seed
        sampler = resample(model, n=256)
        print("flat     ", sampler.flat.n_faces, "faces of", sampler.flat.n, "x", sampler.flat.n)
        print("points   ", sampler.points.shape)   # (n_faces * n, 3) the new points
        print("faces    ", sampler.faces.shape)    # (n_faces * n,)   face of each new point
        print("pixels   ", sampler.pixels.shape)   # (n_faces * n, 3) float (face, row, col)

        # The new points (discs) on the mesh of the full cloud (quads of sampler.flat)
        fig = sampler.plot()
        fig.savefig(f"plots/{model}_resampled.png")
        plt.show()

    else:
        # Every model, with the hyperuniform seed,
        random = np.random.default_rng(0).random((2 ** 17, 2))
        seed, name  = None, "resampled"
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), layout="constrained")
        for m, ax in zip(MODELS, axes.ravel()):
            sampler = resample(m, n=256, seed=seed)
            sampler.plot(ax=ax)
            ax.set_title(f"{m} ({len(sampler.points)} pts)", fontsize=14)
        fig.suptitle("random i.i.d." if name.endswith("random") else "hyperuniform",
                        fontsize=18)
        fig.savefig(f"plots/{name}.png")
        plt.show()
