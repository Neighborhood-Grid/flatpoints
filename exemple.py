"""Usage: python exemple.py --cow   (or any model of data/: --dragon, --bunny, ...)"""
import sys

import numpy as np
import src.flatpoints as fp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "cow"
    points = np.load(f"data/{model}_points.npy")   # (N, 3)
    points = np.random.permutation(points)
    print(points.shape)

    flat = fp.Flat(points, single_face=False, verbose=False)

    fig = fp.plot(flat)
    fig.savefig(f"plots/{model}.png")
    plt.show()
