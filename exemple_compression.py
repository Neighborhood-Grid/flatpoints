"""Usage: python exemple_compression.py --bunny   (or any model of data/: --dragon, --cow, --everest...)"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import src.flatpoints as fp
from src.flatpoints.viz import camera

if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "bunny"
    points = np.load(f"data/{model}_points.npy")  # (N, 3)
    everest = model == "everest"  # open surface: a single face, seen from above
    axis = "z" if everest else "x"
    flat = fp.Flat(points, single_face=everest, normal_axis=axis)

    # Truncated SVD of the xyz images, 80% of the storage saved
    comp = flat.compress(factor=0.8)
    print("rank     ", comp.k, "of", flat.n)
    print("saved    ", f"{comp.ratio:.0%}")
    print("U, S, Vt ", comp.U.shape, comp.S.shape, comp.Vt.shape)
    print("points   ", comp.points.shape)  # (N, 3) reconstructed, same order as points

    # Error of each point, in point spacings (mean distance to the nearest neighbour)
    spacing = cKDTree(points).query(points, k=2)[0][:, 1].mean()
    factors = [0.5, 0.8, 0.9, 0.95]
    comps = [flat.compress(f) for f in factors]
    errors = [np.linalg.norm(c.points - points, axis=1) / spacing for c in comps]
    for f, c, e in zip(factors, comps, errors):
        print(f"factor {f:.0%}: rank {c.k:3d}, saved {c.ratio:.0%}, "
              f"error median {np.median(e):.2f}, 95% {np.percentile(e, 95):.2f} spacings")

    # Original and reconstructions, coloured by the error; then the error vs the storage saved
    proj = camera(axis)
    fig, axs = plt.subplots(1, len(factors) + 2, figsize=(4 * (len(factors) + 2), 4),
                            layout="constrained")
    vmax = np.percentile(errors[-1], 95)
    for ax, (title, xyz, err) in zip(axs, [("original", points, None)] + [
            (f"{c.ratio:.0%} saved, rank {c.k}", c.points, e) for c, e in zip(comps, errors)]):
        p = xyz @ proj.T
        order = np.argsort(p[:, 2])  # far first
        if err is None:
            ax.scatter(p[order, 0], p[order, 1], s=1, c="0.4")
        else:
            sc = ax.scatter(p[order, 0], p[order, 1], s=1, c=err[order], cmap="magma_r",
                            vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sc, ax=axs[-2], label="error (point spacings)")
    curve = [flat.compress(f) for f in np.linspace(0, 0.97, 40)]
    axs[-1].plot([100 * c.ratio for c in curve],
                 [np.median(np.linalg.norm(c.points - points, axis=1)) / spacing for c in curve])
    axs[-1].set_xlabel("storage saved (%)")
    axs[-1].set_ylabel("median error (point spacings)")
    axs[-1].set_yscale("log")
    axs[-1].grid(alpha=0.3)

    Path("plots_compression").mkdir(exist_ok=True)
    fig.savefig(f"plots_compression/{model}.png", dpi=100)
    plt.show()
