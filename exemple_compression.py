"""Usage: python exemple_compression.py --bunny   (or any model of data/: --dragon, --cow, --everest...)"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import src.flatpoints as fp
from src.flatpoints.compressor import available
from src.flatpoints.viz import camera

if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "bunny"
    points = np.load(f"data/{model}_points.npy")  # (N, 3)
    everest = model == "everest"  # open surface: a single face, seen from above
    axis = "z" if everest else "x"
    flat = fp.Flat(points, single_face=everest, normal_axis=axis)

    # 80% of the storage saved (against float32 points), the most accurate method
    comp = flat.compress(factor=0.8)
    print("method   ", comp.method)              # "dct", "wavelet" or "svd"
    print("saved    ", f"{comp.ratio:.1%}", f"({comp.bits / len(points):.1f} bits per point)")
    print("code     ", {k: getattr(v, "shape", v) for k, v in comp.code.items()})
    print("points   ", comp.points.shape)        # (N, 3) reconstructed, same order as points

    # Error of each point, in point spacings (mean distance to the nearest neighbour)
    spacing = cKDTree(points).query(points, k=2)[0][:, 1].mean()
    for method in available():
        c = flat.compress(0.8, method=method)
        print(f"{method:8s} 80%: RMS error {c.error / spacing:.3f} spacings")

    # Strong compressions (default method), coloured by the error of each point; then the
    # RMS error of each method against the storage saved
    factors = [0.9, 0.95, 0.98, 0.99]
    comps = [flat.compress(f) for f in factors]
    errors = [np.linalg.norm(c.points - points, axis=1) / spacing for c in comps]
    proj = camera(axis)
    fig, axs = plt.subplots(1, len(factors) + 2, figsize=(4 * (len(factors) + 2), 4),
                            layout="constrained")
    vmax = np.percentile(errors[-1], 95)
    panels = [("original", points, None)] + [
        (f"{c.ratio:.0%} saved, {c.method}", c.points, e) for c, e in zip(comps, errors)]
    for ax, (title, xyz, err) in zip(axs, panels):
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
    grid = np.linspace(0.5, 0.995, 30)
    for method in available():
        curve = [flat.compress(f, method=method) for f in grid]
        axs[-1].plot([100 * c.ratio for c in curve], [c.error / spacing for c in curve],
                     label=method)
    axs[-1].set_xlabel("storage saved (%)")
    axs[-1].set_ylabel("RMS error (point spacings)")
    axs[-1].set_yscale("log")
    axs[-1].set_xlim(50, 100)
    axs[-1].grid(alpha=0.3)
    axs[-1].legend()

    Path("plots_compression").mkdir(exist_ok=True)
    fig.savefig(f"plots_compression/{model}.png", dpi=100)
    plt.show()
