"""Usage: python exemple.py --cow   (or any model of data/: --dragon, --bunny, ...)"""
import sys

import numpy as np
import src.flatpoints as fp
import matplotlib.pyplot as plt
  

if __name__ == "__main__":
    model = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "cow"
    points = np.load(f"data/{model}_points.npy")   # (N, 3)
    print(points.shape)

    everest = model == "everest" 
    single_face = everest #(topology closed ojects like balls have two hemispheres, everest have only one)
    lateral = "z" if everest else "x" #lateral is a plan roughly orthonormal to the u,v isomap
    flat = fp.Flat(points, single_face = single_face, lateral = lateral,  verbose=False)
    fig = fp.plot(flat)
    fig.savefig(f"plots/{model}.png")
    plt.show()
