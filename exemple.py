import numpy as np
import src.flatpoints as fp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = "dragon"
    points = np.load(f"data/{model}_points.npy")   # (N, 3)
    print(points.shape)

    flat = fp.Flat(points, single_face=False, verbose=False)

    fig = fp.plot(flat, squares = 5)
    fig.savefig(f"plots/{model}.png")
    plt.show()

    fp.imshow(flat)
    plt.show()
