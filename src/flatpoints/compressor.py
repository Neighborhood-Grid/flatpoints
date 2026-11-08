"""Compression of the images of a `Flat`: three methods, used by `main.Compressor`.

Each method takes the images A (m, n, n), m = n_faces x channels, and a budget in bits, and
returns (code, reconstructed A, bits): what would be stored, what it decodes to, its size.

- svd: truncated SVD of each image, U, S, Vt stored as float32.
- dct: 2D DCT of each image, uniformly quantized, entropy coded per frequency band (as JPEG,
  with one transform over the whole image).
- wavelet: 2D wavelet transform (biorthogonal 4.4, as JPEG2000), uniformly quantized,
  entropy coded per subband. Needs PyWavelets (`pip install PyWavelets`).

dct and wavelet use the step of the quantization to fit the budget. Their size is the one of
an ideal entropy coder per band (zeroth-order entropy): a real arithmetic coder is a few
percent larger.
"""
from importlib.util import find_spec

import numpy as np
from scipy.fft import dctn, idctn

FLOAT = 32   # bits of a stored float (svd, quantization step)
BANDS = 12   # dct: number of frequency bands, each with its own entropy coder


def svd(A, budget):
    """Keep the k largest singular values of each image, k (2n + 1) floats per image."""
    m, n = A.shape[:2]
    k = int(np.clip(budget // (FLOAT * m * (2 * n + 1)), 1, n))
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    code = {"U": U[..., :k], "S": S[..., :k], "Vt": Vt[..., :k, :]}
    rec = (code["U"] * code["S"][..., None, :]) @ code["Vt"]
    return code, rec, FLOAT * m * k * (2 * n + 1)


def dct(A, budget):
    """Quantized 2D DCT, the frequencies grouped in bands of u + v."""
    n = A.shape[1]
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    bands = np.minimum((u + v) * BANDS // (2 * n - 1), BANDS - 1)
    C = dctn(A, axes=(1, 2), norm="ortho")
    q, step, bits = _quantize(C, [bands == b for b in range(BANDS)], budget)
    code = {"coefficients": q, "step": step}
    return code, idctn(q * step, axes=(1, 2), norm="ortho"), bits


def wavelet(A, budget, name="bior4.4"):
    """Quantized 2D wavelet transform (periodic), one entropy coder per subband."""
    import pywt
    n = A.shape[1]
    level = min(4, pywt.dwt_max_level(n, pywt.Wavelet(name).dec_len))
    arrays = [pywt.coeffs_to_array(pywt.wavedec2(a, name, mode="periodization", level=level))
              for a in A]
    C, slices = np.stack([a for a, _ in arrays]), arrays[0][1]
    bands = [np.zeros(C.shape[1:], bool) for _ in range(1 + 3 * level)]
    bands[0][slices[0]] = True
    for j, details in enumerate(slices[1:]):
        for i, key in enumerate(("ad", "da", "dd")):
            bands[1 + 3 * j + i][details[key]] = True
    q, step, bits = _quantize(C, bands, budget)

    def inverse(c):
        coeffs = pywt.array_to_coeffs(c, slices, output_format="wavedec2")
        return pywt.waverec2(coeffs, name, mode="periodization")[:n, :n]

    code = {"coefficients": q, "step": step, "wavelet": name, "level": level}
    return code, np.stack([inverse(c) for c in q * step]), bits


METHODS = {"svd": svd, "dct": dct, "wavelet": wavelet}


def available():
    """The methods that can run here: wavelet only if PyWavelets is installed."""
    return list(METHODS) if find_spec("pywt") else ["svd", "dct"]


# === Quantization ===

def _entropy(q):
    """Bits of an ideal entropy coder (zeroth order) for the integers q."""
    _, counts = np.unique(q, return_counts=True)
    return -(counts * np.log2(counts / counts.sum())).sum()


def _quantize(C, bands, budget):
    """(q, step, bits): C / step rounded, with the finest step whose size fits the
    budget (bisection on log(step)); each band of coefficients is coded on its own."""
    def size(step):
        q = np.round(C / step)
        return q, FLOAT + sum(_entropy(q[:, b]) for b in bands)

    lo, hi = np.log(np.abs(C).max() * 1e-7 + 1e-30), np.log(np.abs(C).max() * 2 + 1e-30)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if size(np.exp(mid))[1] <= budget else (mid, hi)
    q, bits = size(np.exp(hi))
    return q.astype(np.int32), float(np.exp(hi)), bits
