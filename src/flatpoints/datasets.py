"""Example point clouds, downloaded on demand from the GitHub repository of flatpoints.

The clouds are derived from third-party 3D models and are NOT covered by the licence of
flatpoints: they are not shipped in the package, but fetched from
https://github.com/neighborhood-grid/flatpoints/tree/main/data the first time they are
used, then cached on disk. Each one keeps the terms of its original model (see `MODELS`,
or data/README.md in the repository): check them before any use, and credit the sources.

    points = fp.load("bunny")   # (10000, 3) float32
    fp.datasets()               # names, sources and terms
"""
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/neighborhood-grid/flatpoints/main/data/{}_points.npy"

_STANFORD = "Stanford 3D Scanning Repository terms: research use, no commercial use without permission"
_RESEARCH = "research and non-commercial use, copyright of the creator"
_AIMATSHAPE = ("AIM@SHAPE licence: research and non-commercial use; credit \"Elephant model is "
               "provided courtesy of INRIA by the AIM@SHAPE Shape Repository\"")

# name: (sha256 of the .npy, source, terms)
MODELS = {
    "airplane": ("3e2f639783dfbe25f20eb8efdab8dad71cf1a49c2c8f8ad7236d024638e18f1f",
                 "PLY format sample file, shipped with PyVista", "origin and licence not documented"),
    "armadillo": ("ce922b4a83772bbe3ecc903acf5dfcec1a567923cc42d6ac748e5d4376cfac9b",
                  "Stanford Computer Graphics Laboratory", _STANFORD),
    "bunny": ("aa3cdba2c957156b197d8e0937587a98281dbd87bd542f601dc06c5973d09a84",
              "Stanford Computer Graphics Laboratory", _STANFORD),
    "cow": ("5cfd15b81d98813ce5e5bab9ccbdc57da31d6f3f03fc4e4d9aea18a47a537dcf",
            "Viewpoint Animation Engineering / Sun Microsystems", _RESEARCH),
    "dragon": ("427a14245f5340a73cad43e85d9133ff0d3fb60e98bac989352bad306670e13d",
               "Stanford Computer Graphics Laboratory", _STANFORD),
    "elephant": ("24649adc1293b018faa4c3cdb2a20b7b7c5fa272280924954dcbb73eacc826d6",
                 "INRIA, via the AIM@SHAPE Shape Repository", _AIMATSHAPE),
    "everest": ("cc9dec8654673f60cef75490d493bb600eeb967f5695e7a2e45d145ce99ea97c",
                "Copernicus DEM GLO-90 (ESA)",
                "free licence; credit \"produced using Copernicus WorldDEM-90 (c) DLR e.V. 2010-2014 and "
                "(c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the "
                "European Union and ESA; all rights reserved\""),
    "horse": ("bb760ad2ed22ca8639c81ce121abf87791ffa50d78af5dec7c311526287f523f",
              "Cyberware, Georgia Tech Large Geometric Models Archive", _RESEARCH),
    "max_planck": ("2cae575f579f426656c25ae8bbb01535a7848d829e684d87e638a03e0ca81f5b",
                   "Max-Planck-Institut für Informatik", _RESEARCH),
    "spot": ("3f0a01a5c628cf112799db085dd0e279ce2c52b7e78a8843514087d689626d2f",
             "Keenan Crane", "CC0 1.0 (public domain)"),
}


def cache_dir():
    """Where the downloaded clouds are kept: $FLATPOINTS_DATA, else the user cache."""
    if os.environ.get("FLATPOINTS_DATA"):
        return Path(os.environ["FLATPOINTS_DATA"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "flatpoints"


def datasets():
    """{name: (source, terms)} of the example point clouds."""
    return {name: (source, terms) for name, (_, source, terms) in MODELS.items()}


def load(name, verbose=True):
    """(N, 3) float32 example point cloud `name` (see `datasets()`), downloaded on first use.

    The cloud is derived from a third-party model and keeps its terms, which are printed
    at download: they are not those of flatpoints.
    """
    if name not in MODELS:
        raise ValueError(f"unknown point cloud {name!r}, available: {', '.join(MODELS)}")
    sha, source, terms = MODELS[name]
    path = cache_dir() / f"{name}_points.npy"
    if not path.exists() or _sha256(path.read_bytes()) != sha:
        url = URL.format(name)
        if verbose:
            print(f"flatpoints: downloading {name} from {url}\n"
                  f"  source: {source}\n  terms: {terms} (not the flatpoints licence)")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if _sha256(data) != sha:
            raise OSError(f"{url}: checksum mismatch, the file may have changed upstream")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return np.load(path)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


__all__ = ["load", "datasets", "cache_dir"]
