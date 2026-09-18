"""Load the generated runs. Fields are (u, v) at the N×N cell centres, stacked as (2, N, N)."""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "data", "runs")


def manifest():
    with open(os.path.join(ROOT, "data", "manifest.json")) as f:
        return json.load(f)


def load(re):
    d = np.load(os.path.join(RUNS, f"Re{int(re)}.npz"))
    return np.stack([d["u"], d["v"]]), d


def load_split(name):
    """name ∈ {'train', 'holdout', 'extrap'} → (Re array, fields (n, 2, N, N))."""
    res = np.array(manifest()[name], float)
    fields = np.stack([load(r)[0] for r in res])
    return res, fields


def feature(re):
    """The model input: log10(Re), scaled so the training range [50, 1500] maps to about [-1, 1]."""
    lo, hi = np.log10(50), np.log10(1500)
    return (np.log10(np.asarray(re, float)) - (lo + hi) / 2) / ((hi - lo) / 2)


# ---------------------------------------------------------------- metrics
def rel_l2(pred, true):
    """Relative L2 error of the (2, N, N) field: ||pred − true|| / ||true||."""
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def centred_divergence(field):
    """∇·u from the cell-centred field by central differences (interior). The solver's own
    centred field has a small non-zero value here (it is exactly divergence-free on the staggered
    grid, not on centres) — so compare a prediction to THAT baseline, not to zero."""
    u, v = field
    N = u.shape[0]
    h = 1.0 / N
    div = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h) + (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h)
    return float(abs(div).max())


def streamfunction_from_centres(field):
    """ψ on cell centres by integrating u along y (ψ = 0 at the bottom wall). Approximate — for
    locating the vortex centre and drawing, not for verification."""
    u = field[0]
    h = 1.0 / u.shape[0]
    return np.cumsum(u * h, axis=1) - 0.5 * u * h


def vortex_centre(field):
    psi = streamfunction_from_centres(field)
    i, j = np.unravel_index(np.argmin(psi), psi.shape)
    h = 1.0 / psi.shape[0]
    return (i + 0.5) * h, (j + 0.5) * h
