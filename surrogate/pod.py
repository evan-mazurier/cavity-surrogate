"""
Surrogate 1 — POD (proper orthogonal decomposition) + cubic-spline interpolation of the mode
coefficients in log Re. The classical reduced-order model:

    field(Re) ≈ mean + Σ_k a_k(Re) · φ_k

φ_k are the leading left-singular vectors of the (centred) training snapshots; a_k(Re) is a natural
cubic spline through the training coefficients (interpolation, not smoothing — there is no noise).
Outside the training range the spline continues linearly, so the error grows honestly.

Deterministic, ~20 numbers per Re, no training loop: this is the baseline the CNN has to beat.

    .venv\\Scripts\\python surrogate\\pod.py            # fits and saves surrogate/models/pod.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data                       # noqa: E402

MODELS = os.path.join(os.path.dirname(__file__), "models")


class CubicSpline1D:
    """Natural cubic spline through (x, y) with linear extrapolation. y may be (n, m)."""

    def __init__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        n = len(x)
        h = np.diff(x)
        # second derivatives M from the tridiagonal system (natural: M0 = Mn-1 = 0)
        A = np.zeros((n, n)); rhs = np.zeros((n, y.shape[1]))
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        self.M = np.linalg.solve(A, rhs)
        self.x, self.y, self.h = x, y, h

    def __call__(self, xq):
        xq = np.atleast_1d(np.asarray(xq, float))
        x, y, h, M = self.x, self.y, self.h, self.M
        out = np.empty((len(xq), y.shape[1]))
        for q, xv in enumerate(xq):
            if xv <= x[0]:                                     # linear extrapolation, left
                slope = (y[1] - y[0]) / h[0] - h[0] * (2 * M[0] + M[1]) / 6
                out[q] = y[0] + slope * (xv - x[0]); continue
            if xv >= x[-1]:                                    # linear extrapolation, right
                slope = (y[-1] - y[-2]) / h[-1] + h[-1] * (M[-2] + 2 * M[-1]) / 6
                out[q] = y[-1] + slope * (xv - x[-1]); continue
            i = np.searchsorted(x, xv) - 1
            t0, t1 = xv - x[i], x[i + 1] - xv
            out[q] = (M[i] * t1**3 + M[i + 1] * t0**3) / (6 * h[i]) \
                + (y[i] / h[i] - M[i] * h[i] / 6) * t1 + (y[i + 1] / h[i] - M[i + 1] * h[i] / 6) * t0
        return out


class POD:
    def __init__(self, k=20):
        self.k = k

    def fit(self, re, fields):
        n = len(re)
        X = fields.reshape(n, -1)
        self.shape = fields.shape[1:]
        self.mean = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.energy = np.cumsum(S**2) / np.sum(S**2)
        k = min(self.k, n)
        self.modes = Vt[:k]                                     # (k, 2N²)
        coeffs = (U[:, :k] * S[:k])                             # (n, k)
        order = np.argsort(re)
        self.spline = CubicSpline1D(data.feature(re[order]), coeffs[order])
        self.singular_values = S
        return self

    def predict(self, re):
        a = self.spline(data.feature(re))                       # (m, k)
        return (self.mean + a @ self.modes).reshape((-1,) + self.shape)

    def save(self, path):
        np.savez_compressed(path, k=self.k, mean=self.mean, modes=self.modes, shape=self.shape,
                            energy=self.energy, sv=self.singular_values,
                            spline_x=self.spline.x, spline_y=self.spline.y, spline_M=self.spline.M)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(int(d["k"]))
        m.mean, m.modes, m.shape, m.energy = d["mean"], d["modes"], tuple(d["shape"]), d["energy"]
        m.singular_values = d["sv"]
        sp = CubicSpline1D.__new__(CubicSpline1D)
        sp.x, sp.y, sp.M = d["spline_x"], d["spline_y"], d["spline_M"]
        sp.h = np.diff(sp.x)
        m.spline = sp
        return m


def main():
    re_tr, f_tr = data.load_split("train")
    re_ho, f_ho = data.load_split("holdout")
    print(f"train {len(re_tr)} runs, holdout {len(re_ho)}")
    print("\n  modes k    energy captured    holdout rel-L2 mean / max")
    best = None
    for k in (2, 5, 10, 15, 20, 30, 40, 64):
        m = POD(k).fit(re_tr, f_tr)
        errs = [data.rel_l2(p, t) for p, t in zip(m.predict(re_ho), f_ho)]
        print(f"  {k:6d}    {m.energy[min(k, len(m.energy)) - 1]:.8f}        {np.mean(errs):.2e} / {np.max(errs):.2e}")
        if best is None or np.mean(errs) < best[0]:
            best = (np.mean(errs), k)
    k = best[1]
    m = POD(k).fit(re_tr, f_tr)
    os.makedirs(MODELS, exist_ok=True)
    m.save(os.path.join(MODELS, "pod.npz"))
    print(f"\nsaved surrogate/models/pod.npz with k = {k} (best holdout mean error {best[0]:.2e})")


if __name__ == "__main__":
    main()
