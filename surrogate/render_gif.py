"""
Render the README's animation: fluid particles carried by the POD surrogate's predicted field, exactly
as the browser demo does it (RK2 advection of the steady velocity field, fading trails). Re sweeps
slowly across the training range so the vortex can be seen migrating while the fluid circulates.

    .venv\\Scripts\\python surrogate\\render_gif.py        # writes figures/cavity.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import data                       # noqa: E402
from pod import POD               # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "cavity.gif")

FRAMES, FPS, SIZE = 64, 16, 400
NP, DT, TRAIL = 1100, 0.45 / FPS, 8
RE_PATH = np.concatenate([np.full(12, 100.0), np.logspace(2, 3, 32), np.full(20, 1000.0)])   # hold, sweep, hold
LV_MAIN = [-0.117, -0.113, -0.108, -0.1, -0.09, -0.075, -0.06, -0.045, -0.03, -0.018, -0.009, -0.003, -0.0005]
LV_EDDY = [1e-6, 1e-5, 5e-5, 1.5e-4, 4e-4, 8e-4, 1.5e-3, 2.5e-3]


def sample(f, x, y):
    """Bilinear sample of the (2, N, N) cell-centred field at points x, y ∈ [0, 1] (arrays)."""
    N = f.shape[1]
    fx = np.clip(x * N - 0.5, 0, N - 1); fy = np.clip(y * N - 0.5, 0, N - 1)
    i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
    i1 = np.minimum(i0 + 1, N - 1); j1 = np.minimum(j0 + 1, N - 1)
    tx = fx - i0; ty = fy - j0
    w = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
    u = w[0] * f[0, i0, j0] + w[1] * f[0, i1, j0] + w[2] * f[0, i0, j1] + w[3] * f[0, i1, j1]
    v = w[0] * f[1, i0, j0] + w[1] * f[1, i1, j0] + w[2] * f[1, i0, j1] + w[3] * f[1, i1, j1]
    return u, v


def main():
    m = POD.load(os.path.join(os.path.dirname(__file__), "models", "pod.npz"))
    rng = np.random.default_rng(0)
    x = rng.random(NP); y = rng.random(NP)
    hist = [(x.copy(), y.copy())]
    frames = []
    N = m.shape[1]; c = (np.arange(N) + 0.5) / N
    for k, re in enumerate(RE_PATH):
        f = m.predict([re])[0]
        # advance particles (two half-steps per frame for smoothness)
        for _ in range(2):
            u1, v1 = sample(f, x, y)
            u2, v2 = sample(f, x + 0.5 * DT / 2 * u1, y + 0.5 * DT / 2 * v1)
            x = x + DT / 2 * u2; y = y + DT / 2 * v2
        out = (x <= 0.002) | (x >= 0.998) | (y <= 0.002) | (y >= 0.998) | (rng.random(NP) < 0.004)
        x[out] = rng.random(out.sum()); y[out] = rng.random(out.sum())
        for hx, hy in hist:
            hx[out] = x[out]; hy[out] = y[out]
        hist.append((x.copy(), y.copy())); hist = hist[-TRAIL:]

        fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        speed = np.hypot(f[0], f[1])
        ax.imshow(speed.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
        psi = data.streamfunction_from_centres(f)
        ax.contour(c, c, psi.T, levels=LV_MAIN, colors="white", linewidths=0.6, alpha=0.6)
        ax.contour(c, c, psi.T, levels=LV_EDDY, colors="#ff8a3d", linewidths=0.9)
        for t in range(1, len(hist)):
            (x0, y0), (x1, y1) = hist[t - 1], hist[t]
            a = (t / len(hist)) ** 1.5
            segs = np.stack([np.stack([x0, y0], 1), np.stack([x1, y1], 1)], 1)
            ax.add_collection(matplotlib.collections.LineCollection(segs, colors=(1, 1, 1, 0.9 * a), linewidths=1.1))
        ax.add_patch(plt.Rectangle((0, 0.985), 1, 0.015, color="#111827"))
        ax.text(0.5, 0.955, "lid slides this way at speed 1  →", color="white", ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#111827", ec="none", alpha=0.85))
        ax.text(0.03, 0.03, f"Re = {re:.0f}", color="white", fontsize=12, fontweight="bold", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.55))
        ax.text(0.97, 0.03, "POD surrogate · 20 modes · ~1 ms per field", color="white", fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.55))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        plt.close(fig)
        if k % 20 == 0:
            print(f"  frame {k}/{FRAMES}  Re = {re:.0f}", flush=True)

    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
