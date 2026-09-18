"""
Evaluate both surrogates against the solver on Re values they never trained on.

  holdout  — inside the training range, interleaved with the training points
  extrap   — outside it (30, 40 and 1750–2500)

For each case and model: relative L2 error of the (u, v) field, RMS centred divergence (against the
solver's own centred-field baseline), and the vortex-centre offset. Writes figures/surrogate_error.png
(error vs Re, training range shaded) and figures/surrogate_fields.png (one held-out case, side by side),
and prints the summary table.

    .venv\\Scripts\\python surrogate\\evaluate.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import data                       # noqa: E402
from pod import POD               # noqa: E402
from cnn import CNN               # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
MODELS = os.path.join(os.path.dirname(__file__), "models")

C_POD = "#3b5bdb"; C_CNN = "#e8590c"; INK = "#1f2933"; MUTED = "#6b7280"; GRID = "#e5e7eb"


def evaluate(models, split):
    re, fields = data.load_split(split)
    rows = []
    preds = {name: m.predict(re) for name, m in models.items()}
    for i, r in enumerate(re):
        row = dict(re=r, div_solver=data.centred_divergence(fields[i]), vc_solver=data.vortex_centre(fields[i]))
        for name in models:
            p = preds[name][i]
            row[f"err_{name}"] = data.rel_l2(p, fields[i])
            row[f"div_{name}"] = data.centred_divergence(p)
            vc = data.vortex_centre(p)
            row[f"vc_{name}"] = float(np.hypot(vc[0] - row["vc_solver"][0], vc[1] - row["vc_solver"][1]))
        rows.append(row)
    return rows


def table(rows, models, title):
    print(f"\n{title}")
    print(f"  {'Re':>6} " + " ".join(f"{'err_' + n:>9} {'div_' + n:>9} {'Δvc_' + n:>8}" for n in models) + f" {'div_solver':>11}")
    for r in rows:
        print(f"  {r['re']:6.0f} " + " ".join(f"{r['err_' + n]:9.2e} {r['div_' + n]:9.2e} {r['vc_' + n]:8.4f}" for n in models)
              + f" {r['div_solver']:11.2e}")
    for n in models:
        e = [r["err_" + n] for r in rows]
        print(f"  {n:>6}: rel-L2 mean {np.mean(e):.2e}  max {np.max(e):.2e}   "
              f"divergence median {np.median([r['div_' + n] for r in rows]):.2e} (solver's own {np.median([r['div_solver'] for r in rows]):.2e})   "
              f"vortex-centre offset mean {np.mean([r['vc_' + n] for r in rows]):.4f}")


def plot_error(models, rows_tr, rows_ho, rows_ex, lo, hi, path):
    fig, ax = plt.subplots(figsize=(9, 4.6)); fig.patch.set_facecolor("white")
    ax.axvspan(lo, hi, color="#eef1fb", zorder=0)
    ax.text((lo * hi) ** 0.5, 0.02, "training range", color=MUTED, fontsize=9, ha="center", va="bottom", transform=ax.get_xaxis_transform())
    for name, col in zip(models, (C_POD, C_CNN)):
        rows = sorted(rows_ho + rows_ex, key=lambda r: r["re"])
        ax.plot([r["re"] for r in rows], [r["err_" + name] for r in rows], "-", color=col, lw=1.2, alpha=0.5, zorder=2)
        ax.plot([r["re"] for r in rows_ho], [r["err_" + name] for r in rows_ho], "o", color=col, ms=6, mfc=col, zorder=3,
                label=f"{name} — held-out Re")
        ax.plot([r["re"] for r in rows_ex], [r["err_" + name] for r in rows_ex], "s", color=col, ms=6, mfc="white", mew=1.6, zorder=3,
                label=f"{name} — outside the range")
        ax.plot([r["re"] for r in rows_tr], [r["err_" + name] for r in rows_tr], ".", color=col, ms=4, alpha=0.35, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Re", color=INK); ax.set_ylabel("relative L2 error of (u, v) vs the solver", color=INK)
    ax.set_title("Surrogate error vs Reynolds number — small dots: training reconstruction", color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, which="both")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="center")
    fig.tight_layout(); fig.savefig(path, dpi=140); print(f"  wrote {path}")


def plot_fields(models, re, path):
    true, _ = data.load(re)
    preds = {n: m.predict([re])[0] for n, m in models.items()}
    cols = [("solver", true)] + [(n, p) for n, p in preds.items()]
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 8.2)); fig.patch.set_facecolor("white")
    N = true.shape[1]; c = (np.arange(N) + 0.5) / N
    levels = np.concatenate([np.linspace(-0.12, -0.005, 12), [-1e-4, -1e-5, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3]])
    for ax, (name, f) in zip(axes[0], cols):
        psi = data.streamfunction_from_centres(f)
        ax.contour(c, c, psi.T, levels=levels, colors=[C_POD if l < 0 else C_CNN for l in levels], linewidths=0.8)
        ax.set_aspect("equal"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(colors=MUTED)
        ax.set_title(f"{name}  Re = {re:g}" + ("" if name == "solver" else f"   rel-L2 {data.rel_l2(f, true):.1e}"),
                     color=INK, fontsize=10, loc="left")
    axes[1, 0].axis("off")
    caption = ("Top: streamfunction (blue ψ < 0, orange ψ > 0).",
               "Bottom: |error| of (u, v) against the solver.",
               "",
               "POD is indistinguishable from the solver.",
               "The CNN's error norm is small, yet it invents",
               "structure along the side walls that is not",
               "there — a low error can still hide wrong physics.")
    axes[1, 0].text(0.0, 0.95, chr(10).join(caption), transform=axes[1, 0].transAxes, va="top",
                    color=MUTED, fontsize=9.5, linespacing=1.5)
    vmax = max(abs(p - true).max() for p in preds.values())
    for ax, (name, f) in zip(axes[1, 1:], cols[1:]):
        err = np.hypot(*(f - true))
        im = ax.imshow(err.T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"|error|  max {err.max():.2e}", color=INK, fontsize=10, loc="left")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(colors=MUTED)
    fig.colorbar(im, ax=axes[1, 1:].tolist(), fraction=0.03, pad=0.02)
    fig.savefig(path, dpi=130, bbox_inches="tight"); print(f"  wrote {path}")


def main():
    models = {"POD": POD.load(os.path.join(MODELS, "pod.npz")), "CNN": CNN.load(os.path.join(MODELS, "cnn.pt"))}
    man = data.manifest()
    lo, hi = min(man["train"]), max(man["train"])
    rows_tr = evaluate(models, "train")
    rows_ho = evaluate(models, "holdout")
    rows_ex = evaluate(models, "extrap")
    for n in models:
        e = [r["err_" + n] for r in rows_tr]
        print(f"train reconstruction {n}: rel-L2 mean {np.mean(e):.2e} max {np.max(e):.2e}")
    table(rows_ho, models, f"HELD-OUT Re (inside [{lo}, {hi}], never trained on)")
    table(rows_ex, models, "OUTSIDE THE TRAINING RANGE")
    os.makedirs(FIG, exist_ok=True)
    plot_error(models, rows_tr, rows_ho, rows_ex, lo, hi, os.path.join(FIG, "surrogate_error.png"))
    plot_fields(models, 895, os.path.join(FIG, "surrogate_fields.png"))


if __name__ == "__main__":
    main()
