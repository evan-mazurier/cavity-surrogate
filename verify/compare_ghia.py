"""
Validate the solver against Ghia et al. (1982): run the cavity at a given Re, overlay the
centreline profiles on the reference points, print the deviations, and draw the streamlines.

    .venv\\Scripts\\python verify\\compare_ghia.py --re 100 --n 64

Writes out/ghia_Re<re>_N<n>.png and prints a table. The verdict line at the end is the number
that matters: max |Δu| and |Δv| over the 17 reference points, as a fraction of the lid speed.
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solver import Cavity          # noqa: E402
import ghia                        # noqa: E402

SOLVER = "#3b5bdb"      # solver profile (line)
REF = "#e8590c"         # Ghia reference (markers) — validated pair, CVD ΔE 30
INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def run(re: float, n: int, tol: float):
    cav = Cavity(n, re)
    t0 = time.perf_counter()
    hist = cav.run(tol=tol, verbose=False)
    print(f"Re = {re:g}  N = {n}: steady at t = {cav.t:.2f}, {cav.steps} steps, "
          f"{time.perf_counter() - t0:.1f}s, max|div u| = {abs(cav.divergence_field()).max():.1e}")
    return cav, hist


def compare(cav: Cavity, re: int):
    (y, u_mid), (x, v_mid) = cav.centrelines()
    u_at = np.interp(ghia.Y, y, u_mid)
    v_at = np.interp(ghia.X, x, v_mid)
    du = u_at - ghia.U[re]
    dv = v_at - ghia.V[re]
    print(f"\n  {'y':>7} {'u_ghia':>9} {'u_solver':>9} {'Δu':>8}   |   {'x':>7} {'v_ghia':>9} {'v_solver':>9} {'Δv':>8}")
    for k in range(len(ghia.Y)):
        print(f"  {ghia.Y[k]:7.4f} {ghia.U[re][k]:9.5f} {u_at[k]:9.5f} {du[k]:+8.4f}   |   "
              f"{ghia.X[k]:7.4f} {ghia.V[re][k]:9.5f} {v_at[k]:9.5f} {dv[k]:+8.4f}")
    vx, vy, psi = cav.primary_vortex()
    gx, gy, gpsi = ghia.VORTEX[re]
    print(f"\n  primary vortex: solver ({vx:.4f}, {vy:.4f}) ψ = {psi:.5f}   "
          f"Ghia ({gx:.4f}, {gy:.4f}) ψ = {gpsi:.5f}   (grid h = {cav.h:.4f})")
    sus_u = np.isin(ghia.Y, ghia.SUSPECT.get(re, {}).get("u", []))
    sus_v = np.isin(ghia.X, ghia.SUSPECT.get(re, {}).get("v", []))
    for k in np.flatnonzero(sus_u | sus_v):
        print(f"  SUSPECT reference point excluded from the verdict: "
              f"{'u' if sus_u[k] else 'v'} at {'y' if sus_u[k] else 'x'} = {(ghia.Y if sus_u[k] else ghia.X)[k]:.4f} "
              f"(printed {(ghia.U if sus_u[k] else ghia.V)[re][k]:+.5f}, solver {(u_at if sus_u[k] else v_at)[k]:+.5f}) — see ghia.SUSPECT")
    du_ok, dv_ok = du[~sus_u], dv[~sus_v]
    print(f"  VERDICT  max|Δu| = {abs(du_ok).max():.4f}   max|Δv| = {abs(dv_ok).max():.4f}   "
          f"rms Δu = {np.sqrt((du_ok**2).mean()):.4f}   rms Δv = {np.sqrt((dv_ok**2).mean()):.4f}   "
          f"(fraction of lid speed, over {(~sus_u).sum()}+{(~sus_v).sum()} reference points)")
    return (y, u_mid, u_at), (x, v_mid, v_at)


def plot(cav: Cavity, re: int, prof_u, prof_v, hist, path: str):
    y, u_mid, _ = prof_u
    x, v_mid, _ = prof_v
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    fig.patch.set_facecolor("white")

    # --- streamlines (ψ contours): diverging, two hues + neutral mid — ψ < 0 primary, ψ > 0 corner eddies
    ax = axes[0]
    psi = cav.streamfunction()
    nodes = np.linspace(0, 1, cav.N + 1)
    lv_main = np.array([-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, -1e-5, -1e-7])
    lv_eddy = np.array([1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3])
    levels = np.concatenate([lv_main, lv_eddy])
    ax.contour(nodes, nodes, psi.T, levels=levels, colors=[SOLVER if l < 0 else REF for l in levels],
               linewidths=0.9)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_title(f"Streamfunction, Re = {re}  (N = {cav.N})", color=INK, fontsize=11, loc="left")
    ax.text(0.02, 0.965, "lid  →", transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
    vx, vy, _ = cav.primary_vortex()
    ax.plot(vx, vy, "+", color=INK, ms=9, mew=1.2)
    ax.text(0.5, -0.12, "blue: primary vortex (ψ < 0) · orange: corner eddies (ψ > 0)",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, ha="center")

    # --- centreline profiles
    for ax, (ref_pos, ref_val, pos, val, xlabel, ylabel, title) in zip(axes[1:], [
        (ghia.Y, ghia.U[re], u_mid, y, "u", "y", "u along x = ½"),
        (ghia.X, ghia.V[re], x, v_mid, "x", "v", "v along y = ½"),
    ]):
        ax.plot(pos, val, color=SOLVER, lw=2, label="this solver", zorder=2)
        sus = np.isin(ref_pos, ghia.SUSPECT.get(re, {}).get(ylabel, []))
        px, py = (ref_pos, ref_val) if xlabel == "x" else (ref_val, ref_pos)
        ax.plot(px[~sus], py[~sus], "o", color=REF, ms=6, mfc="white", mew=1.8,
                label="Ghia et al. 1982", zorder=3)
        if sus.any():
            ax.plot(px[sus], py[sus], "x", color=MUTED, ms=8, mew=1.6,
                    label="Ghia point flagged suspect", zorder=3)
        ax.set_xlabel(xlabel, color=INK); ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
    for ax in axes[1:]:
        if len(ax.get_legend_handles_labels()[0]) > 2 or ax is axes[1]:
            ax.legend(frameon=False, fontsize=9, loc="lower right" if ax is axes[1] else "lower left")
    axes[1].set_xlim(-0.5, 1.05); axes[1].set_ylim(0, 1)
    axes[2].set_xlim(0, 1)

    # --- residual inset on the v panel
    ins = axes[2].inset_axes([0.58, 0.62, 0.4, 0.33])
    t, r = zip(*hist)
    ins.semilogy(t, r, color=MUTED, lw=1.2)
    ins.set_title("|∂u/∂t|max vs t", fontsize=7.5, color=MUTED, loc="left")
    ins.tick_params(labelsize=6.5, colors=MUTED)
    for s in ins.spines.values(): s.set_color(GRID)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=int, default=100, choices=[100, 400, 1000])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    os.makedirs("out", exist_ok=True)
    cav, hist = run(a.re, a.n, a.tol)
    prof_u, prof_v = compare(cav, a.re)
    plot(cav, a.re, prof_u, prof_v, hist, f"out/ghia_Re{a.re}_N{a.n}.png")


if __name__ == "__main__":
    main()
