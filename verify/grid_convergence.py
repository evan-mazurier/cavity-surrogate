"""
Grid convergence: run the same Re on successively finer grids and measure how fast the solution
stops changing. For a second-order scheme the difference between successive grids should shrink
by ~4× per halving of h, i.e. an observed order p ≈ 2, with p = log2(|f_h − f_2h| / |f_h/2 − f_h|).

Compared quantity: the centreline profiles u(½, y) and v(x, ½), sampled at the 17 Ghia points
(so the same numbers also show how the deviation from Ghia behaves as h → 0).

    .venv\\Scripts\\python verify\\grid_convergence.py --re 100 --grids 32 64 128
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solver import Cavity          # noqa: E402
import ghia                        # noqa: E402


def profiles(cav):
    (y, u_mid), (x, v_mid) = cav.centrelines()
    return np.interp(ghia.Y, y, u_mid), np.interp(ghia.X, x, v_mid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=int, default=100)
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    runs = {}
    for n in a.grids:
        cav = Cavity(n, a.re)
        t0 = time.perf_counter()
        cav.run(tol=a.tol, verbose=False)
        u, v = profiles(cav)
        vx, vy, psi = cav.primary_vortex()
        runs[n] = dict(u=u, v=v, psi=psi, vortex=(vx, vy))
        print(f"N = {n:4d}: {cav.steps:7d} steps, {time.perf_counter() - t0:6.1f}s, "
              f"psi_min = {psi:.6f}, vortex ({vx:.4f}, {vy:.4f})")

    g = a.grids
    print(f"\nRe = {a.re}: change between successive grids (max over the 17 centreline points)")
    print(f"  {'grids':>12} {'|Δu|max':>10} {'|Δv|max':>10} {'|Δψmin|':>10}")
    du, dv, dp = [], [], []
    for n1, n2 in zip(g[:-1], g[1:]):
        du.append(abs(runs[n2]["u"] - runs[n1]["u"]).max())
        dv.append(abs(runs[n2]["v"] - runs[n1]["v"]).max())
        dp.append(abs(runs[n2]["psi"] - runs[n1]["psi"]))
        print(f"  {n1:>5} → {n2:<5} {du[-1]:10.2e} {dv[-1]:10.2e} {dp[-1]:10.2e}")
    if len(g) >= 3:
        print("\n  observed order p = log2(Δ_coarse / Δ_fine)   [second-order scheme → p ≈ 2]")
        for k in range(len(du) - 1):
            print(f"  {g[k]}/{g[k+1]}/{g[k+2]}:  p_u = {np.log2(du[k] / du[k+1]):.2f}   "
                  f"p_v = {np.log2(dv[k] / dv[k+1]):.2f}   p_ψ = {np.log2(dp[k] / dp[k+1]):.2f}")

    if a.re in ghia.U:
        print(f"\n  deviation from Ghia (1982) as the grid refines — should plateau at THEIR error, not shrink to 0")
        for n in g:
            eu = abs(runs[n]["u"] - ghia.U[a.re]).max()
            ev = abs(runs[n]["v"] - ghia.V[a.re]).max()
            print(f"  N = {n:4d}: max|u − Ghia| = {eu:.4f}   max|v − Ghia| = {ev:.4f}   "
                  f"ψmin = {runs[n]['psi']:.5f} (Ghia {ghia.VORTEX[a.re][2]:.5f})")


if __name__ == "__main__":
    main()
