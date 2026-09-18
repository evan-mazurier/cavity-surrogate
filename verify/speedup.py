"""
Gate for the time-to-steady-state speed-ups (stable_dt fix + continuation):

  (a) Re = 1000, N = 128, from rest at the corrected step → must reproduce the Ghia verdict on main
      (max |Δu| 0.0030, max |Δv| 0.0124) — the discretisation is unchanged, only Δt.
  (b) The same case started from a converged 64² solution (init_from) → must land on the SAME field:
      max |u_b − u_a| at tolerance level, and in fewer steps.
  (c) Continuation in Re: Re = 1000 → 1100 starting from (a)'s field → steps needed, for the dataset plan.

    .venv\\Scripts\\python verify\\speedup.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solver import Cavity          # noqa: E402
import ghia                        # noqa: E402


def ghia_dev(cav, re):
    (y, u_mid), (x, v_mid) = cav.centrelines()
    du = np.interp(ghia.Y, y, u_mid) - ghia.U[re]
    dv = np.interp(ghia.X, x, v_mid) - ghia.V[re]
    return abs(du).max(), abs(dv).max()


def timed(cav, **kw):
    t0 = time.perf_counter()
    cav.run(verbose=False, **kw)
    return time.perf_counter() - t0


# (a) from rest, corrected dt
a = Cavity(128, 1000)
ta = timed(a)
du, dv = ghia_dev(a, 1000)
print(f"(a) from rest:        dt = {a.dt:.2e}  steps = {a.steps:7d}  t_end = {a.t:6.2f}  wall = {ta:6.1f}s  "
      f"Ghia max|Δu| = {du:.4f}  max|Δv| = {dv:.4f}   [main: 0.0030 / 0.0124, 447500 steps, 1042s]")

# (b) from a converged 64^2 solution
c = Cavity(64, 1000)
tc = timed(c)
b = Cavity(128, 1000)
b.init_from(c)
tb = timed(b)
diff = max(abs(b.u - a.u).max(), abs(b.v - a.v).max())
print(f"(b) 64² ({c.steps} steps, {tc:.0f}s) → 128²: steps = {b.steps:7d}  t_end = {b.t:6.2f}  wall = {tb:6.1f}s  "
      f"max|field_b − field_a| = {diff:.2e}   total wall = {tb + tc:.0f}s  ({(tb + tc) / ta:.2f}× of (a))")

# (c) continuation in Re
d = Cavity(128, 1100)
d.init_from(a)
td = timed(d)
print(f"(c) Re 1000 → 1100 by continuation: steps = {d.steps:7d}  t_end = {d.t:6.2f}  wall = {td:6.1f}s  "
      f"wiggle = {d.wiggle():.4f}")
print(f"\nGATE: {'PASS' if du < 0.0035 and dv < 0.013 and diff < 1e-4 else 'FAIL'}")
