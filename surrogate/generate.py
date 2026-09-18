"""
Generate the solver runs the surrogate learns from.

  training range  Re ∈ [50, 1500], 80 values log-spaced   (every 5th is held out → never trained on)
  extrapolation   Re ∈ {30, 40} and {1750, 2000, 2250, 2500} — outside the range, to measure how the
                  surrogate degrades where it has no right to be good

Each run: 128² grid, converged to |∂u/∂t|max < 1e-6, saved as data/runs/Re<re>.npz with the cell-centred
u, v, p, the streamfunction ψ, and its own diagnostics (steps, wall time, max |∇·u|, wiggle metric).

Speed: NW worker processes, each walking an interleaved ascending chain of Re values; every run after
the first starts from the previous converged field (continuation — the steady state is unique, so this
only shortens the transient; measured ~15%. A coarse-grid ladder for the first run cost more than it saved).
Re-running skips runs that already exist.

    .venv\\Scripts\\python surrogate\\generate.py --workers 4
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from solver import Cavity          # noqa: E402

N = 128
TOL = 1e-6
RUNS = os.path.join(ROOT, "data", "runs")

TRAIN_ALL = np.round(np.logspace(np.log10(50), np.log10(1500), 80)).astype(int)
HOLDOUT = TRAIN_ALL[2::5]                       # 16 values, never trained on
TRAIN = np.array([r for r in TRAIN_ALL if r not in HOLDOUT])
EXTRAP = np.array([30, 40, 1750, 2000, 2250, 2500])


def path(re):
    return os.path.join(RUNS, f"Re{int(re)}.npz")


def save(cav, re, wall, ladder):
    uc, vc, pc = cav.centre_fields()
    np.savez_compressed(path(re), Re=float(re), N=cav.N, u=uc, v=vc, p=pc, psi=cav.streamfunction(),
                        steps=cav.steps, t_end=cav.t, dt=cav.dt, wall=wall,
                        div=abs(cav.divergence_field()).max(), wiggle=cav.wiggle(), ladder=ladder)


def chain(args):
    """Run one ascending chain of Re values with continuation."""
    wid, res = args
    os.environ["OMP_NUM_THREADS"] = "1"
    prev = None
    for re in res:
        if os.path.exists(path(re)):
            prev = None            # can't continue from a run we didn't just compute; ladder restarts
            continue
        t0 = time.perf_counter()
        cav = Cavity(N, re)
        if prev is None:
            ladder = "from rest"       # a coarse-grid ladder was measured to cost more than it saves
        else:
            cav.init_from(prev); ladder = f"from Re{int(prev.Re)}"
        cav.run(tol=TOL, verbose=False)
        wall = time.perf_counter() - t0
        save(cav, re, wall, ladder)
        print(f"[w{wid}] Re = {re:5d}  steps = {cav.steps:7d}  t_end = {cav.t:6.2f}  wall = {wall:6.1f}s  "
              f"div = {abs(cav.divergence_field()).max():.1e}  wiggle = {cav.wiggle():.4f}  ({ladder})", flush=True)
        prev = cav
    return wid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)

    todo = np.array(sorted(set(TRAIN_ALL.tolist() + EXTRAP.tolist())))
    todo = [r for r in todo if not os.path.exists(path(r))]
    print(f"{len(todo)} runs to do ({len(TRAIN)} train + {len(HOLDOUT)} holdout + {len(EXTRAP)} extrapolation "
          f"= {len(TRAIN_ALL) + len(EXTRAP)} total), {a.workers} workers", flush=True)
    chains = [(w, todo[w::a.workers]) for w in range(a.workers)]
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        pool.map(chain, chains)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    manifest = dict(N=N, tol=TOL, train=TRAIN.tolist(), holdout=HOLDOUT.tolist(), extrap=EXTRAP.tolist())
    with open(os.path.join(ROOT, "data", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
