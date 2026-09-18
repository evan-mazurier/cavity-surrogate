# cavity-surrogate

A validated 2D Navier–Stokes solver for the lid-driven cavity, and a neural surrogate trained on it.

The lid-driven cavity is the canonical benchmark for incompressible flow solvers: a square box of fluid,
three fixed walls, a top wall sliding at constant speed. One parameter — the Reynolds number — and a
published reference solution (Ghia, Ghia & Shin, 1982) that every CFD code is checked against.

This repo does two things, in order:

1. **Solver** — writes the incompressible Navier–Stokes equations down, discretises them (Chorin
   projection on a staggered grid, pure numpy) and proves the implementation right against analytic
   limits and the Ghia reference data.
2. **Surrogate** — trains a small model on many solver runs so that, for a new Reynolds number, the
   whole flow field is predicted in milliseconds instead of minutes — with its error against the real
   solver measured and shown, inside and outside the training range.

The method is the point: derive → verify → build. Nothing here is trusted because it looks right.

## Verification ladder

Solver
- [ ] Discrete divergence of the corrected velocity field ≈ machine zero every step
- [ ] Converges to steady state (residual history) at Re = 100
- [ ] Grid convergence 32² → 64² → 128² → 256² at the expected order
- [ ] Centreline profiles match Ghia et al. (1982) at Re = 100, 400, 1000
- [ ] Primary vortex centre matches published location

Surrogate
- [ ] Held-out Re: relative L2 error of the predicted field vs the solver
- [ ] Predicted field respects ∇·u ≈ 0 (a constraint the model was never told)
- [ ] Error vs Re, with the training range marked — honest inside, degrading outside

## Layout

```
solver.py      the Navier–Stokes solver (numpy)
verify/        the checks above, each a script that prints its numbers
data/          solver runs used to train the surrogate (generated, not committed)
surrogate/     model, training, evaluation
web/           the interactive Re-slider demo (last)
```

## Running

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python solver.py --re 100 --n 64
```

## Reference

Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the
Navier–Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
