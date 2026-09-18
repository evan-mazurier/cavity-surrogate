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

Solver — all on `main`, reproducible with the commands in `verify/`
- [x] Discrete divergence of the corrected velocity field ≈ machine zero every step — **1e-13 to 1e-14**
- [x] Converges to steady state (residual history) at Re = 100, 400, 1000
- [x] Grid convergence 16² → 32² → 64² → 128² at the expected order — **observed order 1.9–2.3** (second-order scheme)
- [x] Centreline profiles match Ghia et al. (1982) at Re = 100, 400, 1000 — **max deviation 0.5% / 0.5% / 1.2% of lid speed**
- [x] Primary vortex centre matches published location — **exact to the grid at Re = 100 and 1000**

Surrogate
- [ ] Held-out Re: relative L2 error of the predicted field vs the solver
- [ ] Predicted field respects ∇·u ≈ 0 (a constraint the model was never told)
- [ ] Error vs Re, with the training range marked — honest inside, degrading outside

## Solver results

![Re = 1000](figures/ghia_Re1000_N128.png)

| Re | grid | max ∣∇·u∣ | max ∣Δu∣ | max ∣Δv∣ | vortex centre (solver / Ghia) | ψ_min (solver / Ghia) |
|---:|---:|---:|---:|---:|---|---|
| 100 | 128² | 3e-13 | 0.0049 | 0.0091 | (0.6172, 0.7344) / (0.6172, 0.7344) | −0.10344 / −0.10342 |
| 400 | 128² | 5e-14 | 0.0020 | 0.0051 † | (0.5547, 0.6094) / (0.5547, 0.6055) | −0.11349 / −0.11391 |
| 1000 | 128² | 2e-14 | 0.0030 | 0.0124 | (0.5312, 0.5625) / (0.5313, 0.5625) | −0.11751 / −0.11793 |

Deviations are against the 17 + 17 tabulated centreline points, as a fraction of the lid speed.

Two things the checks turned up that a "looks right" pass would have missed:

- **The deviation from Ghia plateaus as the grid refines** (Re = 100: 0.38% at 64², 0.49% at 128²) while the
  solver's own grid-to-grid change keeps shrinking 4× per halving of h. The residual gap is therefore the 1982
  reference's own discretisation error, not this solver's — consistent with later spectral results (Botella &
  Peyret 1998) that differ from Ghia by a similar amount.
- **† One printed reference value is wrong.** Ghia's Table II, Re = 400, v at x = 0.9063 = −0.23827 is
  non-monotone against its own neighbours (−0.228 at 0.9453, −0.450 at 0.8594) and 15% off a solver that matches
  the other 16 points to 0.2%. It is reproduced identically in every copy of the table found, so it is in the
  paper. It stays verbatim in `verify/ghia.py`, flagged `SUSPECT`, drawn as a grey × in the figures and excluded
  from the verdict line — never silently corrected.

![Re = 400](figures/ghia_Re400_N128.png)

## Layout

```
solver.py      the Navier–Stokes solver (numpy)
verify/        the checks above, each a script that prints its numbers
  ghia.py               the 1982 reference tables, verbatim, with the suspect point flagged
  compare_ghia.py       run + overlay + verdict + figure        --re 100|400|1000 --n 128
  grid_convergence.py   observed order of accuracy               --re 100 --grids 16 32 64 128
figures/       the committed figures the README shows
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
