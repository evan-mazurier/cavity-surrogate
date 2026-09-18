"""
Lid-driven cavity — incompressible Navier–Stokes, Chorin projection on a staggered (MAC) grid.

    ∂u/∂t + (u·∇)u = −∇p + (1/Re) ∇²u
    ∇·u = 0

Non-dimensional: box side L = 1, lid speed U = 1, so Re = U·L/ν = 1/ν.

Grid (N×N cells, spacing h = 1/N), staggered so the pressure–velocity coupling has no
checkerboard mode:
    p[i, j]   cell centres     x = (i−½)h, y = (j−½)h     shape (N+2, N+2)  (1 ghost cell each side)
    u[i, j]   x-faces          x = i·h,    y = (j−½)h     shape (N+1, N+2)  (ghost rows for top/bottom walls)
    v[i, j]   y-faces          x = (i−½)h, y = j·h        shape (N+2, N+1)  (ghost columns for side walls)
Index 0 of a ghost axis is outside the box; interior cells are 1..N.

Each step (explicit Euler in time — only the steady state is of interest):
    1. u* = u + Δt·(−convection + diffusion)        ignores pressure → ∇·u* ≠ 0
    2. ∇²p = ∇·u* / Δt                               Poisson, solved EXACTLY by eigen-decomposition
    3. u  = u* − Δt·∇p                               now ∇·u = 0 to machine precision

The Poisson solve is exact (no iteration), so the divergence check in verify/ is a true test of
the discretisation, not of an iterative tolerance.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _interp2(xs, ys, F, xt, yt):
    """Bilinear interpolation of F (len(xs) x len(ys)) onto the grid xt x yt; clamps at the edges."""
    tmp = np.empty((len(xt), len(ys)))
    for j in range(len(ys)):
        tmp[:, j] = np.interp(xt, xs, F[:, j])
    out = np.empty((len(xt), len(yt)))
    for i in range(len(xt)):
        out[i, :] = np.interp(yt, ys, tmp[i, :])
    return out


class NeumannPoisson:
    """Exact solver for the cell-centred Laplacian with homogeneous Neumann walls on an N×N grid.

    The 1-D Neumann Laplacian (ghost-cell mirror) has eigenvectors cos(πk(i+½)/N) and eigenvalues
    −(4/h²) sin²(πk/2N). The 2-D operator is separable, so   p = Q (Qᵀ f Q ⊘ (λ_i + λ_j)) Qᵀ.
    The k = 0 mode (constant pressure) is the null space: its coefficient is set to zero, which
    requires the RHS to have zero mean — true for a closed cavity up to round-off.
    """

    def __init__(self, N: int, h: float):
        k = np.arange(N)
        i = np.arange(N)
        Q = np.cos(np.pi * np.outer(i + 0.5, k) / N)           # columns = eigenvectors
        Q[:, 0] *= 1 / np.sqrt(2)                              # orthonormalise
        Q *= np.sqrt(2 / N)
        lam = -(4 / h**2) * np.sin(np.pi * k / (2 * N)) ** 2
        self.Q = Q
        denom = lam[:, None] + lam[None, :]
        denom[0, 0] = 1.0                                       # null mode, coefficient zeroed below
        self.inv = 1.0 / denom
        self.inv[0, 0] = 0.0

    def solve(self, f: np.ndarray) -> np.ndarray:
        """f: (N, N) right-hand side on the interior cells → p on the interior cells (zero mean)."""
        Q = self.Q
        fh = Q.T @ f @ Q
        return Q @ (fh * self.inv) @ Q.T


class Cavity:
    def __init__(self, N: int, Re: float, lid: float = 1.0):
        if N % 2:
            raise ValueError("N must be even so a face lies exactly on the centreline")
        self.N, self.Re, self.lid = N, float(Re), lid
        self.nu = 1.0 / Re
        self.h = 1.0 / N
        self.u = np.zeros((N + 1, N + 2))
        self.v = np.zeros((N + 2, N + 1))
        self.p = np.zeros((N + 2, N + 2))
        self.t = 0.0
        self.poisson = NeumannPoisson(N, self.h)
        self.apply_bc()

    # ------------------------------------------------------------------ boundaries
    def apply_bc(self):
        u, v, N = self.u, self.v, self.N
        u[0, :] = 0.0                     # left wall  (on the boundary)
        u[N, :] = 0.0                     # right wall
        u[:, 0] = -u[:, 1]                # bottom: no-slip via ghost mirror  (u = 0 at y = 0)
        u[:, N + 1] = 2 * self.lid - u[:, N]   # top: lid moves at +lid       (u = lid at y = 1)
        v[:, 0] = 0.0                     # bottom wall (on the boundary)
        v[:, N] = 0.0                     # top wall
        v[0, :] = -v[1, :]                # left: no-slip via ghost mirror
        v[N + 1, :] = -v[N, :]            # right

    # ------------------------------------------------------------------ one step
    def step(self, dt: float):
        u, v, N, h, nu = self.u, self.v, self.N, self.h, self.nu

        # --- u-momentum on interior x-faces i = 1..N-1, j = 1..N
        # convection, conservative form, central differences
        uc = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])                    # u at cell centres  (N, N)
        uu = uc * uc
        duu_dx = (uu[1:, :] - uu[:-1, :]) / h                      # (N-1, N) at interior x-faces
        # uv at cell corners: u averaged in y, v averaged in x  → shape (N+1, N+1) at nodes
        u_node = 0.5 * (u[:, :-1] + u[:, 1:])                       # (N+1, N+1)
        v_node = 0.5 * (v[:-1, :] + v[1:, :])                       # (N+1, N+1)
        uv = u_node * v_node
        duv_dy = (uv[1:-1, 1:] - uv[1:-1, :-1]) / h                 # (N-1, N)
        lap_u = ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
                 + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])) / h**2
        u_star = u.copy()
        u_star[1:-1, 1:-1] += dt * (-duu_dx - duv_dy + nu * lap_u)

        # --- v-momentum on interior y-faces i = 1..N, j = 1..N-1
        vc = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])                    # v at cell centres  (N, N)
        vv = vc * vc
        dvv_dy = (vv[:, 1:] - vv[:, :-1]) / h                      # (N, N-1)
        duv_dx = (uv[1:, 1:-1] - uv[:-1, 1:-1]) / h                 # (N, N-1)
        lap_v = ((v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1])
                 + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2])) / h**2
        v_star = v.copy()
        v_star[1:-1, 1:-1] += dt * (-duv_dx - dvv_dy + nu * lap_v)

        self.u, self.v = u_star, v_star
        self.apply_bc()

        # --- pressure projection
        div = self.divergence_field()                               # (N, N)
        p = self.poisson.solve(div / dt)
        self.p[1:-1, 1:-1] = p
        self.p[0, :] = self.p[1, :]; self.p[-1, :] = self.p[-2, :]
        self.p[:, 0] = self.p[:, 1]; self.p[:, -1] = self.p[:, -2]
        self.u[1:-1, 1:-1] -= dt * (p[1:, :] - p[:-1, :]) / h
        self.v[1:-1, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / h
        self.apply_bc()
        self.t += dt

    # ------------------------------------------------------------------ initial state
    def init_from(self, other: "Cavity"):
        """Start from another run's fields (any grid). The steady state is unique at these Re,
        so the initial condition only shortens the transient — it cannot change the answer."""
        if other.N == self.N:
            self.u[:] = other.u; self.v[:] = other.v; self.p[:] = other.p
        else:
            ho, N = other.h, self.N
            # u lives at x = i*h, y = (j-0.5)*h (ghost rows at j = 0 and N+1)
            xo = np.arange(other.N + 1) * ho;  yo = (np.arange(other.N + 2) - 0.5) * ho
            xt = np.arange(N + 1) * self.h;    yt = (np.arange(N + 2) - 0.5) * self.h
            self.u[:] = _interp2(xo, yo, other.u, xt, yt)
            xo = (np.arange(other.N + 2) - 0.5) * ho;  yo = np.arange(other.N + 1) * ho
            xt = (np.arange(N + 2) - 0.5) * self.h;    yt = np.arange(N + 1) * self.h
            self.v[:] = _interp2(xo, yo, other.v, xt, yt)
        self.apply_bc()
        self.t = 0.0

    # ------------------------------------------------------------------ diagnostics
    def divergence_field(self) -> np.ndarray:
        """∇·u on the N×N interior cells."""
        u, v, h = self.u, self.v, self.h
        return (u[1:, 1:-1] - u[:-1, 1:-1]) / h + (v[1:-1, 1:] - v[1:-1, :-1]) / h

    def wiggle(self) -> float:
        """Largest grid-scale (2h) oscillation in u or v: max |f[i+1] - 2 f[i] + f[i-1]| over the
        interior away from the walls. Small (< ~0.05 of lid speed) means the grid resolves the flow; a jump with Re
        means the boundary layers have become thinner than the cells."""
        b = 4                                     # skip a band along the walls: the lid corners are
        u, v = self.u[b:-b, b:-b], self.v[b:-b, b:-b]   # genuinely singular (u jumps 0 → 1 in one cell)
        wu = max(abs(u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]).max(), abs(u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]).max())
        wv = max(abs(v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]).max(), abs(v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).max())
        return max(wu, wv)

    def stable_dt(self, safety: float = 0.5) -> float:
        """Explicit-Euler limits: diffusion h²/4ν, CFL h/U, and central-convection 2ν/U²."""
        umax = max(abs(self.u[:, 1:-1]).max(), abs(self.v[1:-1, :]).max(), self.lid)
        return safety * min(self.h**2 / (4 * self.nu), self.h / umax, 2 * self.nu / umax**2)

    def run(self, tol: float = 1e-6, max_steps: int = 2_000_000, check_every: int = 100,
            dt: float | None = None, verbose: bool = True):
        """March to steady state: stop when max|Δu|/Δt (the rate of change) drops below tol.

        Returns the residual history as a list of (t, residual).
        """
        dt = dt or self.stable_dt()
        hist = []
        t0 = time.perf_counter()
        for n in range(1, max_steps + 1):
            if n % check_every == 1 or check_every == 1:
                u_prev, v_prev = self.u.copy(), self.v.copy()
            self.step(dt)
            if n % check_every == 0:
                # mean rate of change over the last check_every steps
                res = max(abs(self.u - u_prev).max(), abs(self.v - v_prev).max()) / (dt * check_every)
                hist.append((self.t, res))
                if verbose and n % (check_every * 20) == 0:
                    print(f"  t = {self.t:7.3f}  |du/dt|max = {res:.3e}  ({time.perf_counter() - t0:.1f}s)")
                if res < tol:
                    break
                if not np.isfinite(res):
                    raise FloatingPointError(f"blew up at t = {self.t:.3f}; dt = {dt:.2e} too large")
        self.dt, self.steps = dt, n
        return hist

    # ------------------------------------------------------------------ fields
    def centre_fields(self):
        """u, v, p interpolated to the N×N cell centres."""
        uc = 0.5 * (self.u[:-1, 1:-1] + self.u[1:, 1:-1])
        vc = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])
        return uc, vc, self.p[1:-1, 1:-1].copy()

    def streamfunction(self) -> np.ndarray:
        """ψ on the (N+1)×(N+1) grid nodes, ψ = 0 on the walls; u = ∂ψ/∂y integrated along each
        column of x-faces. Exact for a divergence-free field, which is why it is a nice check too."""
        N, h = self.N, self.h
        psi = np.zeros((N + 1, N + 1))
        psi[:, 1:] = np.cumsum(self.u[:, 1:-1] * h, axis=1)
        return psi

    def centrelines(self):
        """(y, u at x = ½) and (x, v at y = ½), including the wall values."""
        N, h = self.N, self.h
        yc = (np.arange(N) + 0.5) * h
        y = np.concatenate([[0.0], yc, [1.0]])
        u_mid = np.concatenate([[0.0], self.u[N // 2, 1:-1], [self.lid]])
        x = np.concatenate([[0.0], yc, [1.0]])
        v_mid = np.concatenate([[0.0], self.v[1:-1, N // 2], [0.0]])
        return (y, u_mid), (x, v_mid)

    def primary_vortex(self):
        """(x, y, ψ_min) of the primary vortex — the streamfunction minimum (clockwise circulation)."""
        psi = self.streamfunction()
        j, k = np.unravel_index(np.argmin(psi), psi.shape)
        return j * self.h, k * self.h, psi[j, k]

    def save(self, path: str):
        uc, vc, pc = self.centre_fields()
        np.savez_compressed(path, N=self.N, Re=self.Re, t=self.t, u=uc, v=vc, p=pc,
                            psi=self.streamfunction())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--re", type=float, default=100)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None, help="save fields to this .npz")
    a = ap.parse_args()

    cav = Cavity(a.n, a.re)
    print(f"cavity  Re = {a.re:g}  N = {a.n}  h = {cav.h:.4g}  dt = {cav.stable_dt():.3e}")
    t0 = time.perf_counter()
    hist = cav.run(tol=a.tol)
    print(f"steady at t = {cav.t:.3f} after {cav.steps} steps in {time.perf_counter() - t0:.1f}s; "
          f"final |du/dt|max = {hist[-1][1]:.3e}")
    print(f"max |div u| = {abs(cav.divergence_field()).max():.3e}")
    x, y, psi = cav.primary_vortex()
    print(f"primary vortex at ({x:.4f}, {y:.4f}), psi_min = {psi:.5f}")
    if a.out:
        cav.save(a.out)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
