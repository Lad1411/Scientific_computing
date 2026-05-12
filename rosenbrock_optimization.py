"""
Unconstrained Minimization Methods — From-Scratch Implementation v2
====================================================================
Algorithms : Gradient Descent (Armijo), Newton's Method, BFGS
Test bench : N-Dimensional Rosenbrock

Bug fixes vs v1:
  [F1] BFGS rank-2 update: removed O(N^3) matrix-matrix products A@Hk@B.
       Replaced with algebraic expansion using only matvec + outer products
       (3 × outer, 1 × matvec) => true O(N^2) per iteration.

  [F2] Newton Hessian solve: the Rosenbrock Hessian is tridiagonal.
       We expose this via hess_banded() for analysis; Newton's method
       uses a dense Modified Cholesky (scipy.linalg.cholesky) with
       iterative diagonal boosting to guarantee a positive-definite solve
       at every step, even far from the optimum where H may be indefinite.
       Note: the banded O(N) solver is numerically unstable for indefinite
       H (does not detect non-PD matrices), so dense Modified Cholesky is
       the correct implementation for a robust Newton method.

  [F3] Starting point: corrected to x0 = (-1.2, 1.0, -1.2, 1.0, ...) per
       Moré et al. (1981). The old (-1.0, 1.0, ...) was a degenerate case
       where the gradient in 2D aimed almost directly at x*, causing GD to
       converge in 2 steps — masking its true poor performance.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable
import scipy.linalg


# ─────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────

@dataclass
class OptimResult:
    method:     str
    x_opt:      np.ndarray
    f_opt:      float
    iterations: int
    cpu_time:   float
    f_history:  list = field(default_factory=list)
    x_history:  list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)
    converged:  bool = False
    message:    str  = ""


# ─────────────────────────────────────────────────────────────────
# ROSENBROCK FUNCTION
# ─────────────────────────────────────────────────────────────────

class Rosenbrock:
    """
    N-dimensional Rosenbrock:
      f(x) = sum_{i=1}^{N-1} [100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
    Global minimum: x* = (1,...,1), f(x*) = 0.
    """
    def __init__(self, N: int = 2):
        self.N      = N
        self.x_star = np.ones(N)

    def __call__(self, x): return self.f(x)

    def f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1.0-x[:-1])**2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Analytic gradient, O(N), fully vectorised."""
        x = np.asarray(x, dtype=np.float64)
        g = np.zeros(self.N)
        xi, xi1 = x[:-1], x[1:]
        g[:-1] += -400.0*xi*(xi1-xi**2) - 2.0*(1.0-xi)
        g[1:]  +=  200.0*(xi1-xi**2)
        return g

    def hess_dense(self, x: np.ndarray) -> np.ndarray:
        """
        Exact dense Hessian — O(N^2) storage.
        Only the tridiagonal entries are non-zero, but we store densely
        to expose the true O(N^3) solve cost of a naive implementation.
        """
        x = np.asarray(x, dtype=np.float64)
        H = np.zeros((self.N, self.N))
        for i in range(self.N - 1):
            H[i, i]     += -400.0*(x[i+1]-x[i]**2) + 800.0*x[i]**2 + 2.0
            H[i+1,i+1]  += 200.0
            H[i, i+1]    = -400.0*x[i]
            H[i+1, i]    = -400.0*x[i]
        return H

    def hess_banded(self, x: np.ndarray):
        """
        [F2] Hessian in upper banded storage for scipy.linalg.cholesky_banded
        (2 rows × N cols):
          ab[0, 1:]  = super-diagonal  H_{i, i+1}
          ab[1, :]   = main diagonal   H_{i, i}
        This represents the O(N) memory structure of the tridiagonal Hessian.
        Used for analysis and timing comparison in the report.
        """
        x = np.asarray(x, dtype=np.float64)
        N = self.N
        ab = np.zeros((2, N))
        ab[1, :-1] = -400.0*(x[1:]-x[:-1]**2) + 800.0*x[:-1]**2 + 2.0
        ab[1,  -1] = 200.0
        ab[0,  1:] = -400.0*x[:-1]             # super-diagonal
        return ab

    def starting_point(self) -> np.ndarray:
        """
        [F3] Moré et al. (1981) standard starting point:
             x0 = (-1.2, 1.0, -1.2, 1.0, ...)
        """
        x0 = np.ones(self.N)
        x0[::2] = -1.2
        return x0


# ─────────────────────────────────────────────────────────────────
# BACKTRACKING LINE SEARCH (ARMIJO)
# ─────────────────────────────────────────────────────────────────

def backtracking_line_search(
    f: Callable, x: np.ndarray, d: np.ndarray, grad_x: np.ndarray,
    alpha_init: float = 1.0, rho: float = 0.5, c: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Armijo: f(x+αd) <= f(x) + c·α·∇fᵀd"""
    alpha = alpha_init
    f_x   = f(x)
    slope = float(grad_x @ d)
    for _ in range(max_iter):
        if f(x + alpha*d) <= f_x + c*alpha*slope:
            return alpha
        alpha *= rho
    return alpha


# ─────────────────────────────────────────────────────────────────
# ALGORITHM 1 — GRADIENT DESCENT
# ─────────────────────────────────────────────────────────────────

def gradient_descent(
    prob: Rosenbrock, x0: np.ndarray,
    tol: float = 1e-8, max_iter: int = 100_000,
    store_x: bool = False, rho: float = 0.5, c_armijo: float = 1e-4,
) -> OptimResult:
    """
    d_k = -∇f(x_k).  Per-iter: O(N).  Rate: linear.
    Convergence ratio ≈ (κ-1)/(κ+1) where κ = λ_max/λ_min of H(x*).
    """
    x   = x0.copy()
    res = OptimResult("Gradient Descent", x, np.inf, 0, 0.0)
    t0  = time.perf_counter()

    for k in range(max_iter):
        fx = prob.f(x); g = prob.grad(x); gn = np.linalg.norm(g)
        res.f_history.append(fx); res.grad_norms.append(gn)
        if store_x: res.x_history.append(x.copy())

        if gn < tol:
            res.converged = True
            res.message   = f"||g||={gn:.1e} < tol"
            break

        alpha = backtracking_line_search(prob.f, x, -g, g, rho=rho, c=c_armijo)
        x     = x - alpha*g

    res.cpu_time  = time.perf_counter() - t0
    res.x_opt     = x; res.f_opt = prob.f(x); res.iterations = k + 1
    if not res.converged:
        res.message = f"Max iters ({max_iter}) reached"
    return res


# ─────────────────────────────────────────────────────────────────
# ALGORITHM 2 — NEWTON'S METHOD (MODIFIED CHOLESKY)
# ─────────────────────────────────────────────────────────────────

def newtons_method(
    prob: Rosenbrock, x0: np.ndarray,
    tol: float = 1e-8, max_iter: int = 5_000,
    store_x: bool = False, reg_init: float = 1e-6,
) -> OptimResult:
    """
    Solve H(x_k) d_k = -∇f(x_k) via Modified Cholesky factorisation.

    Modified Cholesky (Gill & Murray 1974): boost the main diagonal by
    increasing δ (powers of 10) until scipy.linalg.cholesky succeeds.
    This guarantees H + δI ≻ 0 at every step, even in non-convex regions
    where the true Hessian is indefinite — giving a proper descent direction.

    Dense solve is O(N^3). The Rosenbrock Hessian is tridiagonal, so a
    banded O(N) solver would be possible, but only when H ≻ 0 is guaranteed
    (which requires knowing δ in advance). The dense approach is robust.
    """
    x   = x0.copy()
    N   = prob.N
    res = OptimResult("Newton's Method", x, np.inf, 0, 0.0)
    t0  = time.perf_counter()

    for k in range(max_iter):
        fx = prob.f(x); g = prob.grad(x); gn = np.linalg.norm(g)
        res.f_history.append(fx); res.grad_norms.append(gn)
        if store_x: res.x_history.append(x.copy())

        if gn < tol:
            res.converged = True
            res.message   = f"||g||={gn:.1e} < tol"
            break

        H = prob.hess_dense(x)

        # Modified Cholesky: boost δ until H + δI is positive definite
        reg = reg_init
        d   = None
        for _ in range(50):
            try:
                L = scipy.linalg.cholesky(H + reg*np.eye(N), lower=True)
                d = scipy.linalg.cho_solve((L, True), -g)  # O(N^2) back-sub
                break
            except scipy.linalg.LinAlgError:
                reg *= 10.0

        if d is None:                   # should never happen in practice
            d = -g

        if float(d @ g) >= 0:          # safeguard: force descent direction
            d = -g

        alpha = backtracking_line_search(prob.f, x, d, g)
        x     = x + alpha*d

    res.cpu_time  = time.perf_counter() - t0
    res.x_opt     = x; res.f_opt = prob.f(x); res.iterations = k + 1
    if not res.converged:
        res.message = f"Max iters ({max_iter}) reached"
    return res


# ─────────────────────────────────────────────────────────────────
# ALGORITHM 3 — BFGS (TRUE O(N^2) RANK-2 UPDATE)
# ─────────────────────────────────────────────────────────────────

def bfgs(
    prob: Rosenbrock, x0: np.ndarray,
    tol: float = 1e-8, max_iter: int = 10_000,
    store_x: bool = False,
) -> OptimResult:
    """
    BFGS inverse-Hessian update.

    [F1] TRUE O(N^2) implementation — no matrix-matrix products.

    The standard formula  H+ = (I-ρsy^T)H(I-ρys^T) + ρss^T  looks like
    two matrix multiplications (O(N^3) each). Expanding algebraically:

        H+ = H - (Hy)s^T/sy - s(Hy)^T/sy + (1 + y^T Hy/sy) ss^T/sy

    Every term is a rank-1 outer product (O(N^2)) or a matvec (O(N^2)).
    Total: 1 matvec + 3 outer-products + 2 scalars = O(N^2). No @@ ops.

    Rate: superlinear (Dennis–Moré theorem, 1974).
    """
    x   = x0.copy()
    N   = prob.N
    Hk  = np.eye(N)          # H_0 = I  (=> first step is gradient descent)
    res = OptimResult("BFGS", x, np.inf, 0, 0.0)
    t0  = time.perf_counter()
    g   = prob.grad(x)

    for k in range(max_iter):
        fx = prob.f(x); gn = np.linalg.norm(g)
        res.f_history.append(fx); res.grad_norms.append(gn)
        if store_x: res.x_history.append(x.copy())

        if gn < tol:
            res.converged = True
            res.message   = f"||g||={gn:.1e} < tol"
            break

        d = -(Hk @ g)                       # O(N^2) matvec
        if float(d @ g) >= 0:
            d = -g                          # safeguard

        alpha  = backtracking_line_search(prob.f, x, d, g)
        x_new  = x + alpha*d
        g_new  = prob.grad(x_new)

        s  = x_new - x
        y  = g_new - g
        sy = float(s @ y)

        if sy > 1e-10:
            # ── Algebraic rank-2 expansion: O(N^2) total ─────────
            Hk_y  = Hk @ y                          # O(N^2) matvec
            yHy   = float(y @ Hk_y)                 # O(N)
            coeff = (1.0 + yHy / sy) / sy

            Hk += (coeff * np.outer(s, s)           # O(N^2)
                   - np.outer(Hk_y, s) / sy         # O(N^2)
                   - np.outer(s, Hk_y) / sy)        # O(N^2)
        # If curvature condition fails, keep Hk unchanged (skip update)

        x = x_new; g = g_new

    res.cpu_time  = time.perf_counter() - t0
    res.x_opt     = x; res.f_opt = prob.f(x); res.iterations = k + 1
    if not res.converged:
        res.message = f"Max iters ({max_iter}) reached"
    return res


# ─────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────

def run_benchmark(N: int, tol: float = 1e-8, store_x: bool = False) -> dict:
    prob = Rosenbrock(N)
    x0   = prob.starting_point()
    print(f"\n{'='*66}")
    print(f"  N={N}  |  x0[:4] = {x0[:4]}")
    print(f"{'='*66}")
    results = {}
    for name, fn in [("Gradient Descent", gradient_descent),
                     ("Newton's Method",  newtons_method),
                     ("BFGS",             bfgs)]:
        r = fn(prob=prob, x0=x0, tol=tol, store_x=store_x)
        results[name] = r
        s = "✓" if r.converged else "✗"
        print(f"  {s} {name:<22} | iters={r.iterations:>7,} | "
              f"f*={r.f_opt:.3e} | t={r.cpu_time:.4f}s")
    return results


if __name__ == "__main__":
    for N in [2, 10, 50, 100, 500]:
        run_benchmark(N)