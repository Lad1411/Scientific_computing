"""
Visualization Script v3
=======================
Key fixes vs v2:
  [V1] GD subsampling: uniform step (every N//300) washes out zigzag because
       consecutive steps are nearly collinear after averaging. Replaced with
       ADAPTIVE sampling that keeps direction-change points (turning vertices),
       preserving the true zigzag character of GD.
  [V2] Multiple starting points per N=2 (inspired by reference implementation):
       tests Normal (-1.5,1), Hard (-1.2,1), Far (4,4), Flat (0,0).
  [V3] Plot ALL history for Newton/BFGS (they have so few points that
       subsampling makes no sense).
  [V4] Status label on plot title: Global / Local Min / No Convergence.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, '/home/claude')
from rosenbrock_optimization import (
    Rosenbrock, gradient_descent, newtons_method, bfgs
)

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#0f1117','axes.facecolor':'#161b22',
    'axes.edgecolor':'#30363d','axes.labelcolor':'#e6edf3',
    'xtick.color':'#8b949e','ytick.color':'#8b949e',
    'text.color':'#e6edf3','grid.color':'#21262d','grid.linewidth':0.6,
    'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
    'axes.labelsize':10,'legend.fontsize':8,
    'legend.facecolor':'#161b22','legend.edgecolor':'#30363d',
    'lines.linewidth':1.6,
})
COLORS  = {'Gradient Descent':'#f97316', "Newton's Method":'#22d3ee', 'BFGS':'#a78bfa'}
MARKERS = {'Gradient Descent':'o', "Newton's Method":'s', 'BFGS':'D'}
METHODS = [('Gradient Descent', gradient_descent),
           ("Newton's Method",  newtons_method),
           ('BFGS',             bfgs)]
ALL_N   = [2, 10, 50, 100, 500]
MAX_ITER = {
    'Gradient Descent': {2:30000, 10:40000, 50:60000, 100:80000, 500:100000},
    "Newton's Method":  {2:500,   10:500,   50:1000,  100:2000,  500:5000},
    'BFGS':             {2:500,   10:1000,  50:2000,  100:3000,  500:6000},
}

# ── Helpers ───────────────────────────────────────────────────────

def run_method(name, fn, prob, x0, store_x=False, tol=1e-8):
    mi = MAX_ITER[name].get(prob.N, 10000)
    r  = fn(prob=prob, x0=x0, tol=tol, max_iter=mi, store_x=store_x)
    T  = r.cpu_time
    t_hist = np.linspace(0, T, len(r.f_history))
    s  = "✓" if r.converged else "✗"
    print(f"    {s} {name:<22} | iters={r.iterations:>7,} | "
          f"f*={r.f_opt:.2e} | t={r.cpu_time:.3f}s")
    return t_hist, np.array(r.f_history), r


def adaptive_subsample(traj, n_target=600):
    """
    [V1] Keep the n_target points with the LARGEST direction change.
    This preserves zigzag vertices instead of averaging them away.
    Always keeps first and last point.
    Falls through unchanged if traj is already small enough.
    """
    if len(traj) <= n_target:
        return traj, np.arange(len(traj))
    dx     = np.diff(traj, axis=0)                          # (K-1, 2)
    norms  = np.linalg.norm(dx, axis=1, keepdims=True) + 1e-30
    dirs   = dx / norms                                     # unit directions
    # cos(angle between consecutive steps) — low value = big turn
    cosines = np.sum(dirs[:-1] * dirs[1:], axis=1)          # (K-2,)
    turns   = 1.0 - cosines                                 # high = sharp turn
    turns   = np.concatenate([[1.0], turns, [1.0]])         # pad to K-1 length
    # Also weight by step size so tiny noisy steps aren't selected
    step_sizes = norms[:, 0]
    step_sizes = np.concatenate([[step_sizes[0]], step_sizes])
    score   = turns * step_sizes
    top_idx = np.argsort(score)[-n_target:]
    idx     = np.sort(np.unique(np.concatenate([[0, len(traj)-1], top_idx])))
    return traj[idx], idx


def status_label(r, prob):
    """Classify convergence outcome."""
    if not r.converged:
        return "No Conv."
    if r.f_opt < 1e-6:
        return "Global ✓"
    if 3.0 < r.f_opt < 5.0:
        return "Local Min ⚠"
    return f"f*={r.f_opt:.2e}"


def style_ax(ax, yscale='log'):
    ax.set_facecolor('#161b22')
    ax.grid(True, alpha=0.2)
    if yscale:
        ax.set_yscale(yscale)


def contour_slice(prob, xlim=(-2.3,2.0), ylim=(-0.8,2.4), res=350):
    xr = np.linspace(*xlim, res); yr = np.linspace(*ylim, res)
    X, Y = np.meshgrid(xr, yr)
    Z = np.zeros_like(X)
    base = np.ones(prob.N)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = base.copy(); pt[0] = X[i,j]; pt[1] = Y[i,j]
            Z[i,j] = prob.f(pt)
    return X, Y, np.log10(Z + 1e-10)


def draw_traj_on_ax(ax, name, traj, color, label, marker='o'):
    """[V1] Use adaptive subsampling for GD; plot ALL for Newton/BFGS."""
    if name == 'Gradient Descent':
        pts, _ = adaptive_subsample(traj, n_target=700)
    else:
        pts = traj                           # Newton/BFGS: always few points
    ax.plot(pts[:,0], pts[:,1], '-', color=color, alpha=0.8, lw=1.3,
            label=label)
    # Mark start and end
    ax.plot(pts[0,0],  pts[0,1],  'o', color=color, ms=6,
            markeredgecolor='white', markeredgewidth=0.7, zorder=8)
    ax.plot(pts[-1,0], pts[-1,1], '^', color=color, ms=7,
            markeredgecolor='white', markeredgewidth=0.7, zorder=9)
    # For GD also mark a few intermediate turning points so zigzag is visible
    if name == 'Gradient Descent' and len(pts) > 10:
        sample = pts[::max(1, len(pts)//40)]
        ax.plot(sample[:,0], sample[:,1], '.', color=color,
                ms=2.5, alpha=0.5, zorder=7)


# ══════════════════════════════════════════════════════════════════
# SCENARIO 1 — Multi-start 2D trajectories
# ══════════════════════════════════════════════════════════════════

# [V2] Multiple starting points like the reference implementation
STARTS_2D = {
    "Normal  x₀=(-1.5, 1.0)": np.array([-1.5, 1.0]),
    "Hard    x₀=(-1.2, 1.0)": np.array([-1.2, 1.0]),
    "Far     x₀=(4.0, 4.0)":  np.array([4.0,  4.0]),
    "Flat    x₀=(0.0, 0.0)":  np.array([0.0,  0.0]),
}

def plot_scenario1_multistart():
    print("\n=== Scenario 1: Multi-start 2D trajectories ===")
    prob   = Rosenbrock(N=2)
    n_cases = len(STARTS_2D)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0f1117')
    axes_flat = axes.flatten()

    # Build one contour for all subplots
    X, Y, Zlog = contour_slice(prob, xlim=(-2.5, 5.5), ylim=(-1.5, 5.5), res=350)

    for ax, (case_name, x0) in zip(axes_flat, STARTS_2D.items()):
        print(f"\n  Case: {case_name}")
        style_ax(ax, yscale=None)

        # Dynamic axis limits
        xlo = min(-2.2, x0[0]-0.5); xhi = max(2.0, x0[0]+0.5)
        ylo = min(-0.8, x0[1]-0.5); yhi = max(2.2, x0[1]+0.5)

        # Clip contour to axis window
        mask_x = (X[0] >= xlo) & (X[0] <= xhi)
        mask_y = (Y[:,0] >= ylo) & (Y[:,0] <= yhi)
        Xc = X[np.ix_(mask_y, mask_x)]
        Yc = Y[np.ix_(mask_y, mask_x)]
        Zc = Zlog[np.ix_(mask_y, mask_x)]

        cf = ax.contourf(Xc, Yc, Zc, levels=40, cmap='inferno', alpha=0.75)
        ax.contour(Xc, Yc, Zc, levels=20, colors='white', alpha=0.10, linewidths=0.35)
        ax.plot(1, 1, '*', color='#fbbf24', ms=15, zorder=10,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.plot(x0[0], x0[1], 'X', color='white', ms=11, zorder=9,
                markeredgecolor='#30363d')

        results = {}
        for name, fn in METHODS:
            _, _, r = run_method(name, fn, prob, x0, store_x=True)
            results[name] = r
            if r.x_history:
                traj = np.array(r.x_history)
                sl   = status_label(r, prob)
                lbl  = f"{name} ({r.iterations:,} it) [{sl}]"
                draw_traj_on_ax(ax, name, traj, COLORS[name], lbl)

        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_title(case_name, fontsize=10)
        ax.legend(loc='upper left', framealpha=0.85, fontsize=7.5)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Rosenbrock 2D — Trajectories (4 starting points)\n'
                 'GD zigzag shown via adaptive direction-change sampling',
                 fontsize=12, y=1.01, color='#e6edf3')
    fig.tight_layout()
    out = 'outputs/scenario1_multistart_2D.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"\n  Saved {out}")


def plot_scenario1_normal_zoom():
    """Single 2-panel figure for the Normal case: full + zoom."""
    print("\n=== Scenario 1b: Normal case full+zoom ===")
    prob = Rosenbrock(N=2)
    x0   = np.array([-1.5, 1.0])
    results = {}
    for name, fn in METHODS:
        _, _, r = run_method(name, fn, prob, x0, store_x=True)
        results[name] = r

    X, Y, Zlog = contour_slice(prob, xlim=(-2.3, 2.0), ylim=(-0.8, 2.4), res=400)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f1117')

    for ax_idx, (ax, (xlim, ylim, title)) in enumerate(zip(axes, [
        ((-2.3, 2.0), (-0.8, 2.4), 'Full Path  (Normal start)'),
        ((-0.4, 1.5), ( 0.1, 1.6), 'Zoomed Near Valley'),
    ])):
        style_ax(ax, yscale=None)
        mask_x = (X[0]  >= xlim[0]) & (X[0]  <= xlim[1])
        mask_y = (Y[:,0] >= ylim[0]) & (Y[:,0] <= ylim[1])
        Xc = X[np.ix_(mask_y, mask_x)]
        Yc = Y[np.ix_(mask_y, mask_x)]
        Zc = Zlog[np.ix_(mask_y, mask_x)]

        cf = ax.contourf(Xc, Yc, Zc, levels=40, cmap='inferno', alpha=0.75)
        ax.contour(Xc, Yc, Zc, levels=20, colors='white', alpha=0.11, linewidths=0.35)
        ax.plot(1, 1, '*', color='#fbbf24', ms=16, zorder=10,
                markeredgecolor='white', markeredgewidth=0.8,
                label='Global min $(1,1)$')
        ax.plot(x0[0], x0[1], 'X', color='white', ms=12, zorder=9,
                label='Start $(-1.5,1.0)$', markeredgecolor='#30363d')
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(title)

        for name, r in results.items():
            if not r.x_history: continue
            traj = np.array(r.x_history)
            sl   = status_label(r, prob)
            lbl  = f'{name} ({r.iterations:,} it) [{sl}]'
            draw_traj_on_ax(ax, name, traj, COLORS[name], lbl)

        ax.legend(loc='upper left', framealpha=0.85, fontsize=8)
        ax.grid(True, alpha=0.2)

    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label('log₁₀ f(x)', color='#e6edf3', labelpad=8)
    cb.ax.yaxis.set_tick_params(color='#8b949e')

    plt.suptitle('Rosenbrock 2D — GD·Newton·BFGS  x₀=(-1.5, 1.0)\n'
                 '[V1] Adaptive sampling preserves GD zigzag character',
                 fontsize=12, y=1.01, color='#e6edf3')
    fig.tight_layout(rect=[0, 0, 0.91, 1])
    out = 'outputs/scenario1_trajectories_N2.png'
    fig.savefig(out, dpi=155, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════
# SCENARIO 1c — Projected trajectories all N (fixed subsampling)
# ══════════════════════════════════════════════════════════════════

def plot_scenario1_all_N():
    print("\n=== Scenario 1c: Projected trajectories all N ===")
    fig = plt.figure(figsize=(16, 6.5 * len(ALL_N)))
    fig.patch.set_facecolor('#0f1117')
    gs  = GridSpec(len(ALL_N), 2, figure=fig, hspace=0.45, wspace=0.08)

    for row, N in enumerate(ALL_N):
        prob = Rosenbrock(N)
        x0   = prob.starting_point()
        print(f"\n  N={N}")
        results = {}
        for name, fn in METHODS:
            _, _, r = run_method(name, fn, prob, x0, store_x=True)
            results[name] = r

        X, Y, Zlog = contour_slice(prob, res=280)

        for col, (xlim, ylim, sfx) in enumerate([
            ((-2.3, 1.9), (-0.8, 2.3), 'Full'),
            ((-0.3, 1.5), ( 0.0, 1.6), 'Zoom'),
        ]):
            ax = fig.add_subplot(gs[row, col])
            style_ax(ax, yscale=None)
            mask_x = (X[0]  >= xlim[0]) & (X[0]  <= xlim[1])
            mask_y = (Y[:,0] >= ylim[0]) & (Y[:,0] <= ylim[1])
            Xc = X[np.ix_(mask_y, mask_x)]
            Yc = Y[np.ix_(mask_y, mask_x)]
            Zc = Zlog[np.ix_(mask_y, mask_x)]

            ax.contourf(Xc, Yc, Zc, levels=35, cmap='inferno', alpha=0.75)
            ax.contour( Xc, Yc, Zc, levels=18, colors='white', alpha=0.09, linewidths=0.3)
            ax.plot(1, 1, '*', color='#fbbf24', ms=13, zorder=10,
                    markeredgecolor='white', markeredgewidth=0.7)
            ax.plot(x0[0], x0[1], 'X', color='white', ms=9, zorder=9,
                    markeredgecolor='#30363d')
            ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            note = '' if N == 2 else ' [proj $x_1$-$x_2$]'
            ax.set_title(f'N={N} — {sfx}{note}', fontsize=10)

            for name, r in results.items():
                if not r.x_history: continue
                traj = np.array(r.x_history)
                sl   = status_label(r, prob)
                lbl  = f'{name} ({r.iterations:,}) [{sl}]' if col == 0 else name
                draw_traj_on_ax(ax, name, traj, COLORS[name], lbl)

            if col == 0:
                ax.legend(loc='upper left', fontsize=7, framealpha=0.85)
            ax.grid(True, alpha=0.15)

    plt.suptitle('Rosenbrock — Trajectories All N  '
                 '(projected onto $x_1$–$x_2$)',
                 fontsize=13, y=1.002, color='#e6edf3')
    out = 'outputs/scenario1_trajectories_ALL.png'
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"\n  Saved {out}")


# ══════════════════════════════════════════════════════════════════
# SCENARIO 2 — Scalability all N
# ══════════════════════════════════════════════════════════════════

def draw_panels(ax_t, ax_i, results, N):
    for name, (t_hist, f_hist, r) in results.items():
        c     = COLORS[name]
        f_pos = np.maximum(f_hist, 1e-30)
        iters = np.arange(1, len(f_hist)+1)
        sl    = status_label(r, Rosenbrock(N))
        lbl   = f'{name} ({r.iterations:,} it) [{sl}]'
        ax_t.plot(t_hist, f_pos, '-', color=c, lw=1.6, alpha=0.9, label=lbl)
        ax_t.plot(t_hist[-1], f_pos[-1], MARKERS[name], color=c, ms=7,
                  markeredgecolor='white', markeredgewidth=0.7, zorder=9)
        ax_i.plot(iters, f_pos, '-', color=c, lw=1.6, alpha=0.9, label=lbl)
        ax_i.plot(iters[-1], f_pos[-1], MARKERS[name], color=c, ms=7,
                  markeredgecolor='white', markeredgewidth=0.7, zorder=9)
    for ax in (ax_t, ax_i):
        ax.axhline(1e-8, color='#fbbf24', lw=0.9, ls='--', alpha=0.7, label='tol=1e-8')
        ax.set_ylabel('$f(x_k)$ (log)')
    ax_t.set_xlabel('CPU time (s)')
    ax_i.set_xlabel('Iteration $k$')
    ax_t.set_title(f'f vs Time  (N={N})')
    ax_i.set_title(f'f vs Iterations  (N={N})')
    ax_t.legend(loc='upper right', framealpha=0.85, fontsize=7.5)
    ax_i.legend(loc='upper right', framealpha=0.85, fontsize=7.5)


def plot_scenario2_all():
    print("\n=== Scenario 2: Scalability all N ===")
    # Individual figures
    for N in ALL_N:
        print(f"\n  N={N}")
        prob = Rosenbrock(N); x0 = prob.starting_point()
        results = {}
        for name, fn in METHODS:
            t, f, r = run_method(name, fn, prob, x0)
            results[name] = (t, f, r)
        fig, (ax_t, ax_i) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0f1117')
        style_ax(ax_t); style_ax(ax_i)
        draw_panels(ax_t, ax_i, results, N)
        plt.suptitle(f'Scalability N={N}  (GD · Newton · BFGS)',
                     fontsize=12, y=1.03, color='#e6edf3')
        fig.tight_layout()
        out = f'outputs/scenario2_scalability_N{N}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f1117')
        plt.close()
        print(f"  Saved {out}")

    # Combined figure
    print("\n  Combined figure...")
    fig = plt.figure(figsize=(14, 4.8*len(ALL_N)))
    fig.patch.set_facecolor('#0f1117')
    gs  = GridSpec(len(ALL_N), 2, figure=fig, hspace=0.60, wspace=0.32)
    for row, N in enumerate(ALL_N):
        prob = Rosenbrock(N); x0 = prob.starting_point(); results = {}
        for name, fn in METHODS:
            t, f, r = run_method(name, fn, prob, x0)
            results[name] = (t, f, r)
        ax_t = fig.add_subplot(gs[row, 0]); ax_i = fig.add_subplot(gs[row, 1])
        style_ax(ax_t); style_ax(ax_i)
        draw_panels(ax_t, ax_i, results, N)
        ax_t.annotate(f'N={N}', xy=(-0.20, 0.5), xycoords='axes fraction',
                      fontsize=13, color='#e6edf3', fontweight='bold',
                      va='center', rotation=90)
    plt.suptitle('Scalability All N  (GD · Newton · BFGS)',
                 fontsize=13, y=1.002, color='#e6edf3')
    out = 'outputs/scenario2_scalability_ALL.png'
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════
# SCENARIO 3 — Convergence grid
# ══════════════════════════════════════════════════════════════════

def plot_scenario3():
    print("\n=== Scenario 3: Convergence grid ===")
    fig, axes = plt.subplots(1, len(ALL_N), figsize=(5*len(ALL_N), 5))
    fig.patch.set_facecolor('#0f1117')
    for ax, N in zip(axes, ALL_N):
        style_ax(ax)
        ax.set_title(f'N = {N}', fontsize=12)
        ax.set_xlabel('Iteration $k$')
        if N == ALL_N[0]: ax.set_ylabel('$f(x_k)$ (log)')
        prob = Rosenbrock(N); x0 = prob.starting_point()
        print(f"  N={N}")
        for name, fn in METHODS:
            mi = MAX_ITER[name].get(N, 10000)
            r  = fn(prob=prob, x0=x0, tol=1e-8, max_iter=mi, store_x=False)
            iters = np.arange(1, len(r.f_history)+1)
            f_pos = np.maximum(r.f_history, 1e-30)
            sl    = status_label(r, prob)
            ax.plot(iters, f_pos, '-', color=COLORS[name], lw=1.6, alpha=0.9,
                    label=f'{name} ({r.iterations:,}) [{sl}]')
        ax.axhline(1e-8, color='#fbbf24', lw=0.9, ls='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
    plt.suptitle('Convergence — f(xₖ) vs Iterations  (N=2·10·50·100·500)',
                 fontsize=13, y=1.02, color='#e6edf3')
    fig.tight_layout()
    out = 'outputs/scenario3_convergence_grid_ALL.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_scenario1_normal_zoom()
    plot_scenario1_multistart()
    plot_scenario1_all_N()
    plot_scenario2_all()
    plot_scenario3()
    print("\n✓  All plots saved to outputs/")