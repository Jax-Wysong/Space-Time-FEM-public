#!/usr/bin/env python3
"""
Plot eigenvalue spectra of J and M^{-1}J for each Nsub case.

Usage:
  source ~/venvs/petsc4py/bin/activate
  python3 plot_eigenvalues.py [matrix_dir]

Produces three figures:
  eig_spectrum_J.png          -- eigenvalues of J for each Nsub
  eig_spectrum_PA.png         -- eigenvalues of M^{-1}J for each Nsub
  eig_spectrum_PA_overlay.png -- all Nsub overlaid (polluted only)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# matplotlib.rcParams['figure.facecolor'] = "#080000"
# matplotlib.rcParams['axes.facecolor']   = '#080000'
# matplotlib.rcParams['axes.edgecolor']   = 'white'
# matplotlib.rcParams['axes.labelcolor']  = 'white'
# matplotlib.rcParams['xtick.color']      = 'white'
# matplotlib.rcParams['ytick.color']      = 'white'
# matplotlib.rcParams['text.color']       = 'white'
# matplotlib.rcParams['legend.facecolor'] = '#080000'
# matplotlib.rcParams['legend.edgecolor'] = 'white'
import matplotlib.lines as mlines

matrix_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

# -------------- for nt-nx-ov0-vary-nsub cases ----------------
# cases = [
#    {'nx': 30, 'nt': 80, 'nsub': 1,  'label': 'nx=30, nt=80, Nsub=1, overlap=0',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 30, 'nt': 80, 'nsub': 2,  'label': 'nx=30, nt=80, Nsub=2, overlap=0',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 30, 'nt': 80, 'nsub': 4,  'label': 'nx=30, nt=80, Nsub=4, overlap=0',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 30, 'nt': 80, 'nsub': 8,  'label': 'nx=30, nt=80, Nsub=8, overlap=0',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
#     {'nx': 30, 'nt': 80, 'nsub': 10, 'label': 'nx=30, nt=80, Nsub=10, overlap=0', 'color': "#002E79", 'marker': 'v', 'zorder': 0},
#     {'nx': 30, 'nt': 80, 'nsub': 20, 'label': 'nx=30, nt=80, Nsub=20, overlap=0', 'color': "#5D0089", 'marker': 'P', 'zorder': -1},
# ]

# # -------------- for nt-nx-ov1-vary-nsub cases ----------------
# cases = [
#    {'nx': 30, 'nt': 80, 'nsub': 1,  'label': 'nx=30, nt=80, Nsub=1, overlap=1',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 30, 'nt': 80, 'nsub': 2,  'label': 'nx=30, nt=80, Nsub=2, overlap=1',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 30, 'nt': 80, 'nsub': 4,  'label': 'nx=30, nt=80, Nsub=4, overlap=1',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 30, 'nt': 80, 'nsub': 8,  'label': 'nx=30, nt=80, Nsub=8, overlap=1',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
#     {'nx': 30, 'nt': 80, 'nsub': 10, 'label': 'nx=30, nt=80, Nsub=10, overlap=1', 'color': "#002E79", 'marker': 'v', 'zorder': 0},
#     {'nx': 30, 'nt': 80, 'nsub': 20, 'label': 'nx=30, nt=80, Nsub=20, overlap=1', 'color': "#5D0089", 'marker': 'P', 'zorder': -1},
# ]

# -------------- for nt-nx-ov2-vary-nsub cases ----------------
# cases = [
#    {'nx': 30, 'nt': 80, 'nsub': 1,  'label': 'nx=30, nt=80, Nsub=1, overlap=2',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 30, 'nt': 80, 'nsub': 2,  'label': 'nx=30, nt=80, Nsub=2, overlap=2',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 30, 'nt': 80, 'nsub': 4,  'label': 'nx=30, nt=80, Nsub=4, overlap=2',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 30, 'nt': 80, 'nsub': 8,  'label': 'nx=30, nt=80, Nsub=8, overlap=2',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
#     {'nx': 30, 'nt': 80, 'nsub': 10, 'label': 'nx=30, nt=80, Nsub=10, overlap=2', 'color': "#002E79", 'marker': 'v', 'zorder': 0},
#     {'nx': 30, 'nt': 80, 'nsub': 20, 'label': 'nx=30, nt=80, Nsub=20, overlap=2', 'color': "#5D0089", 'marker': 'P', 'zorder': -1},
# ]

# # -------------- for nt-nx-ov3-vary-nsub cases ----------------
# cases = [
#    {'nx': 30, 'nt': 80, 'nsub': 1,  'label': 'nx=30, nt=80, Nsub=1, overlap=3',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 30, 'nt': 80, 'nsub': 2,  'label': 'nx=30, nt=80, Nsub=2, overlap=3',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 30, 'nt': 80, 'nsub': 4,  'label': 'nx=30, nt=80, Nsub=4, overlap=3',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 30, 'nt': 80, 'nsub': 8,  'label': 'nx=30, nt=80, Nsub=8, overlap=3',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
#     {'nx': 30, 'nt': 80, 'nsub': 10, 'label': 'nx=30, nt=80, Nsub=10, overlap=3', 'color': "#002E79", 'marker': 'v', 'zorder': 0},
#     {'nx': 30, 'nt': 80, 'nsub': 20, 'label': 'nx=30, nt=80, Nsub=20, overlap=3', 'color': "#5D0089", 'marker': 'P', 'zorder': -1},
# ]

# ------ no Dirichlet case -----------------
# cases = [
#    {'nx': 30, 'nt': 80, 'nsub': 1,  'label': 'nx=30, nt=80, Nsub=1',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 30, 'nt': 80, 'nsub': 2,  'label': 'nx=30, nt=80, Nsub=2',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 30, 'nt': 80, 'nsub': 4,  'label': 'nx=30, nt=80, Nsub=4',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 30, 'nt': 80, 'nsub': 8,  'label': 'nx=30, nt=80, Nsub=8',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
# ]

# ------- constant ntNsub-vary-nx cases -----------------
# cases = [
#    {'nx': 20, 'nt': 40, 'nsub': 4, 'label': 'nx=20, nt=40, Nsub=4',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 40, 'nt': 40, 'nsub': 4, 'label': 'nx=40, nt=40, Nsub=4',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 80, 'nt': 40, 'nsub': 4, 'label': 'nx=80, nt=40, Nsub=4',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 160, 'nt': 40, 'nsub': 4, 'label': 'nx=160, nt=40, Nsub=4',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
# ]

# ---------- constant nx20Nsub-vary-nt cases -----------------
# cases = [
#    {'nx': 20, 'nt': 20, 'nsub': 4, 'label': 'nx=20, nt=20, Nsub=4',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 20, 'nt': 40, 'nsub': 4, 'label': 'nx=20, nt=40, Nsub=4',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 20, 'nt': 80, 'nsub': 4, 'label': 'nx=20, nt=80, Nsub=4',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 20, 'nt': 320, 'nsub': 4, 'label': 'nx=20, nt=320, Nsub=4',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
# ]

# ---------- constant nx40Nsub-vary-nt cases -----------------
# cases = [
#    {'nx': 40, 'nt': 20, 'nsub': 4, 'label': 'nx=40, nt=20, Nsub=4',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 40, 'nt': 40, 'nsub': 4, 'label': 'nx=40, nt=40, Nsub=4',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 40, 'nt': 80, 'nsub': 4, 'label': 'nx=40, nt=80, Nsub=4',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 40, 'nt': 160, 'nsub': 4, 'label': 'nx=40, nt=160, Nsub=4',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
# ]

# ---------- constant nx80Nsub-vary-nt cases -----------------
# cases = [
#    {'nx': 80, 'nt': 20, 'nsub': 4, 'label': 'nx=80, nt=20, Nsub=4',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 80, 'nt': 40, 'nsub': 4, 'label': 'nx=80, nt=40, Nsub=4',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 80, 'nt': 80, 'nsub': 4, 'label': 'nx=80, nt=80, Nsub=4',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 80, 'nt': 160, 'nsub': 4, 'label': 'nx=80, nt=160, Nsub=4',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
# ]

# -------- constant nx and Nsub size, vary nt cases -----------------
# cases = [
#    {'nx': 10, 'nt': 10, 'nsub': 1,  'label': 'nx=10, nt=10, Nsub=1',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
#    {'nx': 10, 'nt': 20, 'nsub': 2,  'label': 'nx=10, nt=20, Nsub=2',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
#    {'nx': 10, 'nt': 40, 'nsub': 4,  'label': 'nx=10, nt=40, Nsub=4',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
#     {'nx': 10, 'nt': 80, 'nsub': 8,  'label': 'nx=10, nt=80, Nsub=8',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
#     {'nx': 10, 'nt': 160, 'nsub': 16, 'label': 'nx=10, nt=160, Nsub=16', 'color': "#002E79", 'marker': 'v', 'zorder': 0},
#     {'nx': 10, 'nt': 320, 'nsub': 32, 'label': 'nx=10, nt=320, Nsub=32', 'color': "#5D0089", 'marker': 'P', 'zorder': -1},
# ]

# -------- constant Nsub 5 vary nx=nt cases -----------------
cases = [
#    {'nx': 10, 'nt': 10, 'nsub': 5,  'label': 'nx=10, nt=10, Nsub=5',  'color': '#1f77b4', 'marker': 'o', 'zorder': 4},
   {'nx': 20, 'nt': 20, 'nsub': 5,  'label': 'nx=20, nt=20, Nsub=5',  'color': '#ff7f0e', 'marker': 's', 'zorder': 3},
   {'nx': 40, 'nt': 40, 'nsub': 5,  'label': 'nx=40, nt=40, Nsub=5',  'color': '#2ca02c', 'marker': '^', 'zorder': 2},
    {'nx': 60, 'nt': 60, 'nsub': 5,  'label': 'nx=60, nt=60, Nsub=5',  'color': "#F80828", 'marker': 'D', 'zorder': 1},
    {'nx': 80, 'nt': 80, 'nsub': 5, 'label': 'nx=80, nt=80, Nsub=5', 'color': "#002E79", 'marker': 'v', 'zorder': 0},
    {'nx': 100, 'nt': 100, 'nsub': 5, 'label': 'nx=100, nt=100, Nsub=5', 'color': "#5D0089", 'marker': 'P', 'zorder': -1},
    {'nx': 120, 'nt': 120, 'nsub': 5, 'label': 'nx=120, nt=120, Nsub=5', 'color': "#8B008B", 'marker': 'X', 'zorder': -2},
]


DOT_SIZE      = 10   # uniform across all subfigures
LEGEND_FS     = 11   # legend font size
SUPER_TITLE_FS = 20   # figure super-title font size
OVERLAY_DOT_S = 16   # dot size in the overlay figure

theta = np.linspace(0, 2*np.pi, 400)


def load_evals(path):
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0] + 1j * data[:, 1]


def spectral_stats(ev):
    mags   = np.abs(ev)
    rho    = mags.max()                 # spectral radius: max |lambda|
    kappa  = mags.max() / mags.min()   # condition number: max|lambda| / min|lambda|
    r_from_1 = np.abs(ev - 1).max()    # radius of tightest circle centred at 1
    return rho, kappa, r_from_1


def stat_handles(kappa):
    """Invisible legend entry that displays the condition number."""
    return [
        mlines.Line2D([], [], linestyle='none', color='none',
                      label=f'κ = {kappa:.4g}'),
    ]


# ── Figure 1: eigenvalues of J ─────────────────────────────────────────────
fig_J, axes_J = plt.subplots(3, 2, figsize=(10, 13), constrained_layout=True)
fig_J.suptitle('Eigenvalues of J (space-time stiffness matrix)', fontsize=SUPER_TITLE_FS)

for ax, c in zip(axes_J.flat, cases):
    path = os.path.join(matrix_dir, f"evals_J_{c['nsub']}-Nsub-{c['nx']}x{c['nt']}.txt")
    if not os.path.exists(path):
        ax.set_visible(False)
        continue
    ev = load_evals(path)
    rho, kappa, r_from_1 = spectral_stats(ev)

    ax.scatter(ev.real, ev.imag, s=DOT_SIZE, color=c['color'],
               alpha=0.7, linewidths=0, label=c['label'])

    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_title(f"{c['label']}  (N={len(ev)})", fontsize=11)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    y_half = np.abs(ev.imag).max()
    if y_half == 0:
        y_half = 0.1
    ax.set_ylim(-y_half * 1.1, y_half * 1.1)
    ax.grid(True, alpha=0.3)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + stat_handles(kappa),
              fontsize=LEGEND_FS, loc='upper left')

fig_J.savefig('eig_spectrum_J.png', dpi=150)
print('Saved eig_spectrum_J.png')


# ── Figure 2: eigenvalues of M^{-1}J ──────────────────────────────────────
fig_PA, axes_PA = plt.subplots(3, 2, figsize=(10, 13), constrained_layout=True)
fig_PA.suptitle('Eigenvalues of M$^{-1}$J  (RAS preconditioned operator)', fontsize=SUPER_TITLE_FS)

for ax, c in zip(axes_PA.flat, cases):
    path = os.path.join(matrix_dir, f"evals_PA_{c['nsub']}-Nsub-{c['nx']}x{c['nt']}.txt")
    if not os.path.exists(path):
        ax.set_visible(False)
        continue
    ev = load_evals(path)
    rho, kappa, r_from_1 = spectral_stats(ev)

    at_one   = ev[np.abs(ev - 1) < 1e-6]
    polluted = ev[np.abs(ev - 1) >= 1e-6]

    h_at_one = ax.scatter(at_one.real, at_one.imag, s=DOT_SIZE, color='#aec7e8',
                          alpha=0.5, linewidths=0, label=f'λ≈1  ({len(at_one)})')
    h_polluted = None
    if len(polluted):
        h_polluted = ax.scatter(polluted.real, polluted.imag, s=DOT_SIZE,
                                color=c['color'], alpha=0.9, linewidths=0.3,
                                edgecolors='k', label=f'polluted ({len(polluted)})')

    # unit circle for reference
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.7, alpha=0.4)

    # dotted circle centred at 1 with radius = max|lambda - 1|
    h_circle, = ax.plot(1 + r_from_1 * np.cos(theta), r_from_1 * np.sin(theta),
                        'r:', lw=1.4, alpha=0.85, label=f'cluster r={r_from_1:.3f}')

    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_title(f"{c['label']}  (N={len(ev)})", fontsize=11)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    pct_polluted = len(polluted) / len(ev) * 100
    pct_handle = mlines.Line2D([], [], linestyle='none', color='none',
                               label=f'  {pct_polluted:.1f}% polluted')
    grp1 = [h_at_one] + ([h_polluted, pct_handle] if h_polluted is not None else [])
    grp2 = [h_circle] + stat_handles(kappa)
    leg1 = ax.legend(handles=grp1, fontsize=LEGEND_FS, loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=grp2, fontsize=LEGEND_FS, loc='lower left')

fig_PA.savefig('eig_spectrum_PA.png', dpi=150)
print('Saved eig_spectrum_PA.png')


# ── Figure 3: all Nsub overlaid (PA only, polluted eigenvalues) ────────────
fig_all, ax_all = plt.subplots(figsize=(8, 7), constrained_layout=True)
# ax_all.set_title('M$^{-1}$J eigenvalues — all Nsub overlaid\n(λ≈1 cluster omitted)',
#                  fontsize=15)

for c in cases:
    path = os.path.join(matrix_dir, f"evals_PA_{c['nsub']}-Nsub-{c['nx']}x{c['nt']}.txt")
    if not os.path.exists(path):
        continue
    ev = load_evals(path)
    rho, kappa, _ = spectral_stats(ev)
    polluted = ev[np.abs(ev - 1) >= 1e-6]
    if len(polluted):
        ax_all.scatter(polluted.real, polluted.imag,
                       s=OVERLAY_DOT_S, color=c['color'],
                       alpha=0.85, linewidths=0.3, edgecolors='k',
                       label=f"{c['label']}  (κ={kappa:.4g})",
                       zorder=c['zorder'])

ax_all.plot(np.cos(theta), np.sin(theta), 'k--', lw=1, alpha=0.5,
            label='unit circle')
ax_all.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax_all.axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax_all.set_xlabel('Re(λ)', fontsize=12)
ax_all.set_ylabel('Im(λ)', fontsize=12)
ax_all.set_aspect('equal')
ax_all.legend(fontsize=LEGEND_FS)
ax_all.grid(True, alpha=0.3)

fig_all.savefig('eig_spectrum_PA_overlay.png', dpi=150)
print('Saved eig_spectrum_PA_overlay.png')
