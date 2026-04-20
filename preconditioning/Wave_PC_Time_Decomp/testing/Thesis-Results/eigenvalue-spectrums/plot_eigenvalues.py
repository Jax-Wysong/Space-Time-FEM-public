#!/usr/bin/env python3
"""
Plot eigenvalue spectra of J and M^{-1}J for each Nsub case.

Usage:
  python3 plot_eigenvalues.py [matrix_dir]

Produces two figures:
  eig_spectrum_J.png   -- eigenvalues of J for each Nsub
  eig_spectrum_PA.png  -- eigenvalues of M^{-1}J for each Nsub
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matrix_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

# cases = [
#     {'nsub': 1, 'label': 'Nsub=1', 'color': '#1f77b4', 'marker': 'o', 'size': 20, 'zorder': 4},
#     {'nsub': 2, 'label': 'Nsub=2', 'color': '#ff7f0e', 'marker': 's', 'size': 15, 'zorder': 3},
#     {'nsub': 4, 'label': 'Nsub=4', 'color': '#2ca02c', 'marker': '^', 'size': 12, 'zorder': 2},
#     {'nsub': 8, 'label': 'Nsub=8', 'color': '#d62728', 'marker': 'D', 'size':  9, 'zorder': 1},
#     {'nsub': 16, 'label': 'Nsub=16', 'color': '#9467bd', 'marker': 'v', 'size':  7, 'zorder': 0},
#     {'nsub': 32, 'label': 'Nsub=32', 'color': '#7f7f7f', 'marker': 'p', 'size':  5, 'zorder': 0},
#     ]

# cases = [
#     {'nsub': 1, 'label': 'Nsub=1', 'color': '#1f77b4', 'marker': 'o', 'size': 20, 'zorder': 4},
#     {'nsub': 2, 'label': 'Nsub=2', 'color': '#ff7f0e', 'marker': 's', 'size': 15, 'zorder': 3},
#     {'nsub': 4, 'label': 'Nsub=4', 'color': '#2ca02c', 'marker': '^', 'size': 12, 'zorder': 2},
#     {'nsub': 8, 'label': 'Nsub=8', 'color': '#d62728', 'marker': 'D', 'size':  9, 'zorder': 1},
#     # {'nsub': 10,'label': 'Nsub=10','color': '#9467bd', 'marker': 'v', 'size':  7, 'zorder': 0},
#     # {'nsub': 20,'label': 'Nsub=20','color': '#7f7f7f', 'marker': 'p', 'size':  5, 'zorder': 0},
#     ]

# cases = [
#     {'nx': 20,  'nt': 40, 'nsub': 4, 'label': 'Nsub=4, nx=20, nt=40',   'color': '#1f77b4', 'marker': 'o', 'size': 20, 'zorder': 4},
#     {'nx': 40,  'nt': 40, 'nsub': 4, 'label': 'Nsub=4, nx=40, nt=40',   'color': "#f30909", 'marker': 'x', 'size': 17, 'zorder': 3},
#     {'nx': 80,  'nt': 40, 'nsub': 4, 'label': 'Nsub=4, nx=80, nt=40',   'color': '#ff7f0e', 'marker': 's', 'size': 15, 'zorder': 2},
#     {'nx': 160, 'nt': 40, 'nsub': 4, 'label': 'Nsub=4, nx=160, nt=40',  'color': '#2ca02c', 'marker': '^', 'size': 12, 'zorder': 1},
# ]

cases = [
    {'nx': 40,  'nt': 20, 'nsub': 4, 'label': 'Nsub=4, nx=20, nt=20',   'color': '#1f77b4', 'marker': 'o', 'size': 20, 'zorder': 5},
    {'nx': 40,  'nt': 40, 'nsub': 4, 'label': 'Nsub=4, nx=20, nt=40',   'color': "#f30909", 'marker': 'x', 'size': 17, 'zorder': 4},
    {'nx': 40,  'nt': 80, 'nsub': 4, 'label': 'Nsub=4, nx=20, nt=80',   'color': '#ff7f0e', 'marker': 's', 'size': 15, 'zorder': 3},
    {'nx': 40, 'nt': 160, 'nsub': 4, 'label': 'Nsub=4, nx=20, nt=160',  'color': '#2ca02c', 'marker': '^', 'size': 12, 'zorder': 2},
]

def load_evals(path):
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0] + 1j * data[:, 1]


# ── Figure 1: eigenvalues of J ─────────────────────────────────────────────
fig_J, axes_J = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
fig_J.suptitle('Eigenvalues of J (space-time stiffness matrix)', fontsize=13)

for ax, c in zip(axes_J.flat, cases):
    path = os.path.join(matrix_dir, f"evals_J_{c['nsub']}-Nsub-{c['nx']}x{c['nt']}.txt")
    if not os.path.exists(path):
        ax.set_visible(False)
        continue
    ev = load_evals(path)
    ax.scatter(ev.real, ev.imag, s=c['size'], color=c['color'],
               alpha=0.7, linewidths=0, label=c['label'])
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_title(f"{c['label']}  (N={len(ev)})", fontsize=11)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    # enforce a minimum y-window so the plot isn't a thin line when Im ≈ 0
    y_half = max(np.abs(ev.imag).max(), np.abs(ev.real).max() * 0.05)
    ax.set_ylim(-y_half * 1.2, y_half * 1.2)
    ax.grid(True, alpha=0.3)

fig_J.savefig('eig_spectrum_J.png', dpi=150)
print('Saved eig_spectrum_J.png')


# ── Figure 2: eigenvalues of M^{-1}J ──────────────────────────────────────
fig_PA, axes_PA = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
fig_PA.suptitle('Eigenvalues of M$^{-1}$J  (RAS preconditioned operator)', fontsize=13)

for ax, c in zip(axes_PA.flat, cases):
    path = os.path.join(matrix_dir, f"evals_PA_{c['nsub']}-Nsub-{c['nx']}x{c['nt']}.txt")
    if not os.path.exists(path):
        ax.set_visible(False)
        continue
    ev = load_evals(path)

    # separate the cluster at 1 from the polluted eigenvalues
    at_one   = ev[np.abs(ev - 1) < 1e-6]
    polluted = ev[np.abs(ev - 1) >= 1e-6]

    ax.scatter(at_one.real,   at_one.imag,   s=c['size'],     color='#aec7e8',
               alpha=0.5, linewidths=0, label=f'λ≈1  ({len(at_one)})')
    if len(polluted):
        ax.scatter(polluted.real, polluted.imag, s=c['size']+6, color=c['color'],
                   alpha=0.9, linewidths=0.3, edgecolors='k',
                   label=f'polluted ({len(polluted)})')

    # unit circle for reference
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.7, alpha=0.4)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)

    ax.set_title(f"{c['label']}  (N={len(ev)})", fontsize=11)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

fig_PA.savefig('eig_spectrum_PA.png', dpi=150)
print('Saved eig_spectrum_PA.png')


# ── Figure 3: all Nsub overlaid (PA only, zoomed to interesting region) ────
fig_all, ax_all = plt.subplots(figsize=(8, 7), constrained_layout=True)
ax_all.set_title('M$^{-1}$J eigenvalues — all Nsub overlaid\n(λ≈1 cluster omitted)', fontsize=12)

for c in cases:
    path = os.path.join(matrix_dir, f"evals_PA_{c['nsub']}-Nsub-{c['nx']}x{c['nt']}.txt")
    if not os.path.exists(path):
        continue
    ev = load_evals(path)
    polluted = ev[np.abs(ev - 1) >= 1e-6]
    if len(polluted):
        ax_all.scatter(polluted.real, polluted.imag,
                       s=c['size'] + 4, color=c['color'],
                       alpha=0.85, linewidths=0.3, edgecolors='k',
                       label=c['label'], zorder=c['zorder'])

theta = np.linspace(0, 2*np.pi, 300)
ax_all.plot(np.cos(theta), np.sin(theta), 'k--', lw=1, alpha=0.5, label='unit circle')
ax_all.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax_all.axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax_all.set_xlabel('Re(λ)', fontsize=12)
ax_all.set_ylabel('Im(λ)', fontsize=12)
ax_all.set_aspect('equal')
ax_all.legend(fontsize=10)
ax_all.grid(True, alpha=0.3)

fig_all.savefig('eig_spectrum_PA_overlay.png', dpi=150)
print('Saved eig_spectrum_PA_overlay.png')
