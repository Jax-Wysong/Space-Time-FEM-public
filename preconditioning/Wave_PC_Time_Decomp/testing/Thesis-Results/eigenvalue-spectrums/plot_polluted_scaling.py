#!/usr/bin/env python3
"""
Plot how the fraction of polluted eigenvalues of M^{-1}J scales with nx, nt, and Nsub.

Prediction: polluted fraction = 2*(Nsub-1)/nt  (independent of nx, for overlap=1)

Three sweeps:
  1. Fixed Nsub=4, nt=40,  vary nx  = 20, 40, 80, 160   (matrix-constant-ntNsub-vary-nx)
  2. Fixed Nsub=4, nx=40,  vary nt  = 20, 40, 80, 160   (matrix-constant-nxNsub-vary-nt)
  3. Fixed nx=30,  nt=80,  vary Nsub= 1,2,4,8,10,20     (matrix-constant-nt-nx-ov1-vary-nsub)

Output: polluted_scaling.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 1e-6   # |lambda - 1| >= this => polluted


def load_evals(path):
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0] + 1j * data[:, 1]


def pct_polluted(ev):
    return 100.0 * np.sum(np.abs(ev - 1) >= THRESHOLD) / len(ev)


# ── Sweep 1: vary nx (Nsub=4, nt=40) ──────────────────────────────────────────
nx_vals  = [20, 40, 80, 160]
nt_s1    = 40
nsub_s1  = 4
dir_s1   = os.path.join(BASE, 'matrix-constant-ntNsub-vary-nx')

pct_s1 = []
for nx in nx_vals:
    path = os.path.join(dir_s1, f'evals_PA_{nsub_s1}-Nsub-{nx}x{nt_s1}.txt')
    pct_s1.append(pct_polluted(load_evals(path)))

Prediction_s1 = 200.0 * (nsub_s1 - 1) / nt_s1   # constant in nx


# ── Sweep 2: vary nt (Nsub=4, nx=40) ──────────────────────────────────────────
nt_vals  = [20, 40, 80, 160]
nx_s2    = 40
nsub_s2  = 4
dir_s2   = os.path.join(BASE, 'matrix-constant-nxNsub-vary-nt')

pct_s2 = []
for nt in nt_vals:
    path = os.path.join(dir_s2, f'evals_PA_{nsub_s2}-Nsub-{nx_s2}x{nt}.txt')
    pct_s2.append(pct_polluted(load_evals(path)))

nt_fine      = np.linspace(15, 170, 300)
Prediction_s2    = 200.0 * (nsub_s2 - 1) / nt_fine


# ── Sweep 3: vary Nsub (nx=30, nt=80) ─────────────────────────────────────────
nsub_vals = [1, 2, 4, 8, 10, 20]
nx_s3     = 30
nt_s3     = 80
dir_s3    = os.path.join(BASE, 'matrix-constant-nt-nx-ov1-vary-nsub')

pct_s3 = []
for ns in nsub_vals:
    path = os.path.join(dir_s3, f'evals_PA_{ns}-Nsub.txt')
    pct_s3.append(pct_polluted(load_evals(path)))

nsub_fine    = np.linspace(1, 21, 300)
Prediction_s3    = 200.0 * (nsub_fine - 1) / nt_s3


# # ── Collapse: all data vs (Nsub-1)/nt ─────────────────────────────────────────
# # ratio = (Nsub-1)/nt for each data point
# ratio_s1 = [(nsub_s1 - 1) / nt_s1] * len(nx_vals)          # constant
# ratio_s2 = [(nsub_s2 - 1) / nt  for nt  in nt_vals]
# ratio_s3 = [(ns - 1)       / nt_s3 for ns in nsub_vals]

# ratio_fine  = np.linspace(0, 0.26, 300)
# Prediction_coll = 200.0 * ratio_fine


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 8), constrained_layout=True)
fig.suptitle('Fraction of polluted eigenvalues of $M^{-1}J$\n'
             r'Prediction: $p = 2(N_\mathrm{sub}-1)/n_t \times 100\%$  (overlap $=1$)',
             fontsize=13)

mkw = dict(marker='o', markersize=7, linewidth=1.5, zorder=3)
tkw = dict(color='k', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

# Panel 1: vary nx
ax = axes[0]
ax.plot(nx_vals, pct_s1, color='#1f77b4', label='data', **mkw)
ax.axhline(Prediction_s1, label=f'Prediction  {Prediction_s1:.1f}%', **tkw)
ax.set_xlabel('$n_x$', fontsize=12)
ax.set_ylabel('% polluted', fontsize=12)
ax.set_title(f'Vary $n_x$  ($N_\\mathrm{{sub}}={nsub_s1}$, $n_t={nt_s1}$)', fontsize=11)
ax.set_ylim(0, max(pct_s1) * 1.5 + 2)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: vary nt
ax = axes[1]
ax.plot(nt_vals, pct_s2, color='#d62728', label='data', **mkw)
ax.plot(nt_fine, Prediction_s2, label=r'Prediction  $600/n_t$', **tkw)
ax.set_xlabel('$n_t$', fontsize=12)
ax.set_ylabel('% polluted', fontsize=12)
ax.set_title(f'Vary $n_t$  ($N_\\mathrm{{sub}}={nsub_s2}$, $n_x={nx_s2}$)', fontsize=11)
ax.set_ylim(0, max(pct_s2) * 1.5 + 2)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: vary Nsub
ax = axes[2]
ax.plot(nsub_vals, pct_s3, color='#2ca02c', label='data', **mkw)
ax.plot(nsub_fine, Prediction_s3,
        label=r'Prediction  $2(N_\mathrm{sub}-1)/80$', **tkw)
ax.set_xlabel('$N_\\mathrm{sub}$', fontsize=12)
ax.set_ylabel('% polluted', fontsize=12)
ax.set_title(f'Vary $N_\\mathrm{{sub}}$  ($n_x={nx_s3}$, $n_t={nt_s3}$)', fontsize=11)
ax.set_ylim(0, max(pct_s3) * 1.2 + 2)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

out = os.path.join(BASE, 'polluted_scaling.png')
fig.savefig(out, dpi=150)
print(f'Saved {out}')

# ── Console summary ───────────────────────────────────────────────────────────
print(f'\n{"Sweep 1: vary nx":=<50}')
print(f'  Prediction: {Prediction_s1:.2f}%  (constant)')
for nx, p in zip(nx_vals, pct_s1):
    print(f'  nx={nx:>4d}:  {p:.2f}%')

print(f'\n{"Sweep 2: vary nt":=<50}')
for nt, p in zip(nt_vals, pct_s2):
    t = 200.0 * (nsub_s2 - 1) / nt
    print(f'  nt={nt:>4d}:  {p:.2f}%  (Prediction {t:.2f}%)')

print(f'\n{"Sweep 3: vary Nsub":=<50}')
for ns, p in zip(nsub_vals, pct_s3):
    t = 200.0 * (ns - 1) / nt_s3
    print(f'  Nsub={ns:>3d}:  {p:.2f}%  (Prediction {t:.2f}%)')
