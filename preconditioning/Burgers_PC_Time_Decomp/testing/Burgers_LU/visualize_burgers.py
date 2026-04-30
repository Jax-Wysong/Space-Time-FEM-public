#!/usr/bin/env python3
"""
visualize_burgers.py

Reads a PETSc binary .dat file produced by Burgers_1p1_PC (when save=1):
    u, x, t

Expected filename format:
    Burgers_visc-{nu}_1x{tF}_{nx}x{nt}.dat

Produces:
  1. Space-time colormaps: numerical u, exact u, pointwise error |u - u_exact|
  2. Time-slice comparison plots at a few selected times

Cole-Hopf exact solution (full periodic image sum):
    psi(x,t) = sum_{k=-K}^{K} exp(-(x - 4t - 2k*pi)^2 / (4*nu*(t+1)))
    u_exact  = -2*nu * psi_x / psi + 4

Usage:
    python visualize_burgers.py Burgers_visc-0.07_1x0.563_400x400.dat
    python visualize_burgers.py Burgers_visc-0.07_1x0.563_400x400.dat --nimages 10
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from petsc4py import PETSc


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_datfile(fname):
    """Read u, x, t Vecs from PETSc binary file."""
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    viewer = PETSc.Viewer().createBinary(fname, "r")

    def load_vec():
        v = PETSc.Vec().create()
        v.setFromOptions()
        v.load(viewer)
        arr = v.getArray().copy()
        v.destroy()
        return arr

    u = load_vec()
    x = load_vec()
    t = load_vec()

    viewer.destroy()
    return u, x, t


def parse_params_from_filename(fname):
    """Extract nu and tF from filename: Burgers_visc-{nu}_1x{tF}_{nx}x{nt}.dat"""
    base = os.path.basename(fname)
    m = re.search(r"visc-([\d.eE+\-]+)_1x([\d.eE+\-]+)", base)
    if m is None:
        raise ValueError(
            f"Cannot parse nu/tF from filename '{base}'.\n"
            "Expected pattern: Burgers_visc-<nu>_1x<tF>_<nx>x<nt>.dat"
        )
    nu = float(m.group(1))
    tF = float(m.group(2))
    return nu, tF


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def cole_hopf_exact(X, T, nu, nimages=20):
    """
    Evaluate the periodic Cole-Hopf exact solution on arrays X (nx,) and T (nt,).

    Returns U_exact of shape (nt, nx).

    Uses 2*nimages+1 image charges to approximate the periodic sum:
        psi(x,t) = sum_{k=-K}^{K} exp(-(x - 4t - 2k*pi)^2 / (4*nu*(t+1)))
        u_exact  = -2*nu * psi_x / psi + 4
    """
    # Shape: (nt, nx) via broadcasting
    x2d = X[np.newaxis, :]   # (1, nx)
    t2d = T[:, np.newaxis]   # (nt, 1)

    D = 4.0 * nu * (t2d + 1.0)   # (nt, 1)

    psi   = np.zeros((T.size, X.size))
    psi_x = np.zeros((T.size, X.size))

    for k in range(-nimages, nimages + 1):
        center = 4.0 * t2d + 2.0 * k * np.pi
        arg = (x2d - center) ** 2 / D
        g = np.exp(-arg)
        psi   += g
        psi_x += -2.0 * (x2d - center) / D * g

    return -2.0 * nu * psi_x / psi + 4.0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def set_plot_style(fontsize=14):
    mpl.rcParams["font.family"] = "DejaVu Serif"
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.titlesize"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = max(8, int(fontsize * 0.85))
    mpl.rcParams["ytick.labelsize"] = max(8, int(fontsize * 0.85))
    mpl.rcParams["legend.fontsize"] = max(8, int(fontsize * 0.85))


def plot_spacetime(x, t, U_num, U_exact, stem):
    """Three side-by-side space-time colormaps: numerical, exact, error."""
    error = np.abs(U_num - U_exact)

    vmin = min(np.nanmin(U_num), np.nanmin(U_exact))
    vmax = max(np.nanmax(U_num), np.nanmax(U_exact))
    emax = np.abs(error).max()

    extent = [x.min(), x.max(), t.min(), t.max()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(U_num, extent=extent, origin="lower",
                         aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Numerical $u$")
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$t$")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_exact, extent=extent, origin="lower",
                         aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Exact $u$")
    axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$t$")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, extent=extent, origin="lower",
                         aspect="auto", cmap="bwr", vmin=0, vmax=emax)
    axes[2].set_title(r"Error |$u_h - u_{\mathrm{exact}}$|")
    axes[2].set_xlabel("$x$"); axes[2].set_ylabel("$t$")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fname = f"{stem}_spacetime.png"
    fig.savefig(fname, dpi=200)
    print(f"Wrote {fname}")
    plt.close(fig)


def plot_slices(x, t, U_num, U_exact, stem, nslices=5):
    """Overlay numerical and exact u at a few evenly-spaced time levels."""
    nt = t.size
    indices = np.round(np.linspace(0, nt - 1, nslices)).astype(int)

    fig, axes = plt.subplots(1, nslices, figsize=(4 * nslices, 4), sharey=True)
    if nslices == 1:
        axes = [axes]

    for ax, ti in zip(axes, indices):
        ax.plot(x, U_num[ti],   label="Numerical", color="steelblue",  lw=2)
        ax.plot(x, U_exact[ti], label="Exact",     color="darkorange", lw=1.5, ls="--")
        ax.set_title(f"$t = {t[ti]:.4f}$")
        ax.set_xlabel("$x$")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("$u$")
    axes[-1].legend(loc="best")

    fig.tight_layout()
    fname = f"{stem}_slices.png"
    fig.savefig(fname, dpi=200)
    print(f"Wrote {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Burgers PETSc binary output vs Cole-Hopf exact solution."
    )
    parser.add_argument("datfile", help="PETSc binary .dat file")
    parser.add_argument(
        "--nimages", type=int, default=20,
        help="Number of image charges on each side for periodic exact solution (default: 20)"
    )
    parser.add_argument(
        "--nslices", type=int, default=5,
        help="Number of time slices to plot (default: 5)"
    )
    args = parser.parse_args()

    set_plot_style()

    # Read data
    print(f"Reading: {args.datfile}")
    u_vec, x, t = read_datfile(args.datfile)

    nx = x.size
    nt = t.size
    print(f"nx = {nx}, nt = {nt},  x in [{x[0]:.4f}, {x[-1]:.4f}],  t in [{t[0]:.6f}, {t[-1]:.6f}]")

    if u_vec.size != nx * nt:
        raise ValueError(f"u size {u_vec.size} != nx*nt = {nx*nt}")

    if not np.all(np.isfinite(u_vec)):
        n_bad = np.sum(~np.isfinite(u_vec))
        print(f"WARNING: {n_bad}/{u_vec.size} non-finite values in u — "
              "the .dat file may have been written with a 2-DOF indexing bug (ua[2*i] "
              "instead of ua[i]). Rebuild and re-save.", file=sys.stderr)

    # DMDA natural ordering: x-fastest, then t  →  shape (nt, nx)
    U_num = u_vec.reshape((nt, nx))

    # Parse nu from filename
    nu, tF = parse_params_from_filename(args.datfile)
    print(f"Parsed from filename: nu = {nu},  tF = {tF}")

    # Exact solution
    print(f"Computing exact solution with {2*args.nimages+1} image charges...")
    U_exact = cole_hopf_exact(x, t, nu, nimages=args.nimages)

    # Error summary
    diff = U_num - U_exact
    hx = x[1] - x[0] if nx > 1 else 1.0
    ht = t[1] - t[0] if nt > 1 else 1.0
    l2  = np.sqrt(np.sum(diff**2) * hx * ht)
    linf = np.max(np.abs(diff))
    print(f"L2  error = {l2:.6e}")
    print(f"Linf error = {linf:.6e}")

    # Output stem: same name as .dat without extension
    stem = os.path.splitext(args.datfile)[0]

    # Plots
    plot_spacetime(x, t, U_num, U_exact, stem)
    plot_slices(x, t, U_num, U_exact, stem, nslices=args.nslices)


if __name__ == "__main__":
    main()
