#!/usr/bin/env python3
"""
post_1p1_local.py  (using petsc4py to read .dat files into numpy arrays)

Reads a PETSc binary .dat file containing 6 Vecs:
    phi, chi, Hphi, Hchi, x, t

Then:
  - reshapes phi, chi to [nt, nx] assuming DMDA natural ordering
  - plots Hamiltonian deviations
  - plots space-time color maps of phi(x,t), chi(x,t)
  
How to run:
In terminal run:
python post_1p1_local.py <your dat file>.dat

"""
 
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from petsc4py import PETSc


def read_datfile(fname):
    """Read 6 PETSc Vecs from binary file using petsc4py."""
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    viewer = PETSc.Viewer().createBinary(fname, "r")

    def load_vec():
        v = PETSc.Vec().create()
        v.setFromOptions()
        v.load(viewer)              # load from viewer
        arr = v.getArray().copy()   # NumPy array
        v.destroy()
        return arr

    phi  = load_vec()
    chi  = load_vec()
    Hphi = load_vec()
    Hchi = load_vec()
    x    = load_vec()
    t    = load_vec()

    viewer.destroy()
    return phi, chi, Hphi, Hchi, x, t


def post_1p1_local(datfile):
    print(f"Reading PETSc binary file via petsc4py: {datfile}")
    phi, chi, Hphi, Hchi, x, t = read_datfile(datfile)

    nx = x.size
    nt = t.size
    print(f"Detected nx = {nx}, nt = {nt}")
    print(f"phi.size = {phi.size}, chi.size = {chi.size}")

    if phi.size != nx * nt or chi.size != nx * nt:
        raise ValueError(
            f"phi/chi size mismatch: expected {nx}*{nt}={nx*nt}, "
            f"got phi={phi.size}, chi={chi.size}"
        )

    # DMDA natural ordering: x fastest, then t
    # reshape [nt, nx]
    Phi = phi.reshape((nt, nx))
    Chi = chi.reshape((nt, nx))
    
    # ---------------- Hamiltonian deviation plot ----------------
    dHphi = np.abs(Hphi - Hphi[0])
    dHchi = np.abs(Hchi - Hchi[0])

    figH, axH = plt.subplots(figsize=(8, 5))
    axH.loglog(t, dHphi, "-o", label=r"$|H_\phi - H_\phi(0)|$", color="chocolate", linewidth=0.7, markersize=3, markeredgewidth=0.5)
    axH.loglog(t, dHchi, "-s", label=r"$|H_\chi - H_\chi(0)|$", color="steelblue", linewidth=0.7, markersize=3, markeredgewidth=0.5)
    axH.set_xlabel("Time")
    axH.set_ylabel("|Hamiltonian deviations|")
    axH.grid(True, which="both")
    axH.legend(loc="best")
    axH.set_title("Hamiltonian deviations (1+1)")

    filenameH = f"IC_5_hamiltonian_deviations_1p1_{nx}x{nt}_GhostOn_lam1_MassOn.png"
    figH.tight_layout()
    figH.savefig(filenameH, dpi=200)
    print(f"Wrote {filenameH}")

    # ---------------- Space-time color plots ----------------
    # ------------ These are currently set up for the 1+1 figures ----------------
    # Custom chi colormap, bluish
    map_arr = np.column_stack(
        [
            np.linspace(0.0, 1.0, 256),
            np.linspace(0.5, 1.0, 256),
            np.linspace(0.5, 1.0, 256),
        ]
    )
    chi_cmap = ListedColormap(map_arr)

    figC, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # phi(x,t)
    im1 = ax1.imshow(
        Phi,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap="pink",
    )
    ax1.set_title(r"$\phi(x,t)$")
    ax1.set_xlabel("Space (x)")
    ax1.set_ylabel("Time (t)")
    figC.colorbar(im1, ax=ax1)

    # chi(x,t)
    im2 = ax2.imshow(
        Chi,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap=chi_cmap,
    )
    ax2.set_title(r"$\chi(x,t)$")
    ax2.set_xlabel("Space (x)")
    ax2.set_ylabel("Time")
    figC.colorbar(im2, ax=ax2)

    filenameC = f"IC_5_fields_1p1_{nx}x{nt}_GhostOn_lam1_MassOn.png"
    figC.tight_layout()
    figC.savefig(filenameC, dpi=200)
    print(f"Wrote {filenameC}")

    plt.close(figH)
    plt.close(figC)


def set_plot_style(font="DejaVu Serif", fontsize=16):
    mpl.rcParams["font.family"] = font
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.titlesize"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = int(max(8, fontsize * 0.9))
    mpl.rcParams["ytick.labelsize"] = int(max(8, fontsize * 0.9))
    mpl.rcParams["legend.fontsize"] = int(max(8, fontsize * 0.9))


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess PETSc binary 1+1 output using petsc4py."
    )
    parser.add_argument(
        "datfile",
        help="PETSc binary .dat file (e.g. lam22_IC_1_A0.6_C1_1x177_100x17701.dat)",
    )
    args = parser.parse_args()

    # Apply global matplotlib style before creating any figures
    # Font Type: DejaVu Serif
    # Font Size: 16 pt
    set_plot_style()

    try:
        post_1p1_local(args.datfile)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
