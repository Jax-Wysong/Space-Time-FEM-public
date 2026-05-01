#!/usr/bin/env python3
"""
Compute and save eigenvalues of J and M^{-1}J.

Usage:
  python3 compute_eigenvalues.py J.dat PA_dense.bin [label]

Outputs:
  evals_J_<label>.txt   -- eigenvalues of J       (columns: Re Im)
  evals_PA_<label>.txt  -- eigenvalues of M^{-1}J (columns: Re Im)
"""
import sys
import numpy as np
import scipy.linalg

import petsc4py
petsc4py.init()
from petsc4py import PETSc


def load_J(filename):
    """Read a PETSc binary matrix and return as a dense numpy array."""
    viewer = PETSc.Viewer().createBinary(filename, mode='r',
                                         comm=PETSc.COMM_SELF)
    mat = PETSc.Mat().load(viewer)
    viewer.destroy()
    N, _ = mat.getSize()
    A = np.zeros((N, N), dtype=np.float64)
    ai, aj, av = mat.getValuesCSR()
    for i in range(N):
        for k in range(ai[i], ai[i + 1]):
            A[i, aj[k]] = av[k]
    mat.destroy()
    return A


def load_PA(filename):
    """Read the custom dense binary written by main.c.

    Format: int32 N, then N*N float64 values in column-major order.
    """
    with open(filename, 'rb') as f:
        N = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        data = np.frombuffer(f.read(8 * N * N), dtype=np.float64)
    return data.reshape((N, N), order='F')


if len(sys.argv) < 3:
    print("Usage: compute_eigenvalues.py J.dat PA_dense.bin [label]")
    sys.exit(1)

J_file  = sys.argv[1]
PA_file = sys.argv[2]
label   = sys.argv[3] if len(sys.argv) > 3 else ""
tag     = f"_{label}" if label else ""

print(f"Loading J from {J_file} ...")
J = load_J(J_file)
print(f"Loading M^{{-1}}J from {PA_file} ...")
PA = load_PA(PA_file)

print("Computing eigenvalues ...")
evals_J  = scipy.linalg.eigvals(J)

bad = ~np.isfinite(PA)
if bad.any():
    rows, cols = np.where(bad)
    print(f"WARNING: PA contains {bad.sum()} non-finite entries "
          f"(rows {rows[:5]}..., cols {cols[:5]}...); "
          f"computing eigenvalues anyway (check_finite=False)")
evals_PA = scipy.linalg.eigvals(PA, check_finite=False)

np.savetxt(f"evals_J{tag}.txt",
           np.column_stack([evals_J.real,  evals_J.imag]),
           header="Re Im", comments='')
np.savetxt(f"evals_PA{tag}.txt",
           np.column_stack([evals_PA.real, evals_PA.imag]),
           header="Re Im", comments='')

nz_J  = np.abs(evals_J)  > 1e-14
nz_PA = np.abs(evals_PA) > 1e-14

print(f"\nJ  ({J.shape[0]}x{J.shape[0]}):")
print(f"  cond (|lam_max|/|lam_min|) ~ {np.abs(evals_J).max() / np.abs(evals_J[nz_J]).min():.3e}")
print(f"  max |Im(lambda)|            = {np.abs(evals_J.imag).max():.3e}")

print(f"\nM^{{-1}}J ({PA.shape[0]}x{PA.shape[0]}):")
print(f"  spectral radius  = {np.abs(evals_PA).max():.6f}")
print(f"  min |lambda|     = {np.abs(evals_PA[nz_PA]).min():.4e}")
print(f"  max |Im(lambda)| = {np.abs(evals_PA.imag).max():.3e}")

print(f"\nSaved evals_J{tag}.txt and evals_PA{tag}.txt")