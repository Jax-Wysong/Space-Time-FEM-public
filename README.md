# space-time-FEM

This repository collects space-time finite element experiments built on Argonne National Laboratory's PETSc library. The examples use PETSc's SNES and DMDA infrastructure to parallelize nonlinear and linear PDE solves under a common workflow, from setting up distributed meshes to assembling residuals/Jacobians and driving the nonlinear solver.

## Repository layout
- `ghost/`: PETSc-based solvers for a "ghost" scalar field model. It contains:
  - `ghost_MMS/`: Manufactured-solution verification problems in 1+1 and 2+1 dimensions. There is a write-up of the PDE setup/discretization and results that verify correct implementation.
  - `ghost_testing_IC_V/`: Initial-condition and potential studies for the same model, preprint of paper will be available soon.
- `numerical_relativity/`: Space-time solver experiments for gravitational radiation from a non-rotating black hole (`gravitational_radiation_non-rotating_black_hole/`). Residuals, Jacobians, and potentials are assembled in PETSc before evolving with specified source term.
  - Current work is attempting to use our framework to reproduce results from Sopuerta and Laguna: https://doi.org/10.48550/arXiv.gr-qc/0512028
- `practice_problems/`: Baseline implementations of linear heat (homogenous) and wave equations (nonhomogenous) (`heat_eqn_linear/` and `wave_eqn_linear/`). These provide small drivers, element stiffness/residual routines, and job scripts that mirror the larger projects.
  - Anyone interested in using this code for a different PDE should refer to these examples to understand how implementation looks. 
  - There are PDF files containing PDE/discretization setup and verification.

## Building and running
There are .slurm scripts available where you can see what SNES/KSP options are being used as well as how to implement the problem specific options such as domain size, number of mesh points, etc.

Each subproject ships with a `makefile` that queries `petsc-config` for compiler and linker flags. A typical build-and-run flow is:

```bash
make               # build the local executable using petsc-config
mpirun -np <ranks> ./<executable> [problem-specific options]


