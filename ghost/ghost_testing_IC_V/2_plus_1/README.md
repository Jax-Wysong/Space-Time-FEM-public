# space-time SNES/DMDA for ghost field project

## Layout
```
2p1_refactor/
├── Makefile
├── include/
│   ├── appctx.h      # AppCtx definition and PETSc includes
│   ├── energies.h    # SliceEnergies prototype
│   ├── ic.h          # Initial condition helpers (inline)
│   ├── jacobian.h    # FormJacobian prototype
│   ├── nonlin.h      # Local nonlinear residual/Jacobian blocks
│   ├── residual.h    # FormResidual prototype
│   └── stiffness.h   # Element-level constant matrices
└── src/
    ├── energies.c    # SliceEnergies implementation
    ├── jacobian.c    # Global Jacobian assembly
    ├── main.c        # Options, DM/SNES setup, solve, output
    ├── nonlin.c      # Nonlinear element contributions
    ├── residual.c    # Global residual assembly
    └── stiffness.c   # Linear element matrices
```

## Build

This project uses **`petsc-config`** (recommended by PETSc) to populate compiler
and linker flags.

```bash
# Ensure petsc-config is on PATH (or provide PETSC_DIR/ARCH, see below)
make
```

If `petsc-config` is not on PATH, point to PETSc:
```bash
make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt
# The Makefile will use: ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/bin/petsc-config
```

## Run

Typical example (4 ranks):
```bash
mpirun -np 4 ./lam22 -nx 40 -ny 40 -nt 240 -tF 6.0 -lam22 1.0
```

