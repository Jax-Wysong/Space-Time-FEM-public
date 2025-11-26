#ifndef NONLIN_H
#define NONLIN_H

#include "appctx.h"

void ComputeR_local_nonlinear(PetscScalar r_nonlin_phi[8], PetscScalar r_nonlin_chi[8],
                              const PetscScalar phi_local[8], const PetscScalar chi_local[8],
                              PetscReal hx, PetscReal hy, PetscReal ht, void *ctx);

void ComputeJ_local_nonlinear(PetscScalar J_local_pp[8][8], PetscScalar J_local_pc[8][8],
                              PetscScalar J_local_cc[8][8], PetscScalar J_local_cp[8][8],
                              const PetscScalar phi_local[8], const PetscScalar chi_local[8],
                              PetscReal hx, PetscReal hy, PetscReal ht, void *ctx);

#endif /* NONLIN_H */