#ifndef NONLIN_H
#define NONLIN_H

#include "appctx.h"

void ComputeR_local_nonlinear(PetscScalar r_nonlin_phi[4], PetscScalar r_nonlin_chi[4], const PetscScalar phi_local[4], const PetscScalar chi_local[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx);

void ComputeJ_local_nonlinear(PetscScalar J_local_pp[4][4], PetscScalar J_local_pc[4][4], PetscScalar J_local_cc[4][4], PetscScalar J_local_cp[4][4], const PetscScalar phi_local[4], const PetscScalar chi_local[4], PetscReal hx, PetscReal ht, void *ctx);

#endif /* NONLIN_H */