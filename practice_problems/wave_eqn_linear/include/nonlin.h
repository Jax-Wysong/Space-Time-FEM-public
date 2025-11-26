#ifndef NONLIN_H
#define NONLIN_H

#include "appctx.h"

void ComputeR_local_nonlinear(PetscScalar r_nonlin_u[4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx);

void ComputeJ_local_nonlinear(PetscScalar J_local_u[4][4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, void *ctx);

#endif /* NONLIN_H */