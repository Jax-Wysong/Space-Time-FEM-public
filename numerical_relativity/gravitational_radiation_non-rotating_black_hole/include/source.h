#ifndef SOURCE_H
#define SOURCE_H

#include "appctx.h"
#include "ic.h"

void ComputeR_local_source(PetscScalar r_source_u[4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx);

void ComputeR_local_pp(PetscScalar r_pp[4],
                       PetscReal hx, PetscReal ht,
                       PetscReal x0, PetscReal t0,
                       void *ctx);

void ComputeJ_local_source(PetscScalar J_local_u[4][4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, void *ctx);

#endif /* SOURCE_H */