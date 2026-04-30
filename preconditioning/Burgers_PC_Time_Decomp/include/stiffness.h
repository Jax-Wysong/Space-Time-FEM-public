#ifndef STIFFNESS_H
#define STIFFNESS_H

#include "appctx.h"

void Compute_linear_stiffness(PetscScalar A_time[4][4], PetscScalar A_space[4][4], PetscScalar A_standard[4][4], PetscReal hx, PetscReal ht);

#endif /* STIFFNESS_H */