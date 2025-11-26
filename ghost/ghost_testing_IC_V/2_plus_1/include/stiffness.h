#ifndef STIFFNESS_H
#define STIFFNESS_H

#include "appctx.h"

void Compute_linear_stiffness(PetscScalar A_time[8][8],
                              PetscScalar A_space_x[8][8],
                              PetscScalar A_space_y[8][8],
                              PetscScalar A_mass[8][8],
                              PetscReal hx, PetscReal hy, PetscReal ht);

#endif /* STIFFNESS_H */