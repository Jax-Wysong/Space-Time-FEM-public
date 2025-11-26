#ifndef ENERGIES_H
#define ENERGIES_H

#include <petsc.h>
#include "appctx.h"

PetscErrorCode SliceEnergies(DM dm,
                             Vec U,
                             PetscInt t_idx,
                             AppCtx *user,
                             PetscReal *Hphi,
                             PetscReal *Hchi);

#endif /* ENERGIES_H */
