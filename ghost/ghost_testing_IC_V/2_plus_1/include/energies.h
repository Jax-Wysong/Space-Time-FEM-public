#ifndef ENERGIES_H
#define ENERGIES_H

#include "appctx.h"

void SliceEnergies(DM dm, Vec U, PetscInt t_idx, AppCtx *user, PetscReal *Hphi, PetscReal *Hchi);

#endif /* ENERGIES_H */