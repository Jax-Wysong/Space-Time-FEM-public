#ifndef SAVESOL_H
#define SAVESOL_H

#include <petsc.h>
#include "appctx.h"


PetscErrorCode BuildOutputFilename(const AppCtx *user, char fname[], size_t len);

PetscErrorCode DumpSolutionAndEnergies(AppCtx    *user,
                                              DM         dm_good,
                                              Vec        Ugood,
                                              PetscInt   last_good_nt,
                                              PetscReal  last_good_tF);

PetscErrorCode SaveUgood(Vec src, Vec *Ugood, PetscInt *Ugood_n);


#endif /* SAVESOL_H */