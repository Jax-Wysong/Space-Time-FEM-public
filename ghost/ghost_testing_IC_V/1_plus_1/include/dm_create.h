#ifndef DM_CREATE_H
#define DM_CREATE_H

#include "appctx.h"

PetscErrorCode CreateDMSNESForGrid(AppCtx *user, PetscInt nx, PetscInt nt, DM *dm, SNES *snes, Mat *J, Vec *U);

                                          
#endif /* DM_CREATE_H */
