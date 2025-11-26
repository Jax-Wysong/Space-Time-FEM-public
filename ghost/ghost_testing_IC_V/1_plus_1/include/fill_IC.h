#ifndef FILL_IC_H
#define FILL_IC_H

#include "appctx.h"

PetscErrorCode FillInitialConditions(DM dm, Vec U, AppCtx *user);

#endif /* FILL_IC_H */