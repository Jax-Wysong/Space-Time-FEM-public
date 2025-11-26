#ifndef EXTEND_IN_TIME_H
#define EXTEND_IN_TIME_H

#include <petscsnes.h>
#include "appctx.h"

/* Performs the tF-extension loop until failure or maxloops reached */
PetscErrorCode ExtendInTimeUntilFailure(
    AppCtx      *user,
    PetscReal    tF_step,
    PetscInt     maxloops,
    Vec         *Ugood_out,
    DM          *dm_good_out,
    PetscReal   *last_good_tF_out,
    PetscInt    *last_good_nt_out,
    PetscBool   *found_failure_out
);

#endif /* EXTEND_IN_TIME_H */