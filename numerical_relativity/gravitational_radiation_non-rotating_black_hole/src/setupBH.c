#include <stdlib.h>
#include "appctx.h"
#include "compute_rstar.h"
#include "potential.h"

/* Convert tortoise coordinate x = r_* to the physical
   Schwarzschild radius r. The Regge–Wheeler (and Zerilli)
   potentials V(r) are defined in terms of r, not r_*, so we
   precompute r(x) here. */
PetscErrorCode SetupBH(void *ctx)
{
    AppCtx    *user = (AppCtx*)ctx;
    PetscInt    nx  = user->nx;
    PetscReal   hx  = user->hx;
    PetscReal   xL  = user->xL;
    PetscScalar M   = user->M;
    PetscInt    ell = user->ell;

    PetscMalloc1(nx, &user->r_of_x);
    PetscMalloc1(nx, &user->V_of_x);

    for (PetscInt i = 0; i < nx; ++i) {
        PetscReal   x    = xL + i*hx;               // r_* coordinate
        PetscScalar r    = invert_rstar(x, M);      // Schwarzschild radius
        user->r_of_x[i] = r;
        user->V_of_x[i] = V_RW(r, ell, M);
    }

    return 0;
}
