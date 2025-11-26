#ifndef COMPUTE_RSTAR_H
#define COMPUTE_RSTAR_H

#include <petsc.h>
#include "appctx.h"

PetscErrorCode SetupBH(void *ctx);

/* this computes the tortoise coordinate r_*(r) */
static PetscScalar rstar_of_r(PetscScalar r, PetscScalar M)
{
    return r + 2.0*M * PetscLogReal(r/(2.0*M) - 1.0);
}


/* small Newton solver for r given r_* (assumes r > 2M) */
/* this gives us the physical radius r  */
static PetscScalar invert_rstar(PetscScalar rstar_target, PetscScalar M)
{
    const PetscScalar r_horizon = 2.0 * M;
    const PetscScalar r_min     = r_horizon * (1.0 + 1e-12); /* small cushion above 2M */

    PetscScalar r  = rstar_target;    // initial guess
    if (r < 2.2*M) r = 3.0*M;         // avoid starting too near horizon

    if (r <= r_min) r = r_min;        // enforce safe start

    for (int k = 0; k < 20; ++k) {
        PetscScalar f  = rstar_of_r(r,M) - rstar_target;
        PetscScalar df = 1.0 + 2.0*M / (r - 2.0*M);  // derivative of r* wrt r

        PetscScalar dr = -f / df;
        r += dr;
        if (r <= r_min) r = r_min;    // keep iterations away from the horizon

        if (PetscAbsScalar(dr) < 1e-12 * PetscAbsScalar(r)) break;
    }
    return r;
}

/* helper to interpolate x_p(t) from table user->x_p_of_t */
/* 
   this is called when we need to know where the particle
   is at a certain time t. It is not necessarily at some
   n*ht, so we must interpolate.
   Essentially, x_p_of_t stores the DISCRETE worldline (at n*ht), 
   this function gives us good approximations via interpolation
   about where the particle is between the dicrete parts.
*/
static inline PetscReal worldline_xp(PetscReal t, const AppCtx *user)
{
    PetscReal ht = user->ht;
    PetscInt  nt = user->nt;

    /* find integer time index */
    PetscReal idx = t / ht;
    PetscInt  n   = (PetscInt)PetscFloorReal(idx);

    if (n < 0)     n = 0;
    if (n > nt-2)  n = nt-2;

    PetscReal t_n   = n*ht;
    PetscReal tau   = (t - t_n)/ht;  /* in [0,1] */
    PetscReal x_n   = user->x_p_of_t[n];
    PetscReal x_np1 = user->x_p_of_t[n+1];

    return (1.0 - tau)*x_n + tau*x_np1;
}



#endif /* COMPUTE_RSTAR_H */