#ifndef APPCTX_H
#define APPCTX_H

#include <petscsnes.h>
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscmath.h>

typedef struct {
    PetscInt nx, ny, nt, IC;
    PetscReal hx, hy, ht;
    PetscReal xL, xR;
    PetscReal yL, yR;
    PetscReal t0, tF;
    PetscReal mphi2, mchi2;
    PetscReal ghost;        /* ghost mode on (-1); ghost mode off (+1) */
    PetscScalar A_time[8][8];
    PetscScalar A_space_x[8][8];
    PetscScalar A_space_y[8][8];
    PetscScalar A_mass[8][8];
    DM dm;

    /* choosing potential and parameters */
    PetscReal lam22;
    PetscReal lam_phi6;
    PetscReal g66;
    PetscReal lambda_66;

    /* IC specific parameters */
    PetscReal A;    /* Initial Amplitude, used for all IC */
    PetscReal C;    /* Controls strength of frequency (k = C * 2*pi/L)*/

	/* IC 4 parameters */
	PetscReal pw_r;               /* amplitude ratio r>0 */
	PetscReal pw_dphi;            /* \Delta\phi in [0,pi] */
	PetscInt  pw_sigma;           /* \pm 1: co/counter  propagation */

	/* IC 5 (oscillon) parameters */
	PetscReal osc_x0;     /* center x0 */
    PetscReal osc_y0;     /* center y0 */
	PetscReal osc_sigma;  /* width \sigma > 0 */
	PetscReal osc_r;      /* amplitude ratio r > 0 */
	PetscReal osc_dphi;   /* \Delta\phi in [0,pi] */
	PetscReal osc_k0;     /* optional carrier k0 (<=0 -> disabled) */

} AppCtx;

#endif /* APPCTX_H */

