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
    PetscInt nx, nt, IC, Loop;
    PetscReal hx, ht;
    PetscReal xL, xR;
    PetscReal t0, tF;
	PetscReal mphi2, mchi2;
	PetscReal A;
	PetscReal lam22;
	PetscReal ghost;
	PetscScalar A_time[4][4];
	PetscScalar A_space[4][4];
	PetscScalar A_standard[4][4];
	DM dm;
	
	PetscReal C; // kL/2pi = C in wave IC
	
    /* ---- colored-noise IC params ---- */
    PetscInt  n1, n2;        /* mode range; require n1>=1 */
    PetscReal ns;            /* spectral tilt */
    PetscReal L;             /* domain length = xR-xL */
    PetscReal dphi;          /* optional phase offset for chi */
    PetscReal *theta;        /* phases θ_n, length n2-n1+1 */
    PetscInt  *ssign;        /* s_n in {\pm 1}, length n2-n1+1 */

	/* IC 4 parameters */
	PetscReal pw_r;               /* amplitude ratio r>0 */
	PetscReal pw_dphi;            /* delta phi in [0,pi] */
	PetscInt  pw_sigma;           /* \pm 1: co/counter */
	
	/* IC 5 (oscillon) parameters */
	PetscReal osc_x0;     /* center x0 */
	PetscReal osc_sigma;  /* width sigma > 0 */
	PetscReal osc_r;      /* amplitude ratio r > 0 */
	PetscReal osc_dphi;   /* delta phi in [0,pi] */
	PetscReal osc_k0;     /* optional carrier k0 (<=0 -> disabled) */

} AppCtx;


#endif /* APPCTX_H */
