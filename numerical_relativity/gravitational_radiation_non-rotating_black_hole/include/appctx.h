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
    PetscInt nx, nt, IC;
    PetscReal hx, ht;
    PetscReal xL, xR;
    PetscReal t0, tF;
	PetscReal L;
	PetscScalar A_time[4][4];
	PetscScalar A_space[4][4];
	PetscScalar A_standard[4][4];
	/* Specific for this problem */
	PetscScalar A_time_edge_R[4][4]; 
	PetscScalar A_time_edge_L[4][4];
	PetscScalar M;
	PetscInt ell;
	PetscScalar *r_of_x;
	PetscScalar *V_of_x;
	PetscScalar *x_p_of_t;
	DM dm;
	
} AppCtx;

#endif /* APPCTX_H */