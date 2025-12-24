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
    PetscInt nx, ny, nt;
    PetscReal hx, hy, ht;
    PetscReal xL, xR, yL, yR;
    PetscReal t0, tF;
    PetscReal lam;
    DM dm;
    PetscScalar A_time[8][8];
    PetscScalar A_space_x[8][8];
    PetscScalar A_space_y[8][8];
    PetscScalar A_mass[8][8];
} AppCtx;

#endif // APPCTX_H
