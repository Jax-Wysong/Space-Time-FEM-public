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

    PetscInt Nsub, overlap;
	PetscBool slab_pc_nonlinear; /* if true, include nonlinear terms in slab pc */
    PetscBool slab_pc_ras;  /* if true: RAS; if false: pure ASM */

} AppCtx;

	/* Define context for user-defined ASM */
typedef enum { SLAB_START=0, SLAB_MIDDLE=1, SLAB_END=2 } SlabType;

typedef struct {
  AppCtx    *user;
  DM         dm;
  SNES       snes;
  PetscBool  use_nonlinear;

  PetscInt   nx, ny, nt, blksize, overlap, Nsub;

  PetscMPIInt rank, size;
  SlabType    slabtype;

  Vec        xloc;
  Vec        Uloc;

  PetscBool use_ras;
  Vec       yloc;   /* ghosted local workspace for ASM add-back */

  /* Only ONE of these will be non-NULL on a given rank */
  Mat        A_start, A_middle, A_end;
  KSP        ksp_start, ksp_middle, ksp_end;
  Vec        x_start, y_start, x_middle, y_middle, x_end, y_end;

} SampleShellPC;

#endif // APPCTX_H
