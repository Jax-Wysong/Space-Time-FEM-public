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
  PetscInt nx, nt, IC, Loop, overlap, Nsub;
  PetscReal hx, ht;
  PetscReal xL, xR;
  PetscReal t0, tF;
	PetscReal mphi2, mchi2;
	PetscReal A;
	PetscReal lam22;
	PetscReal ghost;
	PetscReal L;
	PetscScalar A_time[4][4];
	PetscScalar A_space[4][4];
	PetscScalar A_standard[4][4];
	DM dm;

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

  PetscInt   nx, nt, blksize, overlap, Nsub;

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

  /* Fieldsplit Stuff */
  PetscBool use_fs_ras;

  /* {phi,u} block slab objects (dof0=0) */
  Mat  A_phiu_start, A_phiu_middle, A_phiu_end;
  KSP  ksp_phiu_start, ksp_phiu_middle, ksp_phiu_end;
  Vec  xp_start, yp_start, xp_middle, yp_middle, xp_end, yp_end;

  /* {chi,v} block slab objects (dof0=2) */
  Mat  A_chiv_start, A_chiv_middle, A_chiv_end;
  KSP  ksp_chiv_start, ksp_chiv_middle, ksp_chiv_end;
  Vec  xc_start, yc_start, xc_middle, yc_middle, xc_end, yc_end;


  /* coarse level */
  PetscBool  use_coarse;
  PetscInt   coarse_thickness;

  /* Nonlinear rebuild guard: skip redundant PCSetUp calls within the same Newton step */
  PetscInt   last_snes_iter;

} SampleShellPC;

#endif /* APPCTX_H */