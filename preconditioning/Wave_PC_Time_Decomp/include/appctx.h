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
  PetscInt nx, nt, IC, overlap, interface_width, Nsub;
  PetscReal hx, ht;
  PetscReal xL, xR;
  PetscReal t0, tF;
	PetscReal L;
	PetscScalar A_time[4][4];
	PetscScalar A_space[4][4];
	PetscScalar A_standard[4][4];
	DM dm;

  PetscBool slab_pc_ras;  /* if true: RAS; if false: pure ASM */
  PetscBool full_overlap_dirichlet; /* if true: pin ov left time levels per slab; if false: pin only 1 */
  PetscBool first_sub_only_bc; /* if true: only impose temporal Dirichlet BC on the first subdomain */

  PetscBool dump_matrices; /* if true, dump J and M^{-1}J to binary files for testing */

	} AppCtx;


	/* Define context for user-defined ASM */
typedef enum { SLAB_START=0, SLAB_MIDDLE=1, SLAB_END=2 } SlabType;

typedef struct {
  AppCtx    *user;
  DM         dm;
  SNES       snes;

  PetscInt   nx, nt, blksize, overlap, Nsub, interface_width;

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

#endif /* APPCTX_H */