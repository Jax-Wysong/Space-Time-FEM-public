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

	PetscBool slab_pc_nonlinear; /* if true, include nonlinear terms in slab pc */

  PetscBool slab_pc_ras;  /* if true: RAS; if false: pure ASM */

  /* Dirichlet/Neumann boundary conditions */
  PetscBool interface_BC_all; /* if true, impose D and N */
  PetscBool interface_BC_dirichlet; /* if true, impose D-like (phi, chi) 0 at interface*/
  PetscBool interface_BC_neumann;   /* if true, impose N-like (u, v) 0 at interface */
  PetscBool interface_BC_none;      /* if true, impose no BC at interface (for testing) */

  PetscBool interface_D_N_alternate; /* if true, alternate D/N on even/odd interfaces (for testing) */

  PetscBool interface_BC_robin; /* if true, impose Robin BC at interface (for testing) */
  PetscReal robin_alpha; /* alpha value for Robin BC */

	} AppCtx;


	/* Define context for user-defined ASM */
typedef enum { SLAB_START=0, SLAB_MIDDLE=1, SLAB_END=2 } SlabType;

typedef struct {
  AppCtx    *user;
  DM         dm;
  SNES       snes;
  PetscBool  use_nonlinear;

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

  /* BC interface condtions */
  PetscBool interface_BC_all;
  PetscBool interface_BC_dirichlet;
  PetscBool interface_BC_neumann;
  PetscBool interface_BC_none;
  PetscBool interface_D_N_alternate;
  PetscBool interface_BC_robin;
  PetscReal robin_alpha;


} SampleShellPC;

#endif /* APPCTX_H */