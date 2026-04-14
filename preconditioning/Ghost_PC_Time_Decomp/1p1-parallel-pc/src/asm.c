#include<petscmat.h>
#include<petscmath.h>
#include<petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include<petscksp.h>
#include<petscvec.h>
#include <petscsnes.h>
#include <petscsys.h>
#include"petscsys.h"
#include"petscviewer.h"
#include "appctx.h"
#include "asm.h"
#include "stiffness.h"
#include "nonlin.h"


static inline PetscInt gid(PetscInt nx, PetscInt t, PetscInt x, PetscInt c)
{
  return 4*(t*nx + x) + c; /* interleaved dofs: phi=0, u=1, chi=2, v=3*/
}

/* for additive FS */
static inline PetscInt gid2(PetscInt nx, PetscInt t, PetscInt x, PetscInt c)
{ 
  return 2*(t*nx + x) + c;
}


static PetscErrorCode PackSlabFromDMDALocal(
    DM dm, Vec xloc,
    PetscInt nx,
    PetscInt t0_global, PetscInt t1_global, /* inclusive global time range */
    Vec x_slab)
{
  PetscErrorCode ierr;
  PetscInt gxs, gys, gxm, gym;
  PetscScalar ***xa = NULL;     /* [t][x][c] */
  PetscScalar *xs = NULL;

  PetscFunctionBegin;

  ierr = DMDAGetGhostCorners(dm, &gxs, &gys, NULL, &gxm, &gym, NULL);CHKERRQ(ierr);

  if (t0_global < gys || t1_global >= gys + gym) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
            "PackSlabFromDMDALocal: requested time range not in ghost region");
  }
  if (0 < gxs || (nx-1) >= gxs + gxm) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
            "PackSlabFromDMDALocal: x range not in ghost region");
  }

  ierr = DMDAVecGetArrayDOFRead(dm, xloc, &xa);CHKERRQ(ierr);
  ierr = VecGetArray(x_slab, &xs);CHKERRQ(ierr);

  PetscInt ntloc = t1_global - t0_global + 1;
  for (PetscInt tt = 0; tt < ntloc; ++tt) {
    PetscInt t = t0_global + tt;
    for (PetscInt x = 0; x < nx; ++x) {
      for (PetscInt c = 0; c < 4; ++c) {
        xs[ gid(nx, tt, x, c) ] = xa[t][x][c];
      }
    }
  }

  ierr = VecRestoreArray(x_slab, &xs);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(dm, xloc, &xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* for additive FS PackSlabFromDMDALocal */
static PetscErrorCode PackSlabFieldBlock(
    DM dm, Vec xloc,
    PetscInt nx,
    PetscInt t0_global, PetscInt t1_global, /* inclusive global time range */
    PetscInt dof0, Vec x_slab)
{
  PetscErrorCode ierr;
  PetscInt gxs, gys, gxm, gym;
  PetscScalar ***xa = NULL;     /* [t][x][c] */
  PetscScalar *xs = NULL;

  PetscFunctionBegin;

  ierr = DMDAGetGhostCorners(dm, &gxs, &gys, NULL, &gxm, &gym, NULL);CHKERRQ(ierr);

  if (t0_global < gys || t1_global >= gys + gym) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
            "PackSlabFromDMDALocal: requested time range not in ghost region");
  }
  if (0 < gxs || (nx-1) >= gxs + gxm) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
            "PackSlabFromDMDALocal: x range not in ghost region");
  }

  ierr = DMDAVecGetArrayDOFRead(dm, xloc, &xa);CHKERRQ(ierr);
  ierr = VecGetArray(x_slab, &xs);CHKERRQ(ierr);

  PetscInt ntloc = t1_global - t0_global + 1;
  for (PetscInt tt = 0; tt < ntloc; ++tt) {
    PetscInt t = t0_global + tt;
    for (PetscInt x = 0; x < nx; ++x) {
      for (PetscInt c = 0; c < 2; ++c){
        xs[ gid2(nx, tt, x, c) ] = xa[t][x][dof0 + c];
      }
    }
  }

  ierr = VecRestoreArray(x_slab, &xs);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(dm, xloc, &xa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode AddBackInteriorToGlobal_DMDARows(
    DM dm, Vec y,
    PetscInt nx,
    PetscInt t0_global,             /* slab solve window start (global t) */
    PetscInt t_add0, PetscInt t_add1,/* interior add-back window (global t, inclusive) */
    Vec y_slab)                     /* seq slab solution */
{
  PetscErrorCode ierr;
  PetscInt xs, ys, xm, ym;
  PetscScalar ***ya = NULL;
  const PetscScalar *yslab = NULL;

  PetscFunctionBegin;

  ierr = DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);

  /* intersect interior add-back range with what this rank owns in time */
  PetscInt own_t0 = ys;
  PetscInt own_t1 = ys + ym - 1;

  PetscInt w0 = PetscMax(t_add0, own_t0);
  PetscInt w1 = PetscMin(t_add1, own_t1);
  if (w0 > w1) PetscFunctionReturn(0);

  ierr = DMDAVecGetArrayDOF(dm, y, &ya);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y_slab, &yslab);CHKERRQ(ierr);

  for (PetscInt t = w0; t <= w1; ++t) {
    PetscInt tt = t - t0_global; /* slab-local time index */
    for (PetscInt x = xs; x < xs + xm; ++x) {
      for (PetscInt c = 0; c < 4; ++c) {
        ya[t][x][c] += yslab[ gid(nx, tt, x, c) ];
      }
    }
  }

  ierr = VecRestoreArrayRead(y_slab, &yslab);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(dm, y, &ya);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* for additive FS AddBackInteriorToGlobal */
static PetscErrorCode AddBackFieldBlockRAS(
    DM dm, Vec y,
    PetscInt nx,
    PetscInt t0_global,             /* slab solve window start (global t) */
    PetscInt t_add0, PetscInt t_add1,/* interior add-back window (global t, inclusive) */
    PetscInt dof0, Vec y_slab)                     /* seq slab solution */
{
  PetscErrorCode ierr;
  PetscInt xs, ys, xm, ym;
  PetscScalar ***ya = NULL;
  const PetscScalar *yslab = NULL;

  PetscFunctionBegin;

  ierr = DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);

  /* intersect interior add-back range with what this rank owns in time */
  PetscInt own_t0 = ys;
  PetscInt own_t1 = ys + ym - 1;

  PetscInt w0 = PetscMax(t_add0, own_t0);
  PetscInt w1 = PetscMin(t_add1, own_t1);
  if (w0 > w1) PetscFunctionReturn(0);

  ierr = DMDAVecGetArrayDOF(dm, y, &ya);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y_slab, &yslab);CHKERRQ(ierr);

  for (PetscInt t = w0; t <= w1; ++t) {
    PetscInt tt = t - t0_global; /* slab-local time index */
    for (PetscInt x = xs; x < xs + xm; ++x) {
      for (PetscInt c = 0; c < 2; ++c) {
        ya[t][x][dof0+c] += yslab[ gid2(nx, tt, x, c) ];
      }
    }
  }

  ierr = VecRestoreArrayRead(y_slab, &yslab);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(dm, y, &ya);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode PackSlabFromGlobal(
    DM dm, Vec g,
    Vec loc,            /* ghosted local workspace (must match dm local layout) */
    PetscInt nx,
    PetscInt t0_global, PetscInt t1_global,
    Vec slab_out)
{
  PetscErrorCode ierr;
  ierr = DMGlobalToLocalBegin(dm, g, INSERT_VALUES, loc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm, g, INSERT_VALUES, loc);CHKERRQ(ierr);
  ierr = PackSlabFromDMDALocal(dm, loc, nx, t0_global, t1_global, slab_out);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode CreateLocalSlabObjects(SampleShellPC *shell, PetscInt nloc,
                                             const char *prefix,
                                             Mat *A, KSP *ksp, Vec *x, Vec *y)
{
  PetscErrorCode ierr;

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nloc, nloc, 40, NULL, A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_SELF, ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(*ksp, prefix);CHKERRQ(ierr);
  ierr = KSPSetType(*ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(*ksp, *A, *A);CHKERRQ(ierr);
  {
    PC pcsub;
    ierr = KSPGetPC(*ksp, &pcsub);CHKERRQ(ierr);
    ierr = PCSetType(pcsub, PCLU);CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(*ksp);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, nloc, x);CHKERRQ(ierr);
  ierr = VecDuplicate(*x, y);CHKERRQ(ierr);

  return 0;
}

static PetscErrorCode UnpackSlabToDMDALocalAdd(
    DM dm, Vec yloc,              /* ghosted local vec to write into */
    PetscInt nx,
    PetscInt t0_global, PetscInt t1_global, /* slab window in GLOBAL t */
    Vec y_slab)                    /* sequential slab vector */
{
  PetscErrorCode ierr;
  PetscInt gxs, gys, gxm, gym;
  PetscScalar ***ya = NULL;        /* [t][x][c] */
  const PetscScalar *ys = NULL;

  PetscFunctionBegin;

  ierr = DMDAGetGhostCorners(dm, &gxs, &gys, NULL, &gxm, &gym, NULL);CHKERRQ(ierr);

  /* ensure requested time window is in this rank's ghost region */
  if (t0_global < gys || t1_global >= gys + gym) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
            "UnpackSlabToDMDALocalAdd: requested time window not in ghost region");
  }
  /* require full x range to be present */
  if (0 < gxs || (nx-1) >= gxs + gxm) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
            "UnpackSlabToDMDALocalAdd: x range not in ghost region");
  }

  ierr = DMDAVecGetArrayDOF(dm, yloc, &ya);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y_slab, &ys);CHKERRQ(ierr);

  PetscInt ntloc = t1_global - t0_global + 1;
  for (PetscInt tt = 0; tt < ntloc; ++tt) {
    PetscInt t = t0_global + tt;
    for (PetscInt x = 0; x < nx; ++x) {
      for (PetscInt c = 0; c < 4; ++c) {
        ya[t][x][c] += ys[ gid(nx, tt, x, c) ];
      }
    }
  }

  ierr = VecRestoreArrayRead(y_slab, &ys);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(dm, yloc, &ya);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode AddBackFullWindow_ASM(
    SampleShellPC *shell,
    Vec y_global,
    PetscInt nx,
    PetscInt t0_global, PetscInt t1_global, /* full slab window */
    Vec y_slab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* start with zero local contribution buffer */
  ierr = VecZeroEntries(shell->yloc);CHKERRQ(ierr);

  /* put the slab solution into the local DMDA vector (ghosted layout) */
  ierr = UnpackSlabToDMDALocalAdd(shell->dm, shell->yloc, nx, t0_global, t1_global, y_slab);CHKERRQ(ierr);

  /* accumulate into the global y, summing overlaps across ranks */
  ierr = DMLocalToGlobalBegin(shell->dm, shell->yloc, ADD_VALUES, y_global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (shell->dm, shell->yloc, ADD_VALUES, y_global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode PCSetUp_SampleShell(PC pc)
{
  PetscErrorCode ierr;
  SampleShellPC *shell = NULL;
  AppCtx        *user  = NULL;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&shell);CHKERRQ(ierr);
  user = shell->user;

  shell->nx      = user->nx;
  shell->nt      = user->nt;
  shell->overlap = user->overlap;
  shell->Nsub    = user->Nsub;

  /* blksize */
  shell->blksize = user->nt / user->Nsub;

  MPI_Comm_rank(PETSC_COMM_WORLD, &shell->rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &shell->size);

  /* always run parallel with size == Nsub for now */
  if (shell->Nsub < 2) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Require Nsub >= 2 in parallel-only mode");
  }
  if (shell->size != shell->Nsub) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Require MPI size == Nsub");
  }

  if (user->nt % user->Nsub != 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "nt must be evenly divisible by Nsub");
  }


  if (!shell->xloc) { ierr = DMCreateLocalVector(shell->dm, &shell->xloc);CHKERRQ(ierr); }
  if (!shell->Uloc) { ierr = DMCreateLocalVector(shell->dm, &shell->Uloc);CHKERRQ(ierr); }
  if (!shell->yloc) { ierr = DMCreateLocalVector(shell->dm, &shell->yloc);CHKERRQ(ierr); }


  const PetscInt nx      = shell->nx;
  const PetscInt nt      = shell->nt;
  const PetscInt ov      = shell->overlap;
  const PetscInt Nsub    = shell->Nsub;
  const PetscInt blksize = shell->blksize;

  /* Decide slab type for this rank */
  if (shell->rank == 0) shell->slabtype = SLAB_START;
  else if (shell->rank == Nsub-1) shell->slabtype = SLAB_END;
  else shell->slabtype = SLAB_MIDDLE;

  /* --------------------------------------------
     Refresh path: if already built and linear, bail
     -------------------------------------------- */
  if (shell->ksp_start || shell->ksp_middle || shell->ksp_end || shell->ksp_phiu_start || shell->ksp_phiu_middle || shell->ksp_phiu_end) {

    if (!shell->use_nonlinear) PetscFunctionReturn(0);

    /* Only rebuild once per Newton step; skip redundant cascaded PCSetUp calls
       (e.g. FieldSplit inner PCs triggering additional outer PCSetUp events). */
    PetscInt cur_snes_iter;
    ierr = SNESGetIterationNumber(shell->snes, &cur_snes_iter);CHKERRQ(ierr);
    if (cur_snes_iter == shell->last_snes_iter) PetscFunctionReturn(0);
    shell->last_snes_iter = cur_snes_iter;

    /* Nonlinear refresh: rebuild only THIS rank’s slab matrix using current SNES solution */
    Vec Ucur = NULL;
    ierr = SNESGetSolution(shell->snes, &Ucur);CHKERRQ(ierr);

    if (shell->use_fs_ras) {
      /* FS-RAS refresh: pack full 4-DOF slab to supply phi and chi to stiff2_block */
      if (shell->slabtype == SLAB_START) {
        PetscInt t0 = 0, t1 = blksize + ov - 1;
        PetscInt nt_slab = blksize + ov;
        Vec x4;
        ierr = VecCreateSeq(PETSC_COMM_SELF, 4*nx*nt_slab, &x4);CHKERRQ(ierr);
        ierr = PackSlabFromGlobal(shell->dm, Ucur, shell->Uloc, nx, t0, t1, x4);CHKERRQ(ierr);
        const PetscScalar *Uslab4 = NULL;
        ierr = VecGetArrayRead(x4, &Uslab4);CHKERRQ(ierr);
        ierr = stiff2_block(shell->A_phiu_start, nx, nt_slab, shell->user, PETSC_TRUE, Uslab4, 0);CHKERRQ(ierr);
        ierr = stiff2_block(shell->A_chiv_start, nx, nt_slab, shell->user, PETSC_TRUE, Uslab4, 2);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(x4, &Uslab4);CHKERRQ(ierr);
        ierr = VecDestroy(&x4);CHKERRQ(ierr);
        ierr = KSPSetUp(shell->ksp_phiu_start);CHKERRQ(ierr);
        ierr = KSPSetUp(shell->ksp_chiv_start);CHKERRQ(ierr);
      } else if (shell->slabtype == SLAB_MIDDLE) {
        PetscInt s  = shell->rank;
        PetscInt t0 = s*blksize - 1 - ov;
        PetscInt t1 = (s+1)*blksize + ov - 1;
        PetscInt nt_slab = blksize + 2*ov + 1;
        Vec x4;
        ierr = VecCreateSeq(PETSC_COMM_SELF, 4*nx*nt_slab, &x4);CHKERRQ(ierr);
        ierr = PackSlabFromGlobal(shell->dm, Ucur, shell->Uloc, nx, t0, t1, x4);CHKERRQ(ierr);
        const PetscScalar *Uslab4 = NULL;
        ierr = VecGetArrayRead(x4, &Uslab4);CHKERRQ(ierr);
        ierr = stiff2_block(shell->A_phiu_middle, nx, nt_slab, shell->user, PETSC_TRUE, Uslab4, 0);CHKERRQ(ierr);
        ierr = stiff2_block(shell->A_chiv_middle, nx, nt_slab, shell->user, PETSC_TRUE, Uslab4, 2);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(x4, &Uslab4);CHKERRQ(ierr);
        ierr = VecDestroy(&x4);CHKERRQ(ierr);
        ierr = KSPSetUp(shell->ksp_phiu_middle);CHKERRQ(ierr);
        ierr = KSPSetUp(shell->ksp_chiv_middle);CHKERRQ(ierr);
      } else { /* SLAB_END */
        PetscInt t0 = (Nsub-1)*blksize - 1 - ov;
        PetscInt t1 = nt - 1;
        PetscInt nt_slab = blksize + ov + 1;
        Vec x4;
        ierr = VecCreateSeq(PETSC_COMM_SELF, 4*nx*nt_slab, &x4);CHKERRQ(ierr);
        ierr = PackSlabFromGlobal(shell->dm, Ucur, shell->Uloc, nx, t0, t1, x4);CHKERRQ(ierr);
        const PetscScalar *Uslab4 = NULL;
        ierr = VecGetArrayRead(x4, &Uslab4);CHKERRQ(ierr);
        ierr = stiff2_block(shell->A_phiu_end, nx, nt_slab, shell->user, PETSC_TRUE, Uslab4, 0);CHKERRQ(ierr);
        ierr = stiff2_block(shell->A_chiv_end, nx, nt_slab, shell->user, PETSC_TRUE, Uslab4, 2);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(x4, &Uslab4);CHKERRQ(ierr);
        ierr = VecDestroy(&x4);CHKERRQ(ierr);
        ierr = KSPSetUp(shell->ksp_phiu_end);CHKERRQ(ierr);
        ierr = KSPSetUp(shell->ksp_chiv_end);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }

    if (shell->slabtype == SLAB_START) {
      PetscInt t0 = 0, t1 = blksize + ov - 1;

      ierr = PackSlabFromGlobal(shell->dm, Ucur, shell->Uloc, nx, t0, t1, shell->x_start);CHKERRQ(ierr);

      const PetscScalar *Uslab = NULL;
      ierr = VecGetArrayRead(shell->x_start, &Uslab);CHKERRQ(ierr);
      ierr = stiff2(shell->A_start, nx, blksize + ov, shell->user, PETSC_TRUE, Uslab);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(shell->x_start, &Uslab);CHKERRQ(ierr);

      ierr = KSPSetUp(shell->ksp_start);CHKERRQ(ierr);

    } else if (shell->slabtype == SLAB_MIDDLE) {
      PetscInt s  = shell->rank;
      PetscInt t0 = s*blksize - 1 - ov;
      PetscInt t1 = (s+1)*blksize + ov - 1;

      ierr = PackSlabFromGlobal(shell->dm, Ucur, shell->Uloc, nx, t0, t1, shell->x_middle);CHKERRQ(ierr);

      const PetscScalar *Uslab = NULL;
      ierr = VecGetArrayRead(shell->x_middle, &Uslab);CHKERRQ(ierr);
      ierr = stiff2(shell->A_middle, nx, blksize + 2*ov + 1, shell->user, PETSC_TRUE, Uslab);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(shell->x_middle, &Uslab);CHKERRQ(ierr);

      ierr = KSPSetUp(shell->ksp_middle);CHKERRQ(ierr);

    } else { /* SLAB_END */
      PetscInt t0 = (Nsub-1)*blksize - 1 - ov;
      PetscInt t1 = nt - 1;

      ierr = PackSlabFromGlobal(shell->dm, Ucur, shell->Uloc, nx, t0, t1, shell->x_end);CHKERRQ(ierr);

      const PetscScalar *Uslab = NULL;
      ierr = VecGetArrayRead(shell->x_end, &Uslab);CHKERRQ(ierr);
      ierr = stiff2(shell->A_end, nx, blksize + ov + 1, shell->user, PETSC_TRUE, Uslab);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(shell->x_end, &Uslab);CHKERRQ(ierr);

      ierr = KSPSetUp(shell->ksp_end);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
  }

  /* --------------------------------------------
     First-time build: allocate only this rank’s slab objects
     -------------------------------------------- */

  if (shell->slabtype == SLAB_START) {

    if(shell->use_fs_ras == PETSC_TRUE){
      PetscInt nloc_block = 2*nx*(blksize + ov);
      /* create local Mat, KSP, Vecs for this slab */
      ierr = CreateLocalSlabObjects(shell, nloc_block, "sub_start_", &shell->A_phiu_start, &shell->ksp_phiu_start,
                                  &shell->xp_start, &shell->yp_start);CHKERRQ(ierr);
      ierr = CreateLocalSlabObjects(shell, nloc_block, "sub_start_", &shell->A_chiv_start, &shell->ksp_chiv_start,
                                  &shell->xc_start, &shell->yc_start);CHKERRQ(ierr);
      /* Sets up blocks for (phi,u) and (chi,v) for fieldsplit */
      ierr = MatSetBlockSize(shell->A_phiu_start, 2);CHKERRQ(ierr);   
      ierr = MatSetBlockSize(shell->A_chiv_start, 2);CHKERRQ(ierr);   

      /* build A using the same logic from FormJacobian */
      ierr = stiff2_block(shell->A_phiu_start, nx, blksize + ov, shell->user, PETSC_TRUE, NULL, 0);CHKERRQ(ierr);
      ierr = stiff2_block(shell->A_chiv_start, nx, blksize + ov, shell->user, PETSC_TRUE, NULL, 2);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_phiu_start);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_chiv_start);CHKERRQ(ierr);

    } else {
      PetscInt nloc = 4*nx*(blksize + ov);
      /* create local Mat, KSP, Vecs for this slab */
      ierr = CreateLocalSlabObjects(shell, nloc, "sub_start_", &shell->A_start, &shell->ksp_start,
                                  &shell->x_start, &shell->y_start);CHKERRQ(ierr);
      /* Sets up blocks for (phi, u, chi, v) for fieldsplit */
      ierr = MatSetBlockSize(shell->A_start, 4);CHKERRQ(ierr);   

      /* build A using the same logic from FormJacobian */
      ierr = stiff2(shell->A_start, nx, blksize + ov, user, PETSC_TRUE, NULL);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_start);CHKERRQ(ierr);
    }

  } else if (shell->slabtype == SLAB_MIDDLE) {
    if (shell->use_fs_ras == PETSC_TRUE){
      PetscInt nloc_block = 2*nx*(blksize + 2*ov + 1);
      /* create local Mat, KSP, Vecs for this slab */
      ierr = CreateLocalSlabObjects(shell, nloc_block, "sub_mid_", &shell->A_phiu_middle, &shell->ksp_phiu_middle,
                                  &shell->xp_middle, &shell->yp_middle);CHKERRQ(ierr);
      ierr = CreateLocalSlabObjects(shell, nloc_block, "sub_mid_", &shell->A_chiv_middle, &shell->ksp_chiv_middle,
                                  &shell->xc_middle, &shell->yc_middle);CHKERRQ(ierr);
      /* Sets up blocks for (phi,u) and (chi,v) for fieldsplit */
      ierr = MatSetBlockSize(shell->A_phiu_middle, 2);CHKERRQ(ierr);   
      ierr = MatSetBlockSize(shell->A_chiv_middle, 2);CHKERRQ(ierr);   

      /* build A using the same logic from FormJacobian */
      ierr = stiff2_block(shell->A_phiu_middle, nx, blksize + 2*ov + 1, shell->user, PETSC_TRUE, NULL, 0);CHKERRQ(ierr);
      ierr = stiff2_block(shell->A_chiv_middle, nx, blksize + 2*ov + 1, shell->user, PETSC_TRUE, NULL, 2);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_phiu_middle);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_chiv_middle);CHKERRQ(ierr);

    } else {
    PetscInt nloc = 4*nx*(blksize + 2*ov + 1);
    /* creat local Mat, KSP, Vecs for this slab */
    ierr = CreateLocalSlabObjects(shell, nloc, "sub_mid_", &shell->A_middle, &shell->ksp_middle,
                                &shell->x_middle, &shell->y_middle);CHKERRQ(ierr);
    /* Sets up blocks for (phi, u, chi, v) for fieldsplit */
    ierr = MatSetBlockSize(shell->A_middle, 4);CHKERRQ(ierr);  

    /* build A using the same logic from FormJacobian */
    ierr = stiff2(shell->A_middle, nx, blksize + 2*ov + 1, user, PETSC_TRUE, NULL);CHKERRQ(ierr);
    ierr = KSPSetUp(shell->ksp_middle);CHKERRQ(ierr);
    }

  } else { /* SLAB_END */
    if (shell->use_fs_ras == PETSC_TRUE){
      PetscInt nloc_block = 2*nx*(blksize + ov + 1);
      /* create local Mat, KSP, Vecs for this slab */
      ierr = CreateLocalSlabObjects(shell, nloc_block, "sub_end_", &shell->A_phiu_end, &shell->ksp_phiu_end,
                                  &shell->xp_end, &shell->yp_end);CHKERRQ(ierr);
      ierr = CreateLocalSlabObjects(shell, nloc_block, "sub_end_", &shell->A_chiv_end, &shell->ksp_chiv_end,
                                  &shell->xc_end, &shell->yc_end);CHKERRQ(ierr);
      /* Sets up blocks for (phi,u) and (chi,v) for fieldsplit */
      ierr = MatSetBlockSize(shell->A_phiu_end, 2);CHKERRQ(ierr);   
      ierr = MatSetBlockSize(shell->A_chiv_end, 2);CHKERRQ(ierr);   

      /* build A using the same logic from FormJacobian */
      ierr = stiff2_block(shell->A_phiu_end, nx, blksize + ov + 1, shell->user, PETSC_TRUE, NULL, 0);CHKERRQ(ierr);
      ierr = stiff2_block(shell->A_chiv_end, nx, blksize + ov + 1, shell->user, PETSC_TRUE, NULL, 2);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_phiu_end);CHKERRQ(ierr);
      ierr = KSPSetUp(shell->ksp_chiv_end);CHKERRQ(ierr);

    } else {
    PetscInt nloc = 4*nx*(blksize + ov + 1);
    /* create local Mat, KSP, Vecs for this slab */
    ierr = CreateLocalSlabObjects(shell, nloc, "sub_end_", &shell->A_end, &shell->ksp_end,
                                &shell->x_end, &shell->y_end);CHKERRQ(ierr);
    /* Sets up blocks for (phi, u, chi, v) for fieldsplit */
    ierr = MatSetBlockSize(shell->A_end, 4);CHKERRQ(ierr); 

    /* build A using the same logic from FormJacobian */
    ierr = stiff2(shell->A_end, nx, blksize + ov + 1, user, PETSC_TRUE, NULL);CHKERRQ(ierr);
    ierr = KSPSetUp(shell->ksp_end);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}


PetscErrorCode PCApply_SampleShell(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;
  SampleShellPC  *shell = NULL;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&shell);CHKERRQ(ierr);

  ierr = VecZeroEntries(y);CHKERRQ(ierr);

  const PetscInt Nsub    = shell->Nsub;
  const PetscInt nx      = shell->nx;
  const PetscInt blksize = shell->blksize;
  const PetscInt ov      = shell->overlap;

  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  /* fill ghosted local copy of x once */
  ierr = DMGlobalToLocalBegin(shell->dm, x, INSERT_VALUES, shell->xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (shell->dm, x, INSERT_VALUES, shell->xloc);CHKERRQ(ierr);

  if (Nsub == 1) {
    PetscInt t0 = 0, t1 = shell->nt - 1;

    ierr = PackSlabFromDMDALocal(shell->dm, shell->xloc, nx, t0, t1, shell->x_start);CHKERRQ(ierr);

    ierr = VecZeroEntries(shell->y_start);CHKERRQ(ierr);
    ierr = KSPSolve(shell->ksp_start, shell->x_start, shell->y_start);CHKERRQ(ierr);

    if (shell->use_ras) {
      ierr = AddBackInteriorToGlobal_DMDARows(shell->dm, y, nx, t0, 0, shell->nt-1, shell->y_start);CHKERRQ(ierr);
    } else {
      /* pure ASM: add back everything (here it's all time anyway) */
      ierr = AddBackFullWindow_ASM(shell, y, nx, t0, t1, shell->y_start);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
  }

  if (size != (PetscMPIInt)Nsub) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"This layout assumes MPI size == Nsub");
  }

  if (rank == 0) {
    PetscInt t0 = 0;
    PetscInt t1 = blksize + ov - 1;
    if (shell->use_fs_ras == PETSC_TRUE){
      /* (phi, u)*/
      ierr = PackSlabFieldBlock(shell->dm, shell->xloc, nx, t0, t1, 0, shell->xp_start);CHKERRQ(ierr);
      ierr =VecZeroEntries(shell->yp_start);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_phiu_start, shell->xp_start, shell->yp_start);CHKERRQ(ierr);
      ierr = AddBackFieldBlockRAS(shell->dm, y, nx, t0, 0, blksize-1, 0, shell->yp_start);CHKERRQ(ierr);

      /* (chi, v) */
      ierr = PackSlabFieldBlock(shell->dm, shell->xloc, nx, t0, t1, 2, shell->xc_start);CHKERRQ(ierr);
      ierr = VecZeroEntries(shell->yc_start);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_chiv_start, shell->xc_start, shell->yc_start);CHKERRQ(ierr);
      ierr = AddBackFieldBlockRAS(shell->dm, y, nx, t0, 0, blksize-1, 2, shell->yc_start);CHKERRQ(ierr);
    } else {
      /* Change x_start from DMDA ordering to local/sequential ordering from t0 - t1*/
      ierr = PackSlabFromDMDALocal(shell->dm, shell->xloc, nx, t0, t1, shell->x_start);CHKERRQ(ierr);

      ierr = VecZeroEntries(shell->y_start);CHKERRQ(ierr);
      /* solve A_start y_start = x_start sequentially*/
      ierr = KSPSolve(shell->ksp_start, shell->x_start, shell->y_start);CHKERRQ(ierr);

      if (shell->use_ras) {
        ierr = AddBackInteriorToGlobal_DMDARows(shell->dm, y, nx, t0, 0, blksize-1, shell->y_start);CHKERRQ(ierr);
      } else {
        ierr = AddBackFullWindow_ASM(shell, y, nx, t0, t1, shell->y_start);CHKERRQ(ierr);
      }
    }
  }

  if (rank > 0 && rank < Nsub-1) {
    PetscInt s  = rank;
    PetscInt t0 = s*blksize - 1 - ov;
    PetscInt t1 = (s+1)*blksize + ov - 1;

    if (shell->use_fs_ras == PETSC_TRUE){
      /* (phi, u)*/
      ierr = PackSlabFieldBlock(shell->dm, shell->xloc, nx, t0, t1, 0, shell->xp_middle);CHKERRQ(ierr);
      ierr = VecZeroEntries(shell->yp_middle);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_phiu_middle, shell->xp_middle, shell->yp_middle);CHKERRQ(ierr);
      ierr = AddBackFieldBlockRAS(shell->dm, y, nx, t0, s*blksize, (s+1)*blksize-1, 0, shell->yp_middle);CHKERRQ(ierr);

      /* (chi, v) */
      ierr = PackSlabFieldBlock(shell->dm, shell->xloc, nx, t0, t1, 2, shell->xc_middle);CHKERRQ(ierr);
      ierr = VecZeroEntries(shell->yc_middle);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_chiv_middle, shell->xc_middle, shell->yc_middle);CHKERRQ(ierr);
      ierr = AddBackFieldBlockRAS(shell->dm, y, nx, t0, s*blksize, (s+1)*blksize-1, 2, shell->yc_middle);CHKERRQ(ierr);

    } else {
      ierr = PackSlabFromDMDALocal(shell->dm, shell->xloc, nx, t0, t1, shell->x_middle);CHKERRQ(ierr);

      ierr = VecZeroEntries(shell->y_middle);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_middle, shell->x_middle, shell->y_middle);CHKERRQ(ierr);

      if (shell->use_ras) {
        ierr = AddBackInteriorToGlobal_DMDARows(shell->dm, y, nx, t0,
                                              s*blksize, (s+1)*blksize-1,
                                              shell->y_middle);CHKERRQ(ierr);
      } else {
        ierr = AddBackFullWindow_ASM(shell, y, nx, t0, t1, shell->y_middle);CHKERRQ(ierr);
      }
    }
  }

  if (rank == Nsub-1) {
    PetscInt t0 = (Nsub-1)*blksize - 1 - ov;
    PetscInt t1 = shell->nt - 1;

    if (shell->use_fs_ras == PETSC_TRUE){
      /* (phi, u)*/
      ierr = PackSlabFieldBlock(shell->dm, shell->xloc, nx, t0, t1, 0, shell->xp_end);CHKERRQ(ierr);
      ierr = VecZeroEntries(shell->yp_end);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_phiu_end, shell->xp_end, shell->yp_end);CHKERRQ(ierr);
      ierr = AddBackFieldBlockRAS(shell->dm, y, nx, t0, (Nsub-1)*blksize, shell->nt-1, 0, shell->yp_end);CHKERRQ(ierr);

      /* (chi, v) */
      ierr = PackSlabFieldBlock(shell->dm, shell->xloc, nx, t0, t1, 2, shell->xc_end);CHKERRQ(ierr);
      ierr = VecZeroEntries(shell->yc_end);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_chiv_end, shell->xc_end, shell->yc_end);CHKERRQ(ierr);
      ierr = AddBackFieldBlockRAS(shell->dm, y, nx, t0, (Nsub-1)*blksize, shell->nt-1, 2, shell->yc_end);CHKERRQ(ierr);

    } else {
      ierr = PackSlabFromDMDALocal(shell->dm, shell->xloc, nx, t0, t1, shell->x_end);CHKERRQ(ierr);

      ierr = VecZeroEntries(shell->y_end);CHKERRQ(ierr);
      ierr = KSPSolve(shell->ksp_end, shell->x_end, shell->y_end);CHKERRQ(ierr);

      if (shell->use_ras) {
        ierr = AddBackInteriorToGlobal_DMDARows(shell->dm, y, nx, t0,
                                              (Nsub-1)*blksize, shell->nt-1,
                                              shell->y_end);CHKERRQ(ierr);
      } else {
        ierr = AddBackFullWindow_ASM(shell, y, nx, t0, t1, shell->y_end);CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);
}


PetscErrorCode PCDestroy_SampleShell(PC pc)
{
  PetscErrorCode ierr;
  SampleShellPC *shell = NULL;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&shell);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc, NULL);CHKERRQ(ierr);
  if (!shell) PetscFunctionReturn(0);

  if (shell->ksp_start)   { ierr = KSPDestroy(&shell->ksp_start);CHKERRQ(ierr); }
  if (shell->ksp_middle)  { ierr = KSPDestroy(&shell->ksp_middle);CHKERRQ(ierr); }
  if (shell->ksp_end)     { ierr = KSPDestroy(&shell->ksp_end);CHKERRQ(ierr); }

  if (shell->x_start)   { ierr = VecDestroy(&shell->x_start);CHKERRQ(ierr); }
  if (shell->y_start)   { ierr = VecDestroy(&shell->y_start);CHKERRQ(ierr); }
  if (shell->x_middle)  { ierr = VecDestroy(&shell->x_middle);CHKERRQ(ierr); }
  if (shell->y_middle)  { ierr = VecDestroy(&shell->y_middle);CHKERRQ(ierr); }
  if (shell->x_end)     { ierr = VecDestroy(&shell->x_end);CHKERRQ(ierr); }
  if (shell->y_end)     { ierr = VecDestroy(&shell->y_end);CHKERRQ(ierr); }

  if (shell->xloc)      { ierr = VecDestroy(&shell->xloc);CHKERRQ(ierr); }
  if (shell->Uloc)      { ierr = VecDestroy(&shell->Uloc);CHKERRQ(ierr); }
  if (shell->yloc)      { ierr = VecDestroy(&shell->yloc);CHKERRQ(ierr); }

  if (shell->A_start)   { ierr = MatDestroy(&shell->A_start);CHKERRQ(ierr); }
  if (shell->A_middle)  { ierr = MatDestroy(&shell->A_middle);CHKERRQ(ierr); }
  if (shell->A_end)     { ierr = MatDestroy(&shell->A_end);CHKERRQ(ierr); }

  if (shell->ksp_phiu_start)   { ierr = KSPDestroy(&shell->ksp_phiu_start);CHKERRQ(ierr); }
  if (shell->ksp_phiu_middle)  { ierr = KSPDestroy(&shell->ksp_phiu_middle);CHKERRQ(ierr); }
  if (shell->ksp_phiu_end)     { ierr = KSPDestroy(&shell->ksp_phiu_end);CHKERRQ(ierr); }

  if (shell->xp_start)   { ierr = VecDestroy(&shell->xp_start);CHKERRQ(ierr); }
  if (shell->yp_start)   { ierr = VecDestroy(&shell->yp_start);CHKERRQ(ierr); }
  if (shell->xp_middle)  { ierr = VecDestroy(&shell->xp_middle);CHKERRQ(ierr); }
  if (shell->yp_middle)  { ierr = VecDestroy(&shell->yp_middle);CHKERRQ(ierr); }
  if (shell->xp_end)     { ierr = VecDestroy(&shell->xp_end);CHKERRQ(ierr); }
  if (shell->yp_end)     { ierr = VecDestroy(&shell->yp_end);CHKERRQ(ierr); }

  if (shell->A_phiu_start)   { ierr = MatDestroy(&shell->A_phiu_start);CHKERRQ(ierr); }
  if (shell->A_phiu_middle)  { ierr = MatDestroy(&shell->A_phiu_middle);CHKERRQ(ierr); }
  if (shell->A_phiu_end)     { ierr = MatDestroy(&shell->A_phiu_end);CHKERRQ(ierr); }

  if (shell->ksp_chiv_start)   { ierr = KSPDestroy(&shell->ksp_chiv_start);CHKERRQ(ierr); }
  if (shell->ksp_chiv_middle)  { ierr = KSPDestroy(&shell->ksp_chiv_middle);CHKERRQ(ierr); }
  if (shell->ksp_chiv_end)     { ierr = KSPDestroy(&shell->ksp_chiv_end);CHKERRQ(ierr); }

  if (shell->xc_start)   { ierr = VecDestroy(&shell->xc_start);CHKERRQ(ierr); }
  if (shell->yc_start)   { ierr = VecDestroy(&shell->yc_start);CHKERRQ(ierr); }
  if (shell->xc_middle)  { ierr = VecDestroy(&shell->xc_middle);CHKERRQ(ierr); }
  if (shell->yc_middle)  { ierr = VecDestroy(&shell->yc_middle);CHKERRQ(ierr); }
  if (shell->xc_end)     { ierr = VecDestroy(&shell->xc_end);CHKERRQ(ierr); }
  if (shell->yc_end)     { ierr = VecDestroy(&shell->yc_end);CHKERRQ(ierr); }

  if (shell->A_chiv_start)   { ierr = MatDestroy(&shell->A_chiv_start);CHKERRQ(ierr); }
  if (shell->A_chiv_middle)  { ierr = MatDestroy(&shell->A_chiv_middle);CHKERRQ(ierr); }
  if (shell->A_chiv_end)     { ierr = MatDestroy(&shell->A_chiv_end);CHKERRQ(ierr); }


  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode stiff2(Mat A, PetscInt nx, PetscInt nt, void *ctx, PetscBool impose_left_dirichlet, const PetscScalar *Uslab)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ctx;

  PetscFunctionBegin;

  /* Use the same element matrices as the DMDA Jacobian */
  PetscScalar (*A_time)[4]     = user->A_time;
  PetscScalar (*A_space)[4]    = user->A_space;
  PetscScalar (*A_std)[4] = user->A_standard;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);

  /* element loop over (tElm,xElm) */
  for (PetscInt tElm = 0; tElm < nt-1; ++tElm) {
    for (PetscInt xElm = 0; xElm < nx; ++xElm) {

      /* Four corners in ordering:
         0: (t,x)
         1: (t,x+1)
         2: (t+1,x+1)
         3: (t+1,x)
      */
      /* for spatial periodicity */
      PetscInt xp = (xElm + 1) % nx;

      PetscInt xg[4] = { xElm, xp, xp, xElm };
      PetscInt tg[4] = { tElm, tElm, tElm+1, tElm+1 };

      /* Build the 16 dof indices for this element */
      PetscInt idx[16];
      for (PetscInt a=0; a<4; ++a) {
        idx[4*a + 0] = gid(nx, tg[a], xg[a], 0); /* phi */
        idx[4*a + 1] = gid(nx, tg[a], xg[a], 1); /* u */
        idx[4*a + 2] = gid(nx, tg[a], xg[a], 2); /* chi */
        idx[4*a + 3] = gid(nx, tg[a], xg[a], 3); /* v */
      }

      /* Element matrix M (16x16), same as FormJacobian */
      PetscScalar M[16][16] = {{0}};


      for (PetscInt i=0; i<4; ++i) {
        PetscInt r_phi = 4*i;     /* row for eqn1 at corner i */
        PetscInt r_u = 4*i + 1; /* row for eqn2 at corner i */
        PetscInt r_chi = 4*i + 2; /* row for eqn3 at corner i */
        PetscInt r_v = 4*i + 3; /* row for eqn4 at corner i */
        for (PetscInt j=0; j<4; ++j) {
          PetscInt c_phi = 4*j;
          PetscInt c_u = 4*j + 1;
          PetscInt c_chi = 4*j + 2;
          PetscInt c_v = 4*j + 3;

          M[r_phi][c_phi] += A_space[i][j] + user->mphi2*A_std[i][j];
          M[r_phi][c_u] += A_time [i][j];
          M[r_u][c_phi] += A_time [i][j];
          M[r_u][c_u] += -A_std [i][j];

          M[r_chi][c_chi] += A_space[i][j] + user->mchi2*A_std[i][j];
          M[r_chi][c_v] += A_time [i][j];
          M[r_v][c_chi] += A_time [i][j];
          M[r_v][c_v] += -A_std [i][j];

        }
      }
      
      /* Nonlinear contributions, if Uslab provided.
         Linear contributions seem to be enough for the PC now,
         but maybe I will revisit this in the future. */
      if (Uslab) {
        /* Gather phi/chi at the 4 corners from slab-local Uslab */
        PetscScalar phi_l[4], chi_l[4];
        for (PetscInt a=0; a<4; ++a) {
          phi_l[a] = Uslab[ gid(nx, tg[a], xg[a], 0) ];
          chi_l[a] = Uslab[ gid(nx, tg[a], xg[a], 2) ];
        }

        /* Nonlinear Jacobian blocks */
        PetscScalar Jpp[4][4], Jpc[4][4], Jcc[4][4], Jcp[4][4];

        PetscReal hx = user->hx;
        PetscReal ht = user->ht;

        ComputeJ_local_nonlinear(Jpp, Jpc, Jcc, Jcp, phi_l, chi_l, hx, ht, ctx);

        /* Add into element matrix:
          - Eqn1 rows (phi-equation): r_phi = 4*i + 0
          - Eqn3 rows (chi-equation): r_chi = 4*i + 2
          - Cols for phi: c_phi = 4*j + 0
          - Cols for chi: c_chi = 4*j + 2
        */
        for (PetscInt i=0; i<4; ++i) {
          PetscInt r_phi = 4*i + 0;
          PetscInt r_chi = 4*i + 2;
          for (PetscInt j=0; j<4; ++j) {
            PetscInt c_phi = 4*j + 0;
            PetscInt c_chi = 4*j + 2;

            M[r_phi][c_phi] += Jpp[i][j];
            M[r_phi][c_chi] += Jpc[i][j];
            M[r_chi][c_chi] += Jcc[i][j];
            M[r_chi][c_phi] += Jcp[i][j];
          }
        }
      }


      ierr = MatSetValues(A, 16, idx, 16, idx, &M[0][0], ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Impose t=0 initial condition rows inside the slab:
     Rows/cols for all x at local t=0 for all components phi, u, chi, and v.
  */
  if (impose_left_dirichlet) {
  PetscInt nbc = 4*nx;
  PetscInt *rows = NULL;
  ierr = PetscMalloc1(nbc, &rows);CHKERRQ(ierr);
  for (PetscInt x=0; x<nx; ++x) {
    rows[4*x + 0] = gid(nx, 0, x, 0);
    rows[4*x + 1] = gid(nx, 0, x, 1);
    rows[4*x + 2] = gid(nx, 0, x, 2);
    rows[4*x + 3] = gid(nx, 0, x, 3);
  }

  ierr = MatZeroRowsColumns(A, nbc, rows, 1.0, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);
}

/* Build stiff mat for FS block */
PetscErrorCode stiff2_block(Mat A, PetscInt nx, PetscInt nt, void *ctx, PetscBool impose_left_dirichlet, const PetscScalar *Uslab4, PetscInt dof0)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ctx;

  PetscFunctionBegin;

  /* Use the same element matrices as the DMDA Jacobian */
  PetscScalar (*A_time)[4]     = user->A_time;
  PetscScalar (*A_space)[4]    = user->A_space;
  PetscScalar (*A_std)[4] = user->A_standard;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);

  /* element loop over (tElm,xElm) */
  for (PetscInt tElm = 0; tElm < nt-1; ++tElm) {
    for (PetscInt xElm = 0; xElm < nx; ++xElm) {

      /* Four corners in ordering:
         0: (t,x)
         1: (t,x+1)
         2: (t+1,x+1)
         3: (t+1,x)
      */
      /* for spatial periodicity */
      PetscInt xp = (xElm + 1) % nx;

      PetscInt xg[4] = { xElm, xp, xp, xElm };
      PetscInt tg[4] = { tElm, tElm, tElm+1, tElm+1 };

      /* Build the 8 dof indices for this element */
      PetscInt idx[8];
      if (dof0 == 0) {
        /* block for phi/u */
        for (PetscInt a=0; a<4; ++a) {
          idx[2*a + 0] = gid2(nx, tg[a], xg[a], 0); /* phi */
          idx[2*a + 1] = gid2(nx, tg[a], xg[a], 1); /* u */
        }
      } else {
        /* block for chi/v */
        for (PetscInt a=0; a<4; ++a) {
          idx[2*a + 0] = gid2(nx, tg[a], xg[a], 0); /* chi */
          idx[2*a + 1] = gid2(nx, tg[a], xg[a], 1); /* v */
        }
      }

      /* Element matrix M (16x16), same as FormJacobian */
      PetscScalar M[8][8] = {{0}};


      if (dof0 == 0) {
         /* block for phi/u */
        for (PetscInt i=0; i<4; ++i) {
          PetscInt r_phi = 2*i;     /* row for eqn1 at corner i */
          PetscInt r_u = 2*i + 1; /* row for eqn2 at corner i */
          for (PetscInt j=0; j<4; ++j) {
            PetscInt c_phi = 2*j;
            PetscInt c_u = 2*j + 1;

            M[r_phi][c_phi] += A_space[i][j] + user->mphi2*A_std[i][j];
            M[r_phi][c_u] += A_time [i][j];
            M[r_u][c_phi] += A_time [i][j];
            M[r_u][c_u] += -A_std [i][j];

          }
        }

      } else {
         /* block for chi/v */
        for (PetscInt i=0; i<4; ++i) {
          PetscInt r_chi = 2*i; /* row for eqn3 at corner i */
          PetscInt r_v = 2*i + 1; /* row for eqn4 at corner i */
          for (PetscInt j=0; j<4; ++j) {
            PetscInt c_chi = 2*j;
            PetscInt c_v = 2*j + 1;

            M[r_chi][c_chi] += A_space[i][j] + user->mchi2*A_std[i][j];
            M[r_chi][c_v] += A_time [i][j];
            M[r_v][c_chi] += A_time [i][j];
            M[r_v][c_v] += -A_std [i][j];

          }
        }
      }

      /* Nonlinear contributions, if Uslab4 provided.
         Linear contributions seem to be enough for the PC now,
         but maybe I will revisit this in the future. */
      if (Uslab4) {
        /* Gather phi/chi at the 4 corners from full 4-DOF slab */
        PetscScalar phi_l[4], chi_l[4];
        for (PetscInt a=0; a<4; ++a) {
          phi_l[a] = Uslab4[ gid(nx, tg[a], xg[a], 0) ];
          chi_l[a] = Uslab4[ gid(nx, tg[a], xg[a], 2) ];
        }

        /* Nonlinear Jacobian blocks */
        PetscScalar Jpp[4][4], Jpc[4][4], Jcc[4][4], Jcp[4][4];

        PetscReal hx = user->hx;
        PetscReal ht = user->ht;

        ComputeJ_local_nonlinear(Jpp, Jpc, Jcc, Jcp, phi_l, chi_l, hx, ht, ctx);

        /* Add into element matrix:
          - Eqn1 rows (phi-equation): r_phi = 4*i + 0
          - Eqn3 rows (chi-equation): r_chi = 4*i + 2
          - Cols for phi: c_phi = 4*j + 0
          - Cols for chi: c_chi = 4*j + 2
        */

        if (dof0 == 0) {
           /* block for phi/u */
          for (PetscInt i=0; i<4; ++i) {
            PetscInt r_phi = 2*i + 0;
            for (PetscInt j=0; j<4; ++j) {
              PetscInt c_phi = 2*j + 0;
              M[r_phi][c_phi] += Jpp[i][j];
            }
          }

        } else {
           /* block for chi/v */
           for (PetscInt i=0; i<4; ++i) {
            PetscInt r_chi = 2*i + 0;
            for (PetscInt j=0; j<4; ++j) {
              PetscInt c_chi = 2*j + 0;
              M[r_chi][c_chi] += Jcc[i][j];
            }
          }
        }
      }


      ierr = MatSetValues(A, 8, idx, 8, idx, &M[0][0], ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Impose t=0 initial condition rows inside the slab:
     Rows/cols for all x at local t=0 for all components phi, u, chi, and v.
  */
  if (impose_left_dirichlet) {
  PetscInt nbc = 2*nx;
  PetscInt *rows = NULL;
  ierr = PetscMalloc1(nbc, &rows);CHKERRQ(ierr);
  if (dof0 == 0) {
    /* block for phi/u */
    for (PetscInt x=0; x<nx; ++x) {
      rows[2*x + 0] = gid2(nx, 0, x, 0); /* phi */
      rows[2*x + 1] = gid2(nx, 0, x, 1); /* u */
    }
  } else {
    /* block for chi/v */
     for (PetscInt x=0; x<nx; ++x) {
      rows[2*x + 0] = gid2(nx, 0, x, 0); /* chi */
      rows[2*x + 1] = gid2(nx, 0, x, 1); /* v */
    }
  }

  ierr = MatZeroRowsColumns(A, nbc, rows, 1.0, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);
}
