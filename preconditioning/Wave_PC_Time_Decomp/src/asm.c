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
  return 2*(t*nx + x) + c; /* interleaved dofs: u=0, v=1 */
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
      for (PetscInt c = 0; c < 2; ++c) {
        xs[ gid(nx, tt, x, c) ] = xa[t][x][c];
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
      for (PetscInt c = 0; c < 2; ++c) {
        ya[t][x][c] += yslab[ gid(nx, tt, x, c) ];
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

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nloc, nloc, 20, NULL, A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
  ierr = MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);

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
      for (PetscInt c = 0; c < 2; ++c) {
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
  shell->interface_width = user->interface_width;

  /* blksize */
  shell->blksize = user->nt / user->Nsub;


  MPI_Comm_rank(PETSC_COMM_WORLD, &shell->rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &shell->size);

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
  const PetscInt iw      = shell->interface_width;


  /* Decide slab type for this rank */
  if (shell->rank == 0) shell->slabtype = SLAB_START;
  else if (shell->rank == Nsub-1) shell->slabtype = SLAB_END;
  else shell->slabtype = SLAB_MIDDLE;

  /* --------------------------------------------
     First-time build: allocate only this rank’s slab objects
     -------------------------------------------- */

  if (shell->slabtype == SLAB_START) {
    /* clip slab window to nt-1 (relevant when Nsub==1: no right neighbour to overlap) */
    PetscInt t1_start = PetscMin(blksize + ov - 1, nt - 1);
    PetscInt nloc = 2*nx*(t1_start + 1);
    /* create local Mat, KSP, Vecs for this slab */
    ierr = CreateLocalSlabObjects(shell, nloc, "sub_start_", &shell->A_start, &shell->ksp_start,
                                &shell->x_start, &shell->y_start);CHKERRQ(ierr);

    /* build A using the same logic from FormJacobian */
    ierr = stiff2(shell->A_start, nx, t1_start + 1, user, 1, NULL, pc);CHKERRQ(ierr);
    ierr = KSPSetUp(shell->ksp_start);CHKERRQ(ierr);

  } else if (shell->slabtype == SLAB_MIDDLE) {
    PetscInt nloc = 2*nx*(blksize + 2*ov + iw);
    /* creat local Mat, KSP, Vecs for this slab */
    ierr = CreateLocalSlabObjects(shell, nloc, "sub_mid_", &shell->A_middle, &shell->ksp_middle,
                                &shell->x_middle, &shell->y_middle);CHKERRQ(ierr);

    /* build A using the same logic from FormJacobian */
    PetscInt n_bc_mid = user->first_sub_only_bc ? 0 : (user->full_overlap_dirichlet ? ov : 1);
    ierr = stiff2(shell->A_middle, nx, blksize + 2*ov + iw, user, n_bc_mid, NULL, pc);CHKERRQ(ierr);
    ierr = KSPSetUp(shell->ksp_middle);CHKERRQ(ierr);

  } else { /* SLAB_END */
    PetscInt nloc = 2*nx*(blksize + ov + iw);
    /* create local Mat, KSP, Vecs for this slab */
    ierr = CreateLocalSlabObjects(shell, nloc, "sub_end_", &shell->A_end, &shell->ksp_end,
                                &shell->x_end, &shell->y_end);CHKERRQ(ierr);

    /* build A using the same logic from FormJacobian */
    PetscInt n_bc_end = user->first_sub_only_bc ? 0 : (user->full_overlap_dirichlet ? ov : 1);
    ierr = stiff2(shell->A_end, nx, blksize + ov + iw, user, n_bc_end, NULL, pc);CHKERRQ(ierr);
    ierr = KSPSetUp(shell->ksp_end);CHKERRQ(ierr);
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
  const PetscInt iw      = shell->interface_width;

  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  /* fill ghosted local copy of x once */
  ierr = DMGlobalToLocalBegin(shell->dm, x, INSERT_VALUES, shell->xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (shell->dm, x, INSERT_VALUES, shell->xloc);CHKERRQ(ierr);


  if (rank == 0) {
    PetscInt t0 = 0;
    /* clip to nt-1: no right neighbour when Nsub==1 */
    PetscInt t1     = PetscMin(blksize + ov - 1, shell->nt - 1);
    PetscInt t_add1 = PetscMin(blksize - 1,      shell->nt - 1);
    ierr = PackSlabFromDMDALocal(shell->dm, shell->xloc, nx, t0, t1, shell->x_start);CHKERRQ(ierr);

    ierr = VecZeroEntries(shell->y_start);CHKERRQ(ierr);
    ierr = KSPSolve(shell->ksp_start, shell->x_start, shell->y_start);CHKERRQ(ierr);

    if (shell->use_ras) {
      ierr = AddBackInteriorToGlobal_DMDARows(shell->dm, y, nx, t0, 0, t_add1, shell->y_start);CHKERRQ(ierr);
    } else {
      ierr = AddBackFullWindow_ASM(shell, y, nx, t0, t1, shell->y_start);CHKERRQ(ierr);
    }
  }

  if (rank > 0 && rank < Nsub-1) {
    PetscInt s  = rank;
    PetscInt t0 = s*blksize - iw - ov;
    PetscInt t1 = (s+1)*blksize + ov - 1;

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

  if (rank == Nsub-1 && Nsub > 1) {
    PetscInt t0 = (Nsub-1)*blksize - iw - ov;
    PetscInt t1 = shell->nt - 1;

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

  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode stiff2(Mat A, PetscInt nx, PetscInt nt, void *ctx, PetscInt n_bc_levels, const PetscScalar *Uslab, PC pc)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ctx;
  SampleShellPC *shell = NULL;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&shell);CHKERRQ(ierr);

  /* Use the same element matrices as the DMDA Jacobian */
  PetscScalar (*A_time)[4]     = user->A_time;
  PetscScalar (*A_space)[4]    = user->A_space;
  PetscScalar (*A_standard)[4] = user->A_standard;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);

  /* element loop over (tElm,xElm) for quads */
  for (PetscInt tElm = 0; tElm < nt-1; ++tElm) {
    for (PetscInt xElm = 0; xElm < nx-1; ++xElm) {

      /* Four corners in ordering:
         0: (t,x)
         1: (t,x+1)
         2: (t+1,x+1)
         3: (t+1,x)
      */
      PetscInt xg[4] = { xElm, xElm+1, xElm+1, xElm };
      PetscInt tg[4] = { tElm, tElm,   tElm+1, tElm+1 };

      /* Build the 8 dof indices for this element */
      PetscInt idx[8];
      for (PetscInt a=0; a<4; ++a) {
        idx[2*a + 0] = gid(nx, tg[a], xg[a], 0); /* u */
        idx[2*a + 1] = gid(nx, tg[a], xg[a], 1); /* v */
      }

      /* Element matrix M (8x8), same as FormJacobian */
      PetscScalar M[8][8] = {{0}};


      for (PetscInt i=0; i<4; ++i) {
        PetscInt r_u = 2*i;     /* row for eqn1 at corner i */
        PetscInt r_v = 2*i + 1; /* row for eqn2 at corner i */
        for (PetscInt j=0; j<4; ++j) {
          PetscInt c_u = 2*j;
          PetscInt c_v = 2*j + 1;

          /* eqn1:  A_space*u + A_time*v */
          M[r_u][c_u] += A_space[i][j];
          M[r_u][c_v] += A_time[i][j];

          /* eqn2: -A_time*u + A_standard*v */
          M[r_v][c_u] += -A_time[i][j];
          M[r_v][c_v] += A_standard[i][j];
        }
      }

      ierr = MatSetValues(A, 8, idx, 8, idx, &M[0][0], ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Impose left Dirichlet: zero rows for local t = 0 .. n_bc_levels-1 */
  if (n_bc_levels > 0) {
    PetscInt nbc = n_bc_levels * 2 * nx;
    PetscInt *rows = NULL;
    ierr = PetscMalloc1(nbc, &rows);CHKERRQ(ierr);
    for (PetscInt tlev = 0; tlev < n_bc_levels; ++tlev) {
      for (PetscInt x = 0; x < nx; ++x) {
        rows[tlev*2*nx + 2*x + 0] = gid(nx, tlev, x, 0);
        rows[tlev*2*nx + 2*x + 1] = gid(nx, tlev, x, 1);
      }
    }
    ierr = MatZeroRows(A, nbc, rows, 1.0, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFree(rows);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

