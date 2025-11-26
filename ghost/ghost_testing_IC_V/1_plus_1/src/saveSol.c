#include <stdlib.h>
#include "saveSol.h"
#include "appctx.h"
#include "energies.h"
#include "stiffness.h"
#include <petscviewer.h>



/* Build output filename for PETSc binary based on IC and parameters */
PetscErrorCode BuildOutputFilename(const AppCtx *user, char fname[], size_t len)
{
  PetscFunctionBeginUser;

  if (user->IC == 1 || user->IC == 2) {
    PetscCall(PetscSNPrintf(fname, len,
                            "lam22_IC_%d_A%.5g_C%.5g_1x%.5g_%dx%d.dat",
                            (int)user->IC,
                            (double)user->A,
                            (double)user->C,
                            (double)user->tF,
                            (int)user->nx,
                            (int)user->nt));
  } else if (user->IC == 4) {
    PetscCall(PetscSNPrintf(fname, len,
                            "lam22_IC_%d_A%.5g_sigma_%d_r%.5g_dphi_%.5g_1x%.5g_%dx%d.dat",
                            (int)user->IC,
                            (double)user->A,
                            (int)user->pw_sigma,
                            (double)user->pw_r,
                            (double)user->pw_dphi,
                            (double)user->tF,
                            (int)user->nx,
                            (int)user->nt));
  } else if (user->IC == 5) {
    PetscCall(PetscSNPrintf(fname, len,
                            "lam22_IC_%d_A%.5g_sigma_%.5g_Carrier_%.5g_1x%.5g_%dx%d.dat",
                            (int)user->IC,
                            (double)user->A,
                            (double)user->osc_sigma,
                            (double)user->osc_k0,
                            (double)user->tF,
                            (int)user->nx,
                            (int)user->nt));
  } else {
    /* Generic fallback for other ICs (3, 6, etc.) */
    PetscCall(PetscSNPrintf(fname, len,
                            "lam22_IC_%d_1x%.5g_%dx%d.dat",
                            (int)user->IC,
                            (double)user->tF,
                            (int)user->nx,
                            (int)user->nt));
  }

  PetscFunctionReturn(0);
}


/* Dump last-good solution: energies, phi/chi fields, x/t coords to PETSc binary */
PetscErrorCode DumpSolutionAndEnergies(AppCtx *user, DM dm_good, Vec Ugood, PetscInt last_good_nt, PetscReal last_good_tF)
{
  PetscFunctionBeginUser;

  /* Nothing to do if we don't have a valid solution */
  if (!Ugood || !dm_good || last_good_nt < 0) PetscFunctionReturn(0);

  DM   mydm = dm_good;
  Vec  HphiVec = NULL, HchiVec = NULL;
  Vec  Unat = NULL, phi = NULL, chi = NULL;
  Vec  vx = NULL, vt = NULL;
  PetscViewer viewer = NULL;
  char fname[PETSC_MAX_PATH_LEN];

  /* --- Recompute user fields to match last-good grid --- */
  user->nt = last_good_nt;
  user->tF = last_good_tF;
  user->hx = (user->xR - user->xL) / user->nx;
  user->ht = (user->tF - user->t0) / ((PetscReal)user->nt - 1.0);
  user->L  = user->xR - user->xL;
  user->dm = mydm;   /* ensure callbacks see correct DM */

  Compute_linear_stiffness(user->A_time,
                           user->A_space,
                           user->A_standard,
                           user->hx,
                           user->ht);

  /* --- Build energy histories into Vecs of length nt --- */
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user->nt, &HphiVec));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user->nt, &HchiVec));
  PetscCall(PetscObjectSetName((PetscObject)HphiVec, "Hphi"));
  PetscCall(PetscObjectSetName((PetscObject)HchiVec, "Hchi"));

  PetscInt rstart, rend;
  PetscCall(VecGetOwnershipRange(HphiVec, &rstart, &rend));

  for (PetscInt t = 0; t < user->nt; ++t) {
    PetscReal Hp, Hc;
    PetscCall(SliceEnergies(user->dm, Ugood, t, user, &Hp, &Hc));
    if (t >= rstart && t < rend) {
      PetscScalar shp = (PetscScalar)Hp, shc = (PetscScalar)Hc;
      PetscCall(VecSetValues(HphiVec, 1, &t, &shp, INSERT_VALUES));
      PetscCall(VecSetValues(HchiVec, 1, &t, &shc, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(HphiVec)); PetscCall(VecAssemblyEnd(HphiVec));
  PetscCall(VecAssemblyBegin(HchiVec)); PetscCall(VecAssemblyEnd(HchiVec));

  /* --- Ugood -> natural ordering (still distributed). x-fastest, then t --- */
  PetscCall(DMDACreateNaturalVector(user->dm, &Unat));
  PetscCall(DMDAGlobalToNaturalBegin(user->dm, Ugood, INSERT_VALUES, Unat));
  PetscCall(DMDAGlobalToNaturalEnd  (user->dm, Ugood, INSERT_VALUES, Unat));
  PetscCall(VecSetBlockSize(Unat, 4)); /* [phi,ut,chi,vt] */

  /* --- Extract components 0 (phi) and 2 (chi) --- */
  PetscInt Nloc, Nglob;
  PetscCall(VecGetLocalSize(Unat, &Nloc));
  PetscCall(VecGetSize(Unat, &Nglob));
  PetscInt nloc  = Nloc  / 4;
  PetscInt nglob = Nglob / 4;

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, nloc, nglob, &phi));
  PetscCall(VecDuplicate(phi, &chi));
  PetscCall(PetscObjectSetName((PetscObject)phi, "phi"));
  PetscCall(PetscObjectSetName((PetscObject)chi, "chi"));

  const PetscScalar *ua;
  PetscCall(VecGetArrayRead(Unat, &ua));

  PetscInt urstart;
  PetscCall(VecGetOwnershipRange(Unat, &urstart, NULL));
  PetscInt bstart = urstart / 4;   /* global block index of local start */

  for (PetscInt i = 0; i < nloc; ++i) {
    PetscInt    g    = bstart + i;         /* global index in phi/chi (0..nx*nt-1) */
    PetscScalar vphi = ua[4*i + 0];
    PetscScalar vchi = ua[4*i + 2];
    PetscCall(VecSetValues(phi, 1, &g, &vphi, INSERT_VALUES));
    PetscCall(VecSetValues(chi, 1, &g, &vchi, INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(Unat, &ua));
  PetscCall(VecAssemblyBegin(phi)); PetscCall(VecAssemblyEnd(phi));
  PetscCall(VecAssemblyBegin(chi)); PetscCall(VecAssemblyEnd(chi));

  /* --- 4) Coordinates x (nx) and t (nt) as Vecs --- */
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user->nx, &vx));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user->nt, &vt));
  PetscCall(PetscObjectSetName((PetscObject)vx, "x"));
  PetscCall(PetscObjectSetName((PetscObject)vt, "t"));

  PetscInt   rs, re;
  PetscScalar val;

  PetscCall(VecGetOwnershipRange(vx, &rs, &re));
  for (PetscInt i = rs; i < re; ++i) {
    val = (PetscScalar)(user->xL + i * user->hx);
    PetscCall(VecSetValues(vx, 1, &i, &val, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(vx)); PetscCall(VecAssemblyEnd(vx));

  PetscCall(VecGetOwnershipRange(vt, &rs, &re));
  for (PetscInt i = rs; i < re; ++i) {
    val = (PetscScalar)(user->t0 + i * user->ht);
    PetscCall(VecSetValues(vt, 1, &i, &val, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(vt)); PetscCall(VecAssemblyEnd(vt));

  /* --- Build filename and open viewer --- */
  PetscCall(BuildOutputFilename(user, fname, sizeof(fname)));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_WRITE, &viewer));

  /* --- Write  what you need --- */
  PetscCall(VecView(phi,     viewer));
  PetscCall(VecView(chi,     viewer));
  PetscCall(VecView(HphiVec, viewer));
  PetscCall(VecView(HchiVec, viewer));
  PetscCall(VecView(vx,      viewer));
  PetscCall(VecView(vt,      viewer));

  /* --- Cleanup --- */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&vx));
  PetscCall(VecDestroy(&vt));
  PetscCall(VecDestroy(&HphiVec));
  PetscCall(VecDestroy(&HchiVec));
  PetscCall(VecDestroy(&phi));
  PetscCall(VecDestroy(&chi));
  PetscCall(VecDestroy(&Unat));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Saved (1+1) phi, chi, Hphi, Hchi, x, t to PETSc binary: %s\n",
                        fname));

  PetscFunctionReturn(0);
}

/* Save (copy) the last convergent solution into Ugood, (re)allocating if size/layout changed */
PetscErrorCode SaveUgood(Vec src, Vec *Ugood, PetscInt *Ugood_n)
{
  PetscInt nsrc;
  PetscFunctionBeginUser;
  PetscCall(VecGetSize(src,&nsrc));
  if (!*Ugood || *Ugood_n != nsrc) {
    PetscCall(VecDestroy(Ugood));
    PetscCall(VecDuplicate(src,Ugood));
    *Ugood_n = nsrc;
  }
  PetscCall(VecCopy(src,*Ugood));
}
