#include "energies.h"

static inline PetscScalar Dx(const PetscScalar (**a)[4], PetscInt t, PetscInt x, PetscInt nx, PetscReal hx, PetscInt comp)      /* 0 = φ, 2 = χ */
{
  PetscInt xm1 = x-1;
  PetscInt xp1 = x+1;
  return (a[t][xp1][comp] - a[t][xm1][comp]) / (2.0*hx);
}


/* integrate H_phi and H_chi at time‑index t_idx */
PetscErrorCode SliceEnergies(DM dm, Vec U, PetscInt t_idx, AppCtx *user, PetscReal *Hphi, PetscReal *Hchi)
{
  DMDALocalInfo info; DMDAGetLocalInfo(dm,&info);
  MPI_Comm      comm; PetscObjectGetComm((PetscObject)dm,&comm);

  /* get ghosted local vector ------------------------------------ */
  Vec Uloc;
  DMGetLocalVector(dm,&Uloc);
  DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Uloc);
  DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Uloc);

  /* read‑only array access -------------------------------------- */
  const PetscScalar (**u)[4];
  DMDAVecGetArrayRead(dm,Uloc,&u);

  /* loop only if this rank actually owns the time‑slice ---------- */
  PetscScalar lsum_phi = 0.0, lsum_chi = 0.0;

  if (t_idx >= info.ys && t_idx < info.ys + info.ym) {
    for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {

      PetscScalar phi = u[t_idx][x][0],
                  ut  = u[t_idx][x][1],
                  chi = u[t_idx][x][2],
                  vt  = u[t_idx][x][3];

      PetscScalar phix = Dx(u,t_idx,x,user->nx,user->hx,0);
      PetscScalar chix = Dx(u,t_idx,x,user->nx,user->hx,2);



      lsum_phi += 0.5*(ut*ut + phix*phix + user->mphi2*phi*phi); /* V=0 for now with V = phi^2 chi^2*/
      lsum_chi += user->ghost*0.5*(vt*vt + chix*chix + user->mchi2*chi*chi); /* ditto      */
    }
  }

  DMDAVecRestoreArrayRead(dm,Uloc,&u);
  DMRestoreLocalVector(dm,&Uloc);

  /* global reduction -------------------------------------------- */
  PetscScalar gsum_phi, gsum_chi;
  MPI_Allreduce(&lsum_phi,&gsum_phi,1,MPIU_SCALAR,MPIU_SUM,comm);
  MPI_Allreduce(&lsum_chi,&gsum_chi,1,MPIU_SCALAR,MPIU_SUM,comm);

  *Hphi = user->hx * gsum_phi; // trapezoidal rule with first and last equal
  *Hchi = user->hx * gsum_chi;
}