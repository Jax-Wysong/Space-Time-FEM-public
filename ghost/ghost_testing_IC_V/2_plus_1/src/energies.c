#include "energies.h"
#include "potential.h"

static inline PetscScalar Dx(const PetscScalar (***a)[4], PetscInt t, PetscInt y, PetscInt x, PetscInt nx, PetscReal hx, PetscInt comp)
{
  PetscInt xm1 = x-1;
  PetscInt xp1 = x+1;
  (void)nx; /* nx not used because of periodic DM in x */
  return (a[t][y][xp1][comp] - a[t][y][xm1][comp]) / (2.0*hx);
}

static inline PetscScalar Dy(const PetscScalar (***a)[4], PetscInt t, PetscInt y, PetscInt x, PetscInt ny, PetscReal hy, PetscInt comp)
{
  PetscInt ym1 = y-1;
  PetscInt yp1 = y+1;
  (void)ny; /* ny not used because of periodic DM in y */
  return (a[t][yp1][x][comp] - a[t][ym1][x][comp]) / (2.0*hy);
}

void SliceEnergies(DM dm, Vec U, PetscInt t_idx, AppCtx *user, PetscReal *Hphi, PetscReal *Hchi)
{
  DMDALocalInfo info; DMDAGetLocalInfo(dm,&info);
  MPI_Comm      comm; PetscObjectGetComm((PetscObject)dm,&comm);

  Vec Uloc;
  DMGetLocalVector(dm,&Uloc);
  DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Uloc);
  DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Uloc);

  const PetscScalar (***u)[4];
  DMDAVecGetArrayRead(dm,Uloc,&u);

  PetscScalar lsum_phi = 0.0, lsum_chi = 0.0;

  if (t_idx >= info.zs && t_idx < info.zs + info.zm) {
      for (PetscInt y = info.ys; y < info.ys + info.ym; ++y) {
        for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {

          PetscScalar phi = u[t_idx][y][x][0],
                      ut  = u[t_idx][y][x][1],
                      chi = u[t_idx][y][x][2],
                      vt  = u[t_idx][y][x][3];

          PetscScalar phix = Dx(u,t_idx,y,x,user->nx,user->hx,0);
          PetscScalar chix = Dx(u,t_idx,y,x,user->nx,user->hx,2);

          PetscScalar phiy = Dy(u,t_idx,y,x,user->ny,user->hy,0);
          PetscScalar chiy = Dy(u,t_idx,y,x,user->ny,user->hy,2);

          /* find V_phi and V_chi at chi = 0.0 and phi = 0.0 respectively */

          PetscScalar V_1, V_phi_1, V_chi_1;
          PetscScalar V_phiphi_1, V_phichi_1, V_chiphi_1, V_chichi_1;

          compute_potential(phi, 0.0, user,
                &V_1,
                &V_phi_1, &V_chi_1,
                &V_phiphi_1, &V_phichi_1,
                &V_chiphi_1, &V_chichi_1);

          PetscScalar V_phi0 = V_1;

          PetscScalar V_2, V_phi_2, V_chi_2;
          PetscScalar V_phiphi_2, V_phichi_2, V_chiphi_2, V_chichi_2;

          compute_potential(0.0, chi, user,
                &V_2,
                &V_phi_2, &V_chi_2,
                &V_phiphi_2, &V_phichi_2,
                &V_chiphi_2, &V_chichi_2);

          PetscScalar V_chi0 = V_2;

          lsum_phi += 0.5*(ut*ut + (phix*phix + phiy*phiy)+ user->mphi2*phi*phi) + V_phi0;
          lsum_chi += (user->ghost)*0.5*(vt*vt + (chix*chix + chiy*chiy) + user->mchi2*chi*chi) + V_chi0;
        }
      }
  }

  DMDAVecRestoreArrayRead(dm,Uloc,&u);
  DMRestoreLocalVector(dm,&Uloc);

  PetscScalar gsum_phi, gsum_chi;
  MPI_Allreduce(&lsum_phi,&gsum_phi,1,MPIU_SCALAR,MPIU_SUM,comm);
  MPI_Allreduce(&lsum_chi,&gsum_chi,1,MPIU_SCALAR,MPIU_SUM,comm);

  *Hphi = user->hx * user->hy * gsum_phi;
  *Hchi = user->hx * user->hy * gsum_chi;
}