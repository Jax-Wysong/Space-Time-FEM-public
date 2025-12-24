#include <stdlib.h>
#include <petscsnes.h>
#include <petscviewer.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;
  Vec            U;
  Mat            J;
  AppCtx         user;

  PetscInt  nx = 10, ny = 10, nt = 10;
  PetscReal xL = -1.0, xR = 1.0;
  PetscReal yL = -1.0, yR = 1.0;
  PetscReal t0 =  0.0, tF = 2.0;
  PetscReal lam = 1.0;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);
  PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL);
  PetscOptionsGetInt(NULL, NULL, "-nt", &nt, NULL);
  PetscOptionsGetReal(NULL, NULL, "-xL", &xL, NULL);
  PetscOptionsGetReal(NULL, NULL, "-xR", &xR, NULL);
  PetscOptionsGetReal(NULL, NULL, "-yL", &yL, NULL);
  PetscOptionsGetReal(NULL, NULL, "-yR", &yR, NULL);
  PetscOptionsGetReal(NULL, NULL, "-t0", &t0, NULL);
  PetscOptionsGetReal(NULL, NULL, "-tF", &tF, NULL);
  PetscOptionsGetReal(NULL, NULL, "-lam", &lam, NULL);

  user.nx  = nx;
  user.ny  = ny;
  user.nt  = nt;
  user.xL  = xL;
  user.xR  = xR;
  user.yL  = yL;
  user.yR  = yR;
  user.t0  = t0;
  user.tF  = tF;
  user.lam = lam;

  user.hx = (user.xR-user.xL)/user.nx;
  user.hy = (user.yR-user.yL)/user.ny;
  user.ht = (user.tF-user.t0)/(user.nt-1);

  Compute_linear_stiffness(
      user.A_time,
      user.A_space_x,
      user.A_space_y,
      user.A_mass,
      user.hx,
      user.hy,
      user.ht);

  ierr = DMDACreate3d(
      PETSC_COMM_WORLD,
      DM_BOUNDARY_PERIODIC,
      DM_BOUNDARY_PERIODIC,
      DM_BOUNDARY_NONE,
      DMDA_STENCIL_BOX,
      user.nx, user.ny, user.nt,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
      4,
      1,
      NULL, NULL, NULL,
      &dm); CHKERRQ(ierr);

  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMSetUp(dm);           CHKERRQ(ierr);
  user.dm = dm;

  ierr = DMCreateGlobalVector(dm, &U); CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);       CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);                 CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, NULL, FormResidual, &user); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, J, J, FormJacobian, &user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  { /* Seeding intial solution guess with Initial Conditions specified by user */
    DMDALocalInfo info;
    DMDAGetLocalInfo(dm,&info);

    PetscInt xs = info.xs, xm = info.xm;
    PetscInt ys = info.ys, ym = info.ym;
    PetscInt zs = info.zs, zm = info.zm;

    PetscScalar (***u)[4];
    DMDAVecGetArray(dm,U,&u);

    for (PetscInt t = zs; t < zs+zm; ++t){
      for (PetscInt y = ys; y < ys+ym; ++y){
        PetscReal y_phys = user.yL + y*user.hy;
        for (PetscInt x = xs; x < xs+xm; ++x){
          PetscReal x_phys = user.xL + x*user.hx;

          if (t == 0){
            u[t][y][x][0] = 1.0 + PetscCosReal(PETSC_PI * x_phys)
                                  * PetscCosReal(PETSC_PI * y_phys);
            u[t][y][x][1] = 0.0;
            u[t][y][x][2] = 1.0;
            u[t][y][x][3] = PETSC_PI * PetscCosReal(PETSC_PI * x_phys)
                                      * PetscCosReal(PETSC_PI * y_phys);
          } else {
            u[t][y][x][0] = 0.0;
            u[t][y][x][1] = 0.0;
            u[t][y][x][2] = 0.0;
            u[t][y][x][3] = 0.0;
          }
        }
      }
    }

    DMDAVecRestoreArray(dm,U,&u);
  }


  ierr = SNESSolve(snes, NULL, U); CHKERRQ(ierr);

  /* Parallel-safe L2 and max error norms for phi and chi */
  {
    DMDALocalInfo info;
    DMDAGetLocalInfo(dm,&info);

    PetscReal l2_phi_local = 0.0, l2_chi_local = 0.0;
    PetscReal max_phi_local = 0.0, max_chi_local = 0.0;

    PetscScalar (***a)[4];
    DMDAVecGetArrayRead(dm, U, &a);

    for (PetscInt tj = info.zs; tj < info.zs + info.zm; ++tj) {
      PetscReal t_phys = user.t0 + tj * user.ht;
      for (PetscInt yj = info.ys; yj < info.ys + info.ym; ++yj) {
        PetscReal y_phys = user.yL + yj * user.hy;
        for (PetscInt xi = info.xs; xi < info.xs + info.xm; ++xi) {
          PetscReal x_phys = user.xL + xi * user.hx;

          PetscReal u_exact_phi =
            1.0 + PetscCosReal(PETSC_PI*x_phys)
                * PetscCosReal(PETSC_PI*y_phys)
                * PetscCosReal(PETSC_PI*t_phys);

          PetscReal u_exact_chi =
            1.0 + PetscCosReal(PETSC_PI*x_phys)
                * PetscCosReal(PETSC_PI*y_phys)
                * PetscSinReal(PETSC_PI*t_phys);

          PetscReal diff_phi = PetscRealPart(a[tj][yj][xi][0]) - u_exact_phi;
          PetscReal diff_chi = PetscRealPart(a[tj][yj][xi][2]) - u_exact_chi;

          l2_phi_local += diff_phi * diff_phi;
          l2_chi_local += diff_chi * diff_chi;

          PetscReal ap = PetscAbsReal(diff_phi);
          PetscReal ac = PetscAbsReal(diff_chi);
          if (ap > max_phi_local) max_phi_local = ap;
          if (ac > max_chi_local) max_chi_local = ac;
        }
      }
    }

    DMDAVecRestoreArrayRead(dm, U, &a);

    PetscReal l2_phi_global = 0.0, l2_chi_global = 0.0;
    PetscReal max_phi_global = 0.0, max_chi_global = 0.0;

    MPIU_Allreduce(&l2_phi_local, &l2_phi_global, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    MPIU_Allreduce(&l2_chi_local, &l2_chi_global, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    MPIU_Allreduce(&max_phi_local, &max_phi_global, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);
    MPIU_Allreduce(&max_chi_local, &max_chi_global, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);

    PetscReal error_L2_phi = PetscSqrtReal(l2_phi_global * user.hx * user.hy * user.ht);
    PetscReal error_L2_chi = PetscSqrtReal(l2_chi_global * user.hx * user.hy * user.ht);

    PetscPrintf(PETSC_COMM_WORLD,
      "Using nx = %d, Using ny = %d, nt = %d\nlambda = %g\n",
      nx, ny, nt, lam);

    PetscPrintf(PETSC_COMM_WORLD,
      "xL = %g, xR = %g\nyL = %g, yR = %g\nt0 = %g, tF = %g\n",
      xL, xR, yL, yR, t0, tF);

    PetscPrintf(PETSC_COMM_WORLD,
      "hx = %.8e, hy = %.8e, ht = %.8e\n\n",
      (double)user.hx, (double)user.hy, (double)user.ht);

    PetscPrintf(PETSC_COMM_WORLD,
      "PHI L2 error norm = %.8e\n", (double)error_L2_phi);
    PetscPrintf(PETSC_COMM_WORLD,
      "PHI max error norm = %.8e\n", (double)max_phi_global);
    PetscPrintf(PETSC_COMM_WORLD,
      "CHI L2 error norm = %.8e\n", (double)error_L2_chi);
    PetscPrintf(PETSC_COMM_WORLD,
      "CHI max error norm = %.8e\n", (double)max_chi_global);
  }

  ierr = VecDestroy(&U);    CHKERRQ(ierr);
  ierr = MatDestroy(&J);    CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  DMDestroy(&dm);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}
