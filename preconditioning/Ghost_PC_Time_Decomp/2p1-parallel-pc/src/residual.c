#include <stdlib.h>
#include "appctx.h"
#include "residual.h"
#include "stiffness.h"
#include "nonlin.h"

PetscErrorCode FormResidual(SNES snes, Vec U, Vec R, void *ctx) {
  AppCtx       *user = (AppCtx*)ctx;
  DM                 dm   = user->dm;
  DMDALocalInfo      info;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscFunctionBegin;


  PetscScalar (*A_time)[8]    = user->A_time;
  PetscScalar (*A_space_x)[8] = user->A_space_x;
  PetscScalar (*A_space_y)[8] = user->A_space_y;
  PetscScalar (*A_mass)[8]    = user->A_mass;


  DMDAGetLocalInfo(dm,&info);
  xs = info.xs;  xm = info.xm;
  ys = info.ys;  ym = info.ym;
  zs = info.zs;  zm = info.zm;

  VecZeroEntries(R);

  Vec Ul, Rl;
  DMGetLocalVector   (dm,&Ul);
  DMGetLocalVector   (dm,&Rl);

  PetscScalar *a; PetscInt nloc;
  VecGetLocalSize(Rl,&nloc);
  VecGetArrayWrite(Rl,&a);  PetscArrayzero(a,nloc);  VecRestoreArrayWrite(Rl,&a);

  

  DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul);
  DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul);

  PetscScalar (***u)[4], (***r)[4];
  DMDAVecGetArrayRead(dm,Ul,&u);
  DMDAVecGetArray    (dm,Rl,&r);

  PetscInt tEnd = PetscMin(zs + zm - 1, user->nt - 2);
  for (PetscInt tElm = zs; tElm <= tEnd; ++tElm) {
    for (PetscInt yElm = ys; yElm < ys+ym;   ++yElm) {
      for (PetscInt xElm = xs; xElm < xs+xm;   ++xElm) {
        
        PetscScalar phi_local[8], u_local[8], chi_local[8], v_local[8];

        PetscInt xloc1 = (xElm + 1);
        PetscInt yloc1 = (yElm + 1);

        phi_local[0] = u[tElm  ][yElm ][xElm ][0];
        phi_local[1] = u[tElm  ][yElm ][xloc1][0];
        phi_local[2] = u[tElm  ][yloc1][xloc1][0];
        phi_local[3] = u[tElm  ][yloc1][xElm ][0];
        phi_local[4] = u[tElm+1][yElm ][xElm ][0];
        phi_local[5] = u[tElm+1][yElm ][xloc1][0];
        phi_local[6] = u[tElm+1][yloc1][xloc1][0];
        phi_local[7] = u[tElm+1][yloc1][xElm ][0];

        u_local[0] = u[tElm  ][yElm ][xElm ][1];
        u_local[1] = u[tElm  ][yElm ][xloc1][1];
        u_local[2] = u[tElm  ][yloc1][xloc1][1];
        u_local[3] = u[tElm  ][yloc1][xElm ][1];
        u_local[4] = u[tElm+1][yElm ][xElm ][1];
        u_local[5] = u[tElm+1][yElm ][xloc1][1];
        u_local[6] = u[tElm+1][yloc1][xloc1][1];
        u_local[7] = u[tElm+1][yloc1][xElm ][1];

        chi_local[0] = u[tElm  ][yElm ][xElm ][2];
        chi_local[1] = u[tElm  ][yElm ][xloc1][2];
        chi_local[2] = u[tElm  ][yloc1][xloc1][2];
        chi_local[3] = u[tElm  ][yloc1][xElm ][2];
        chi_local[4] = u[tElm+1][yElm ][xElm ][2];
        chi_local[5] = u[tElm+1][yElm ][xloc1][2];
        chi_local[6] = u[tElm+1][yloc1][xloc1][2];
        chi_local[7] = u[tElm+1][yloc1][xElm ][2];

        v_local[0] = u[tElm  ][yElm ][xElm ][3];
        v_local[1] = u[tElm  ][yElm ][xloc1][3];
        v_local[2] = u[tElm  ][yloc1][xloc1][3];
        v_local[3] = u[tElm  ][yloc1][xElm ][3];
        v_local[4] = u[tElm+1][yElm ][xElm ][3];
        v_local[5] = u[tElm+1][yElm ][xloc1][3];
        v_local[6] = u[tElm+1][yloc1][xloc1][3];
        v_local[7] = u[tElm+1][yloc1][xElm ][3];

        PetscScalar eqn1_r_local[8]={0}, eqn2_r_local[8]={0};
        PetscScalar eqn3_r_local[8]={0}, eqn4_r_local[8]={0};

        for (PetscInt i=0;i<8;++i){
          for (PetscInt j=0;j<8;++j){
            eqn1_r_local[i] += A_time[i][j] * u_local[j];
            eqn1_r_local[i] += (A_space_x[i][j] + A_space_y[i][j]) * phi_local[j];
            eqn1_r_local[i] += A_mass[i][j] * phi_local[j];

            eqn2_r_local[i] += A_time[i][j] * phi_local[j];
            eqn2_r_local[i] += -A_mass[i][j] * u_local[j];

            eqn3_r_local[i] += A_time[i][j] * v_local[j];
            eqn3_r_local[i] += (A_space_x[i][j] + A_space_y[i][j]) * chi_local[j];
            eqn3_r_local[i] += A_mass[i][j] * chi_local[j];

            eqn4_r_local[i] += A_time[i][j] * chi_local[j];
            eqn4_r_local[i] += -A_mass[i][j] * v_local[j];
          }
        }


        PetscReal x0 = user->xL + xElm*user->hx;
        PetscReal y0 = user->yL + yElm*user->hy;
        PetscReal t0 = tElm*user->ht;
        PetscScalar eqn1_r_nonlin[8], eqn3_r_nonlin[8];
        ComputeR_local_nonlinear_phi(eqn1_r_nonlin, phi_local, chi_local, user->hx, user->hy, user->ht, x0, y0, t0, ctx);
        ComputeR_local_nonlinear_chi(eqn3_r_nonlin, chi_local, phi_local, user->hx, user->hy, user->ht, x0, y0, t0, ctx);



        for (PetscInt i=0;i<8;++i){
          eqn1_r_local[i]+=eqn1_r_nonlin[i];
          eqn3_r_local[i]+=eqn3_r_nonlin[i];
        }



        r[tElm  ][yElm ][xElm ][0] += eqn1_r_local[0];
        r[tElm  ][yElm ][xloc1][0] += eqn1_r_local[1];
        r[tElm  ][yloc1][xloc1][0] += eqn1_r_local[2];
        r[tElm  ][yloc1][xElm ][0] += eqn1_r_local[3];
        r[tElm+1][yElm ][xElm ][0] += eqn1_r_local[4];
        r[tElm+1][yElm ][xloc1][0] += eqn1_r_local[5];
        r[tElm+1][yloc1][xloc1][0] += eqn1_r_local[6];
        r[tElm+1][yloc1][xElm ][0] += eqn1_r_local[7];

        r[tElm  ][yElm ][xElm ][1] += eqn2_r_local[0];
        r[tElm  ][yElm ][xloc1][1] += eqn2_r_local[1];
        r[tElm  ][yloc1][xloc1][1] += eqn2_r_local[2];
        r[tElm  ][yloc1][xElm ][1] += eqn2_r_local[3];
        r[tElm+1][yElm ][xElm ][1] += eqn2_r_local[4];
        r[tElm+1][yElm ][xloc1][1] += eqn2_r_local[5];
        r[tElm+1][yloc1][xloc1][1] += eqn2_r_local[6];
        r[tElm+1][yloc1][xElm ][1] += eqn2_r_local[7];

        r[tElm  ][yElm ][xElm ][2] += eqn3_r_local[0];
        r[tElm  ][yElm ][xloc1][2] += eqn3_r_local[1];
        r[tElm  ][yloc1][xloc1][2] += eqn3_r_local[2];
        r[tElm  ][yloc1][xElm ][2] += eqn3_r_local[3];
        r[tElm+1][yElm ][xElm ][2] += eqn3_r_local[4];
        r[tElm+1][yElm ][xloc1][2] += eqn3_r_local[5];
        r[tElm+1][yloc1][xloc1][2] += eqn3_r_local[6];
        r[tElm+1][yloc1][xElm ][2] += eqn3_r_local[7];

        r[tElm  ][yElm ][xElm ][3] += eqn4_r_local[0];
        r[tElm  ][yElm ][xloc1][3] += eqn4_r_local[1];
        r[tElm  ][yloc1][xloc1][3] += eqn4_r_local[2];
        r[tElm  ][yloc1][xElm ][3] += eqn4_r_local[3];
        r[tElm+1][yElm ][xElm ][3] += eqn4_r_local[4];
        r[tElm+1][yElm ][xloc1][3] += eqn4_r_local[5];
        r[tElm+1][yloc1][xloc1][3] += eqn4_r_local[6];
        r[tElm+1][yloc1][xElm ][3] += eqn4_r_local[7];

          
        }
      }
    }


  VecAssemblyBegin(Rl); VecAssemblyEnd(Rl);

  DMDAVecRestoreArrayRead(dm,Ul,&u);
  DMDAVecRestoreArray    (dm,Rl,&r);
  DMLocalToGlobalBegin(dm,Rl,ADD_VALUES,R);
  DMLocalToGlobalEnd  (dm,Rl,ADD_VALUES,R);

  if (zs==0){
    PetscScalar (***rg)[4];
    PetscScalar (***ug)[4];
    DMDAVecGetArray    (dm,R,&rg);
    DMDAVecGetArrayRead(dm,Ul,&ug);
    for (PetscInt y=ys; y<ys+ym; ++y){
      PetscReal y_phys = user->yL + y*user->hy;
      for (PetscInt x=xs; x<xs+xm; ++x){
        PetscReal x_phys = user->xL + x*user->hx;
        rg[0][y][x][0] = ug[0][y][x][0] - (1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys));
        rg[0][y][x][1] = ug[0][y][x][1] - 0.0;
        rg[0][y][x][2] = ug[0][y][x][2] - 1.0;
        rg[0][y][x][3] = ug[0][y][x][3] - (PETSC_PI * PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys));
      }
    }

 

      DMDAVecRestoreArrayRead(dm,Ul,&ug);
      DMDAVecRestoreArray    (dm,R,&rg);
  }

  VecAssemblyBegin(R);  VecAssemblyEnd(R);
  DMRestoreLocalVector(dm,&Ul);
  DMRestoreLocalVector(dm,&Rl);

  PetscFunctionReturn(0);
}
