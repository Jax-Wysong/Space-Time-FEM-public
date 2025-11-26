#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"
#include "energies.h"
#include "ic.h"
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  AppCtx user;
  user.nx   = 40;  user.ny   = 40;  user.nt   = 240;
  user.xL   = 0.0; user.xR   = 1.0;
  user.yL   = 0.0; user.yR   = 1.0;
  user.t0   = 0.0; user.tF   = 6.0;
  user.mphi2 = 0.0; user.mchi2 = 0.0; /* mphi2 = m^2 so whatever mass you implement is the squared version*/
  user.A    = 1.0; 
  user.C    = 1.0; 

  /* pick potential V */
  user.lam22= 0.0;
  user.lam_phi6 = 0.0;
  user.g66 = 1.0;
  user.lambda_66 = 1.0;

  user.ghost = -1.0;  /* ghost on (-1) ghost off (+1) */
  user.IC    = 0;     /* controls which IC is being used */

	/* ---- defaults for phase-correlated PW -IC 4 ---- */
	user.pw_r     = 1.0;
	user.pw_dphi  = 0.0;
	user.pw_sigma = -1;

 	/* ---- defaults for Oscillon-like -IC 5 ---- */
	user.osc_x0    = 0.5*(user.xL + user.xR);
  user.osc_y0    = 0.5*(user.yL + user.yR);
	user.osc_sigma = 0.05*(user.xR - user.xL); 
	user.osc_r     = 1.0;
	user.osc_dphi  = 0.0;
  /* 
  note that carrier is controlled by C. 
  If you set C > 0 with -IC 5, then it implements
  k0 = C * 2 pi /L
  */

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Space-time solver options",NULL);
  PetscOptionsInt  ("-nx"   ,"# x-nodes"          ,"" ,user.nx    ,&user.nx   ,NULL);
  PetscOptionsInt  ("-ny"   ,"# y-nodes"          ,"" ,user.ny    ,&user.ny   ,NULL);
  PetscOptionsInt  ("-nt"   ,"# t-nodes"          ,"" ,user.nt    ,&user.nt   ,NULL);
  PetscOptionsReal ("-xL"   ,"left  x"            ,"" ,user.xL    ,&user.xL   ,NULL);
  PetscOptionsReal ("-xR"   ,"right x"            ,"" ,user.xR    ,&user.xR   ,NULL);
  PetscOptionsReal ("-yL"   ,"left  y"            ,"" ,user.yL    ,&user.yL   ,NULL);
  PetscOptionsReal ("-yR"   ,"right y"            ,"" ,user.yR    ,&user.yR   ,NULL);
  PetscOptionsReal ("-t0"   ,"initial t"          ,"" ,user.t0    ,&user.t0   ,NULL);
  PetscOptionsReal ("-tF"   ,"final   t"          ,"" ,user.tF    ,&user.tF   ,NULL);
  PetscOptionsReal ("-mphi2" ,"mass phi squared"  ,"" ,user.mphi2 ,&user.mphi2 ,NULL);
  PetscOptionsReal ("-mchi2" ,"mass chi squared"  ,"" ,user.mchi2 ,&user.mchi2 ,NULL);
  PetscOptionsReal ("-A"    ,"amplitude"          ,"" ,user.A     ,&user.A    ,NULL);
  PetscOptionsReal ("-C"    ,"controls wave numebr","" ,user.C     ,&user.C    ,NULL);
  PetscOptionsReal ("-ghost","ghost system toggle","" ,user.ghost ,&user.ghost,NULL);
  /* ------------  choosing potential V at run time --------------*/
  PetscOptionsReal ("-lam22","lambda_22 potential"   ,"" ,user.lam22 , &user.lam22,NULL);
  PetscOptionsReal ("-lam_phi6","phi^6 for oscillons","" ,user.lam_phi6 ,&user.lam_phi6,NULL);
	PetscOptionsReal ("-g66" ,"parameter within the potential","" ,user.g66 ,&user.g66 ,NULL);
	PetscOptionsReal ("-lam66" ,"parameter within the potential","" ,user.lambda_66 ,&user.lambda_66 ,NULL);
  /* ------------  IC specific options -------------- */
  PetscOptionsInt  ("-IC"   ,"# initial conditions"      ,"" ,user.IC         ,&user.IC,NULL);
  /* ------------ -IC 4 plane wave ------------------- */
  PetscOptionsInt  ("-pw_sigma","plane wave sigma"       ,"" ,user.pw_sigma   ,&user.pw_sigma   ,NULL);
  PetscOptionsReal ("-pw_r","plane wave amplitude ratio" ,"" ,user.pw_r       ,&user.pw_r       ,NULL);
  PetscOptionsReal ("-pw_dphi","plane wave phi offset"   ,"" ,user.pw_dphi    ,&user.pw_dphi    ,NULL);
  /* ------------ -IC 5 oscillon ------------------- */
	PetscOptionsReal ("-osc_sigma" ,"oscillon sigma"       ,"" ,user.osc_sigma  ,&user.osc_sigma  ,NULL);
	PetscOptionsReal ("-osc_r" ,"oscillon ratio"           ,"" ,user.osc_r      ,&user.osc_r      ,NULL);
	PetscOptionsReal ("-osc_dphi" ,"oscillon offset"       ,"" ,user.osc_dphi   ,&user.osc_dphi   ,NULL);
  PetscOptionsEnd();

  DM  dm;
  DMDACreate3d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC,
               DM_BOUNDARY_PERIODIC,
               DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX,
               user.nx, user.ny, user.nt,
               PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
               4,
               1,
               NULL,NULL,NULL,
               &dm);

  DMSetFromOptions(dm);
  DMDASetStencilType(dm,DMDA_STENCIL_BOX);
  DMSetUp(dm);


  user.hx = (user.xR-user.xL)/user.nx;
  user.hy = (user.yR-user.yL)/user.ny;
  user.ht = (user.tF-user.t0)/(user.nt-1);
  Compute_linear_stiffness(user.A_time, user.A_space_x, user.A_space_y, user.A_mass, user.hx, user.hy, user.ht);
  user.dm = dm;

  { /* Print statements and user-define erros for clarity when running and 
    also to help keep track of parameters used for each run */
    PetscPrintf(PETSC_COMM_WORLD, "\n\nInitial condtion options (-IC #):\n1) Plane Wave\n2) Gaussian Packet\n\n4) Phase-Correlated Two-Field Plane Waves\n5) Oscillon-Like, Time-Symmetric Seeds\n");
    PetscPrintf(PETSC_COMM_WORLD, "\n\nPotential options:\n1) -lam22 1.0 -> V = lambda * phi^2 * chi^2\n2) -lam_phi6 1.0 -> V = 1/2 m^2 W - lambda/4 W^2 + g/6 W^3 (W = phi^2 + chi^2)\n");

    if(user.IC == 0)
      {
        PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY INITIAL CONDITION !!!\n");
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "User-defined fatal error!\nForgot to specify initial condition\n\nInitial condtion options CLI (-IC #):\n1) Plane Wave\n2) Gaussian Packet\n\n4) Phase-Correlated Two-Field Plane Waves\n5) Oscillon-Like, Time-Symmetric Seeds\n");
      }
    if(user.lam22 == 0.0 && user.lam_phi6 == 0.0)
    {
      PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY POTENTIAL V TERM !!!\n");
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "User-defined fatal error!\nForgot to specify potential V term\n\nPotential options:\n1) -lam22 1.0 -> V = lambda * phi^2 * chi^2\n2) -lam_phi6 1.0 -> 1/2 m^2 W - lambda/6 W^2 + g/6 W^3 (W = phi^2 + chi^2)\n");
    }

    if(user.lam22 == 1.0 && user.lam_phi6 == 1.0)
    {
      PetscPrintf(PETSC_COMM_WORLD, "\n!!! TRIED TO SPECIFY DIFFERENT POTENTIAL V TERMs !!!\n");
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "User-defined fatal error!\nTried to specify different potential V terms, only turn on one of these\n\nPotential options:\n1) -lam22 1.0 -> V = lambda * phi^2 * chi^2\n2) -lam_phi6 1.0 -> 1/2 m^2 W - lambda/6 W^2 + g/6 W^3 (W = phi^2 + chi^2)\n");
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n===== Running with -IC %d =====\nUsing nx = %d, ny = %d, nt = %d\nmphi2 = %g, mchi2 = %g\nA = %g\nC = %g\nlam22 = %g\nlam_phi6 = %g\n",
                (int)user.IC, (int)user.nx, (int)user.ny, (int)user.nt, user.mphi2, user.mchi2, user.A, user.C, user.lam22, user.lam_phi6);
    PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nyL = %g, yR = %g\nt0 = %g, tF = %g\n",
                user.xL, user.xR, user.yL, user.yR, user.t0, user.tF);
    PetscPrintf(PETSC_COMM_WORLD, "hx = %.5f, hy = %.5f, ht = %.5f\n",
                (double)user.hx, (double)user.hy, (double)user.ht);

      if(user.IC == 4)
      {
        PetscPrintf(PETSC_COMM_WORLD, "===== IC 4 Specific Params ===== \nr = %g\nsigma = %d\ndphi = %g\n\n", (double)user.pw_r, (int)user.pw_sigma, (double)user.pw_dphi);
      }

    if(user.IC == 5)
      {
        PetscPrintf(PETSC_COMM_WORLD, "===== IC 5 Specific Params ===== \nsigma = %g\ndphi = %g\nr = %g\nC = %g\n\n", (double)user.osc_sigma, (double)user.osc_dphi, (double)user.osc_r, (double)user.C);
      }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Solve (T=[%.3f,%.3f], nt=%d) ===\n",
                (double)user.t0,(double)user.tF,user.nt);
  }



  Mat J;  DMCreateMatrix(dm,&J);

  SNES snes;
  SNESCreate(PETSC_COMM_WORLD,&snes);
  SNESSetDM(snes,dm);
  SNESSetFunction(snes,NULL,FormResidual,&user);
  SNESSetJacobian(snes,J,J,FormJacobian,&user);
  SNESSetFromOptions(snes);

  Vec  U;   DMCreateGlobalVector(dm,&U);

  { /* Seeding intial solution guess with Initial Conditions specified by user */
    DMDALocalInfo info;  DMDAGetLocalInfo(dm,&info);
    PetscInt xs=info.xs, xm=info.xm;
    PetscInt ys=info.ys, ym=info.ym;
    PetscInt zs=info.zs, zm=info.zm;
    PetscScalar (***u)[4];
    DMDAVecGetArray(dm,U,&u);
    for (PetscInt t=zs; t<zs+zm; ++t){
        for (PetscInt y=ys; y<ys+ym; ++y){
            PetscReal y_phys = user.yL + y*user.hy;
            for (PetscInt x=xs; x<xs+xm; ++x){
                PetscReal x_phys = user.xL + x*user.hx;
                if (t==0){
                  if(user.IC == 1){
                    u[t][y][x][0] = phi_wave_IC(x_phys,y_phys,user.A, user.C);
                    u[t][y][x][1] = u_wave_IC  (x_phys,y_phys,user.mphi2,user.A, user.C);
                    u[t][y][x][2] = chi_wave_IC(x_phys,y_phys,user.A, user.C);
                    u[t][y][x][3] = v_wave_IC  (x_phys,y_phys,user.mchi2,user.A, user.C);
                  }
                  if(user.IC == 2)
                  {
                    u[t][y][x][0] = phi_gauss_IC(x_phys,y_phys,user.A,user.C);
                    u[t][y][x][1] = u_gauss_IC  (x_phys,y_phys,user.mphi2,user.A,user.C);
                    u[t][y][x][2] = chi_gauss_IC(x_phys,y_phys,user.A,user.C);
                    u[t][y][x][3] = v_gauss_IC  (x_phys,y_phys,user.mchi2,user.A,user.C);
                  }
                  if(user.IC == 4)
                  {
                    u[t][y][x][0] = phi_pw_IC(x_phys, y_phys,user.A, user.C);
                    u[t][y][x][1] = u_pw_IC  (x_phys, y_phys,user.mphi2, user.A, user.C);
                    u[t][y][x][2] = chi_pw_IC(x_phys, y_phys,user.A, user.pw_r,user.pw_dphi, user.C);
                    u[t][y][x][3] = v_pw_IC  (x_phys, y_phys,user.mchi2, user.A, user.pw_r, user.pw_dphi, user.pw_sigma, user.C);
                  }
                  if (user.IC == 5) {
                    u[t][y][x][0] = phi_osc_IC(x_phys, y_phys,user.A,
                                user.osc_x0, user.osc_y0, user.osc_sigma, user.C);
                    u[t][y][x][1] = u_osc_IC  (x_phys, y_phys,user.mphi2, user.A,
                                user.osc_x0, user.osc_y0, user.osc_sigma, user.C);
                    u[t][y][x][2] = chi_osc_IC(x_phys, y_phys,user.A, user.osc_r,
                                user.osc_x0, user.osc_y0, user.osc_sigma, user.osc_dphi, user.C);
                    u[t][y][x][3] = v_osc_IC  (x_phys, y_phys,user.mchi2, user.A, user.osc_r,
                                user.osc_x0, user.osc_y0, user.osc_sigma, user.osc_dphi, user.C);
                  }
                }else {
                  u[t][y][x][0] = u[t][y][x][1] = u[t][y][x][2] = u[t][y][x][3] = 0.0;
                }
          }
        }
    }
    DMDAVecRestoreArray(dm,U,&u);
  }


  SNESSolve(snes,NULL,U);

  PetscInt save = 1;
  if (save) {
    Vec HphiVec, HchiVec;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nt, &HphiVec);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nt, &HchiVec);
    PetscObjectSetName((PetscObject)HphiVec, "Hphi");
    PetscObjectSetName((PetscObject)HchiVec, "Hchi");

    PetscInt rstart, rend;
    VecGetOwnershipRange(HphiVec, &rstart, &rend);

    for (PetscInt t = 0; t < user.nt; ++t) {
      PetscReal Hp, Hc;
      SliceEnergies(dm, U, t, &user, &Hp, &Hc);
      if (t >= rstart && t < rend) {
        PetscScalar shp = (PetscScalar)Hp, shc = (PetscScalar)Hc;
        VecSetValues(HphiVec, 1, &t, &shp, INSERT_VALUES);
        VecSetValues(HchiVec, 1, &t, &shc, INSERT_VALUES);
      }
    }
    VecAssemblyBegin(HphiVec); VecAssemblyEnd(HphiVec);
    VecAssemblyBegin(HchiVec); VecAssemblyEnd(HchiVec);

    Vec Unat;
    DMDACreateNaturalVector(dm, &Unat);
    DMDAGlobalToNaturalBegin(dm, U, INSERT_VALUES, Unat);
    DMDAGlobalToNaturalEnd  (dm, U, INSERT_VALUES, Unat);
    VecSetBlockSize(Unat, 4);

    PetscInt Nloc, Nglob;
    VecGetLocalSize(Unat, &Nloc);
    VecGetSize(Unat, &Nglob);
    PetscInt nloc = Nloc/4, nglob = Nglob/4;

    Vec phi, chi;
    VecCreateMPI(PETSC_COMM_WORLD, nloc, nglob, &phi);
    VecDuplicate(phi, &chi);
    PetscObjectSetName((PetscObject)phi, "phi");
    PetscObjectSetName((PetscObject)chi, "chi");

    const PetscScalar *ua;
    VecGetArrayRead(Unat, &ua);

    PetscInt urstart; VecGetOwnershipRange(Unat, &urstart, NULL);
    PetscInt bstart = urstart/4;

    for (PetscInt i = 0; i < nloc; ++i) {
      PetscInt g = bstart + i;
      PetscScalar vphi = ua[4*i + 0];
      PetscScalar vchi = ua[4*i + 2];
      VecSetValues(phi, 1, &g, &vphi, INSERT_VALUES);
      VecSetValues(chi, 1, &g, &vchi, INSERT_VALUES);
    }
    VecRestoreArrayRead(Unat, &ua);
    VecAssemblyBegin(phi); VecAssemblyEnd(phi);
    VecAssemblyBegin(chi); VecAssemblyEnd(chi);

    Vec vx=NULL, vy=NULL, vt=NULL;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nx, &vx);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.ny, &vy);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nt, &vt);
    PetscObjectSetName((PetscObject)vx, "x");
    PetscObjectSetName((PetscObject)vy, "y");
    PetscObjectSetName((PetscObject)vt, "t");

    PetscInt rs,re,i; PetscScalar val;
    VecGetOwnershipRange(vx,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.xL + (i)*(user.hx)); VecSetValues(vx,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vx); VecAssemblyEnd(vx);

    VecGetOwnershipRange(vy,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.yL + (i)*(user.hy)); VecSetValues(vy,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vy); VecAssemblyEnd(vy);

    VecGetOwnershipRange(vt,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.t0 + (i)*(user.ht)); VecSetValues(vt,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vt); VecAssemblyEnd(vt);

  /*---------- name the file conveniently ---------- */
    char fname[PETSC_MAX_PATH_LEN];
    if (user.IC == 1){
      PetscSNPrintf(fname,sizeof(fname),
            "lam22_%g_lam_phi6_%g_IC_%d_A%.5g_C%.5g_1x%.5g_%dx%dx%d.dat",(double)user.lam22, (double)user.lam_phi6, (int)user.IC, (double)user.A, (double)user.C, (double)user.tF, (int)user.nx, (int)user.ny, (int)user.nt);		
    }
    if (user.IC == 2){
      PetscSNPrintf(fname,sizeof(fname),
            "lam22_%g_lam_phi6_%g_IC_%d_A%.5g_C%.5g_1x%.5g_%dx%dx%d.dat",(double)user.lam22, (double)user.lam_phi6, (int)user.IC, (double)user.A, (double)user.C, (double)user.tF, (int)user.nx, (int)user.ny, (int)user.nt);		
    }
    if (user.IC == 4){
      PetscSNPrintf(fname,sizeof(fname),
            "lam22_%g_lam_phi6_%g_IC_%d_A%.5g_C%.5g_sigma_%d_r%.5g_dphi_%.5g_1x%.5g_%dx%dx%d.dat",(double)user.lam22, (double)user.lam_phi6, (int)user.IC, (double)user.A, (double)user.C, (int)user.pw_sigma, (double)user.pw_r, (double)user.pw_dphi, (double)user.tF, (int)user.nx, (int)user.ny, (int)user.nt);		
    }
    if (user.IC == 5){
      PetscSNPrintf(fname,sizeof(fname),
            "lam22_%g_lam_phi6_%g_IC_%d_A%.5g_C_%.5g_sigma_%.5g_1x%.5g_%dx%dx%d.dat",(double)user.lam22, (double)user.lam_phi6, (int)user.IC, (double)user.A, (double)user.C, (double)user.osc_sigma, (double)user.tF, (int)user.nx, (int)user.ny, (int)user.nt);		
    }

    PetscViewer viewer = NULL;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_WRITE, &viewer);

    VecView(phi,     viewer);
    VecView(chi,     viewer);
    VecView(HphiVec, viewer);
    VecView(HchiVec, viewer);
    VecView(vx,      viewer);
    VecView(vy,      viewer);
    VecView(vt,      viewer);

    PetscViewerDestroy(&viewer);

    VecDestroy(&vx); VecDestroy(&vy); VecDestroy(&vt);
    VecDestroy(&HphiVec); VecDestroy(&HchiVec);
    VecDestroy(&phi); VecDestroy(&chi);
    VecDestroy(&Unat);

    PetscPrintf(PETSC_COMM_WORLD,"Saved phi, chi, Hphi, Hchi (plus x,y,t) to PETSc binary.\n");
  }

  VecDestroy(&U);   SNESDestroy(&snes);   DMDestroy(&dm); MatDestroy(&J);
  PetscFinalize();
  return 0;
}