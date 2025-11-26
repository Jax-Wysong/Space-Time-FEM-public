#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "ic.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"
#include <petscviewer.h>


/*----------------------------------------------------*
   Parallel driver with DMDA
 *----------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  /* ------------ user parameters + command-line opts ------------ */
  AppCtx user;
  user.nx   = 100;  user.nt   = 101;
  user.xL   = 0.0;  user.xR   = 1.0;
  user.t0   = 0.0;  user.tF   = 1.0;
  user.mphi2 = 0.0; user.mchi2 = 0.0;
  user.A    = 1.0;  user.lam22 = 1.0;
  user.IC = 0;		
  user.ghost = -1.0;



  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
                    "Space-time solver options",NULL);
  PetscOptionsInt  ("-nx"   ,"# x-nodes"      ,"" ,user.nx   ,&user.nx   ,NULL);
  PetscOptionsInt  ("-nt"   ,"# t-nodes"      ,"" ,user.nt   ,&user.nt   ,NULL);
  PetscOptionsInt  ("-IC"   ,"# initial conditions","" ,user.IC   ,&user.IC,NULL);
  PetscOptionsReal ("-Ghost"   ,"Ghost on/off"        ,"" ,user.ghost   ,&user.ghost   ,NULL);
  PetscOptionsReal ("-xL"   ,"left  x"        ,"" ,user.xL   ,&user.xL   ,NULL);
  PetscOptionsReal ("-xR"   ,"right x"        ,"" ,user.xR   ,&user.xR   ,NULL);
  PetscOptionsReal ("-t0"   ,"initial t"      ,"" ,user.t0   ,&user.t0   ,NULL);
  PetscOptionsReal ("-tF"   ,"final   t"      ,"" ,user.tF   ,&user.tF   ,NULL);
  PetscOptionsReal ("-mphi2" ,"mass φ"         ,"" ,user.mphi2 ,&user.mphi2 ,NULL);
  PetscOptionsReal ("-mchi2" ,"mass χ"         ,"" ,user.mchi2 ,&user.mchi2 ,NULL);
  PetscOptionsReal ("-A"    ,"amplitude"      ,"" ,user.A    ,&user.A    ,NULL);
  PetscOptionsReal ("-lam22","lambda_22"            ,"" ,user.lam22,&user.lam22,NULL);
  PetscOptionsEnd();



  /* ------------ build the DM ------------ */
  DM  dm;
  DMDACreate2d(PETSC_COMM_WORLD,                /* communicator          */
               DM_BOUNDARY_PERIODIC,            /* x boundary (periodic) */
               DM_BOUNDARY_NONE,                /* t boundary            */
               DMDA_STENCIL_BOX,                /* stencil               */
               user.nx, user.nt,                /* global grid           */
               PETSC_DECIDE,PETSC_DECIDE,       /* owner splits          */
               /* ----- dof  ---- */ 4,         /* (phi,u,chi,v) per node      */
               /* stencil width */ 1,           /* need +1 in x,+1 in t  */
               NULL,NULL,                       /* no custom d-grid      */
               &dm);
			   
  DMSetFromOptions(dm);   /* allow -da_* flags   */
  DMDASetStencilType(dm,DMDA_STENCIL_BOX);  /* override STAR if still there */
  DMSetUp(dm);
  
  PetscPrintf(PETSC_COMM_WORLD, "\n\nInitial condtion options (-IC #):\n1) Plane Wave\n2) Gaussian Packet\n3) Colored-Noise Spectra\n4) Phase-Correlated Two-Field Plane Waves\n5) Oscillon-Like, Time-Symmetric Seeds\n6) Domain-Wall/Bubble Data\n");

	if(user.IC == 0)
	{
		PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY INITIAL CONDITION !!!\n");
	}


  user.hx = (user.xR-user.xL)/user.nx;
  user.ht = (user.tF-user.t0)/(user.nt-1);
  user.L  = user.xR - user.xL;
  Compute_linear_stiffness(user.A_time, user.A_space, user.A_standard, user.hx, user.ht);
  user.dm = dm;           /* <-- FormResidual/Jacobian read it */


  /* ------------ create SNES + work vectors/mats ------------ */
  
  Mat J;
  DMCreateMatrix(dm,&J);
  
  SNES snes;
  SNESCreate(PETSC_COMM_WORLD,&snes);
  SNESSetDM(snes,dm);                       /* tells SNES to use dm */
  SNESSetFunction(snes,NULL,FormResidual,&user);
  SNESSetJacobian(snes,J,J,FormJacobian,&user);
  SNESSetFromOptions(snes);                 /* allow command line options to change snes options */

  Vec  U;   /* solution vector */
  DMCreateGlobalVector(dm,&U);     

  /* ------------ fill initial conditions on rank-local part ------------ */
  {
    DMDALocalInfo info;  DMDAGetLocalInfo(dm,&info);
    PetscInt xs=info.xs, xm=info.xm;
    PetscInt ys=info.ys, ym=info.ym;
    PetscScalar (**u)[4];
    DMDAVecGetArray(dm,U,&u);
    for (PetscInt t=ys; t<ys+ym; ++t){
      for (PetscInt x=xs; x<xs+xm; ++x){
        PetscReal x_phys = user.xL + x*user.hx;
        if (t==0){                                 /* IC at t=0      */
          if(user.IC == 1)
          {
            u[t][x][0] = phi_MMS_IC(x_phys);
            u[t][x][1] = u_MMS_IC  (x_phys);
            u[t][x][2] = chi_MMS_IC(x_phys);
            u[t][x][3] = v_MMS_IC  (x_phys);
          }
        }else{
          u[t][x][0] = u[t][x][1] = u[t][x][2] = u[t][x][3] = 0.0;
        }
      }
    }
    DMDAVecRestoreArray(dm,U,&u);
  }

  /* ------------ solve ------------ */
  
	PetscPrintf(PETSC_COMM_WORLD, "\n=================== MMS IC %d ===================\n\nUsing nx = %d, nt = %d\nmphi2 = %g, mchi2 = %g\nlam22 = %g\nGhost = %g\n", (int)user.IC, (int)user.nx, (int)user.nt, user.mphi2, user.mchi2, user.lam22,user.ghost);
	PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nt0 = %g, tF = %g\n", user.xL, user.xR, user.t0, user.tF);
	PetscPrintf(PETSC_COMM_WORLD, "hx = %.8e, ht = %.8e\n\n", (double)user.hx, (double)user.ht);

    PetscPrintf(PETSC_COMM_WORLD,
      "=== Solve (T=[%.3f,%.3f], nt=%d) ===\n", (double)user.t0,(double)user.tF,user.nt);

    SNESSolve(snes,NULL,U); /* solution vector is U */
	
  /* Parallel-safe L2 and max error norms for phi and chi */
  {
    DMDALocalInfo info; DMDAGetLocalInfo(dm,&info);
    /* Local accumulators */
    PetscReal l2_phi_local = 0.0, l2_chi_local = 0.0;
    PetscReal max_phi_local = 0.0, max_chi_local = 0.0;

    /* Access distributed array with dof */
    PetscScalar (**a)[4];  /* a[t][x][comp] */
    DMDAVecGetArrayRead(dm, U, &a);

    /* Global starts for physical coords */
    for (PetscInt tj = info.ys; tj < info.ys + info.ym; ++tj) {
      PetscReal t_phys = user.t0 + tj * user.ht;
      for (PetscInt xi = info.xs; xi < info.xs + info.xm; ++xi) {
        PetscReal x_phys = user.xL + xi * user.hx;

        /* Manufactured analytic solutions */
        PetscReal u_exact_phi = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * t_phys);
        PetscReal u_exact_chi = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscSinReal(PETSC_PI * t_phys);

        /* Extract solution values (dof 0 = phi, dof 2 = chi) */
        PetscReal diff_phi = PetscRealPart(a[tj][xi][0]) - u_exact_phi;
        PetscReal diff_chi = PetscRealPart(a[tj][xi][2]) - u_exact_chi;

        l2_phi_local += diff_phi * diff_phi;
        l2_chi_local += diff_chi * diff_chi;

        PetscReal ap = PetscAbsReal(diff_phi);
        PetscReal ac = PetscAbsReal(diff_chi);
        if (ap > max_phi_local) max_phi_local = ap;
        if (ac > max_chi_local) max_chi_local = ac;
      }
    }

    DMDAVecRestoreArrayRead(dm, U, &a);

    /* Global reductions */
    PetscReal l2_phi_global = 0.0, l2_chi_global = 0.0;
    PetscReal max_phi_global = 0.0, max_chi_global = 0.0;

    MPIU_Allreduce(&l2_phi_local, &l2_phi_global, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    MPIU_Allreduce(&l2_chi_local, &l2_chi_global, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    MPIU_Allreduce(&max_phi_local, &max_phi_global, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);
    MPIU_Allreduce(&max_chi_local, &max_chi_global, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);

    /* Scale for discrete L2 norm over the (t,x) grid */
    PetscReal error_L2_phi = PetscSqrtReal(l2_phi_global * user.hx * user.ht);
    PetscReal error_L2_chi = PetscSqrtReal(l2_chi_global * user.hx * user.ht);

    PetscPrintf(PETSC_COMM_WORLD, "PHI  L2 error norm  = %.8e\n", (double)error_L2_phi);
    PetscPrintf(PETSC_COMM_WORLD, "PHI  max error norm = %.8e\n", (double)max_phi_global);
    PetscPrintf(PETSC_COMM_WORLD, "CHI  L2 error norm  = %.8e\n", (double)error_L2_chi);
    PetscPrintf(PETSC_COMM_WORLD, "CHI  max error norm = %.8e\n", (double)max_chi_global);
  }




  /* ---------------------------------------------------------------------
              Dump field snapshots to a MATLAB script
    --------------------------------------------------------------------- */
  PetscInt save = 1;

  if (save) {
    /* U -> natural ordering (still distributed). x-fastest, then t */
    Vec Unat;
    DMDACreateNaturalVector(user.dm, &Unat);
    DMDAGlobalToNaturalBegin(user.dm, U, INSERT_VALUES, Unat);
    DMDAGlobalToNaturalEnd  (user.dm, U, INSERT_VALUES, Unat);
    VecSetBlockSize(Unat, 4); /* enforce bs=4: [phi,ut,chi,vt] */

    /* Manually extract components 0 (phi) and 2 (chi) into distributed Vecs */
    PetscInt Nloc, Nglob; 
    VecGetLocalSize(Unat, &Nloc);
    VecGetSize(Unat, &Nglob);
    PetscInt nloc = Nloc/4, nglob = Nglob/4;  /* number of (x,t) points */

    Vec phi, chi;
    VecCreateMPI(PETSC_COMM_WORLD, nloc, nglob, &phi);
    VecDuplicate(phi, &chi);
    PetscObjectSetName((PetscObject)phi, "phi");
    PetscObjectSetName((PetscObject)chi, "chi");

    const PetscScalar *ua;
    VecGetArrayRead(Unat, &ua);

    PetscInt urstart; VecGetOwnershipRange(Unat, &urstart, NULL);
    PetscInt bstart = urstart/4; /* global block index of local start */

    for (PetscInt i = 0; i < nloc; ++i) {
      PetscInt g = bstart + i;                /* global index in phi/chi (0..nx*nt-1) */
      PetscScalar vphi = ua[4*i + 0];
      PetscScalar vchi = ua[4*i + 2];
      VecSetValues(phi, 1, &g, &vphi, INSERT_VALUES);
      VecSetValues(chi, 1, &g, &vchi, INSERT_VALUES);
    }
    VecRestoreArrayRead(Unat, &ua);
    VecAssemblyBegin(phi); VecAssemblyEnd(phi);
    VecAssemblyBegin(chi); VecAssemblyEnd(chi);

    /* Coordinates x (nx) and time t (nt) as Vecs */
    Vec vx=NULL, vt=NULL;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nx, &vx);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nt, &vt);
    PetscObjectSetName((PetscObject)vx, "x");
    PetscObjectSetName((PetscObject)vt, "t");

    PetscInt rs,re,i; PetscScalar val;
    VecGetOwnershipRange(vx,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.xL + (i)*(user.hx)); VecSetValues(vx,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vx); VecAssemblyEnd(vx);

    VecGetOwnershipRange(vt,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.t0 + (i)*(user.ht)); VecSetValues(vt,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vt); VecAssemblyEnd(vt);

    /* Viewer: PETSc binary */
      char fname[PETSC_MAX_PATH_LEN];
    
    if (user.IC == 1){
      PetscSNPrintf(fname,sizeof(fname),
            "MMS_IC_%d_1x%.5g_%dx%d.dat",(int)user.IC, (double)user.tF, (int)user.nx, (int)user.nt);		
    }


    PetscViewer viewer = NULL;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname , FILE_MODE_WRITE, &viewer);
    /* Write what you need */
    VecView(phi,     viewer);
    VecView(chi,     viewer);
    VecView(vx,      viewer);
    VecView(vt,      viewer);

    PetscViewerDestroy(&viewer);

    /* 6) Cleanup */
    VecDestroy(&vx); VecDestroy(&vt);
    VecDestroy(&phi); VecDestroy(&chi);
    VecDestroy(&Unat);

    PetscPrintf(PETSC_COMM_WORLD,"Saved (1+1) phi, chi, x, t to PETSc binary.\n");

  }





    /* ------------ clean up ------------ */
  SNESDestroy(&snes); VecDestroy(&U); MatDestroy(&J); DMDestroy(&dm);
  PetscFinalize();
  return 0;
}
