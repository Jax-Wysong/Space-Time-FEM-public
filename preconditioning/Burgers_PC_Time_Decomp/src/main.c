#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "ic.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"
#include "asm.h"
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
  user.nx   = 100;  user.nt   = 100;
  user.xL   = 0.0;  user.xR   = 2.0*PETSC_PI;
  user.t0   = 0.0;  user.tF   = 2.0;
  user.IC = 0;		
  user.Nsub = 1;
  user.overlap = 1;
  user.interface_width = 1; 
  user.slab_pc_nonlinear = PETSC_FALSE;
  user.slab_pc_ras = PETSC_FALSE;
  user.lag_freq = 1;
  user.save = 1;

  user.viscosity = 0.07;


  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
                    "Space-time solver options",NULL);
  PetscOptionsInt  ("-save"   ,"save solution to binary file? (0 or 1)"      ,"" ,user.save   ,&user.save   ,NULL);
  PetscOptionsInt  ("-nx"   ,"# x-nodes"      ,"" ,user.nx   ,&user.nx   ,NULL);
  PetscOptionsInt  ("-nt"   ,"# t-nodes"      ,"" ,user.nt   ,&user.nt   ,NULL);
  PetscOptionsInt  ("-IC"   ,"# initial conditions","" ,user.IC   ,&user.IC,NULL);
  PetscOptionsInt("-Nsub","number of time slabs","",user.Nsub,&user.Nsub,NULL);
  PetscOptionsInt("-overlap","overlap in time levels","",user.overlap,&user.overlap,NULL);
  PetscOptionsInt("-interface_width","width of interface","",user.interface_width,&user.interface_width,NULL);
  PetscOptionsBool("-slab_pc_nonlinear","use nonlinear slab PC","",user.slab_pc_nonlinear,&user.slab_pc_nonlinear,NULL);
  PetscOptionsBool("-slab_pc_ras","use RAS slab PC","",user.slab_pc_ras,&user.slab_pc_ras,NULL);
  PetscOptionsInt("-lag_freq","rebuild slab PC every N Newton steps (1=every step, 0=never after first)","",user.lag_freq,&user.lag_freq,NULL);
  PetscOptionsReal ("-xL"   ,"left  x"        ,"" ,user.xL   ,&user.xL   ,NULL);
  PetscOptionsReal ("-xR"   ,"right x"        ,"" ,user.xR   ,&user.xR   ,NULL);
  PetscOptionsReal ("-t0"   ,"initial t"      ,"" ,user.t0   ,&user.t0   ,NULL);
  PetscOptionsReal ("-tF"   ,"final   t"      ,"" ,user.tF   ,&user.tF   ,NULL);
  PetscOptionsReal ("-viscosity" ,"viscosity"         ,"" ,user.viscosity ,&user.viscosity ,NULL);
  PetscOptionsEnd();


  PetscInt swidth = user.overlap + user.interface_width;
  if (swidth == 0){ swidth = 1;}
  PetscPrintf(PETSC_COMM_WORLD, "Using overlap %d and interface width %d, so stencil width = %d\n", user.overlap, user.interface_width, swidth);
  /* ------------ build the DM ------------ */
  DM  dm;
  DMDACreate2d(PETSC_COMM_WORLD,                /* communicator          */
               DM_BOUNDARY_PERIODIC,            /* x boundary (periodic) */
               DM_BOUNDARY_NONE,                /* t boundary            */
               DMDA_STENCIL_BOX,                /* stencil               */
               user.nx, user.nt,                /* global grid           */
               PETSC_DECIDE,PETSC_DECIDE,       /* owner splits          */
               /* ----- dof  ---- */ 1,         /* (u) per node      */
               /* stencil width */ swidth,   /* need +1 in x,+1 in t  */
               NULL,NULL,                       /* no custom d-grid      */
               &dm);
			   
  DMSetFromOptions(dm);   /* allow -da_* flags   */
  DMDASetStencilType(dm,DMDA_STENCIL_BOX);  /* override STAR if still there */
  DMSetUp(dm);
  ierr = DMDASetFieldName(dm,0,"u");CHKERRQ(ierr);
  
	if(user.IC == 0)
	{
		PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY INITIAL CONDITION !!!\n");
	}


  user.hx = (user.xR-user.xL)/(user.nx); // note: global points = nx, so intervals = nx-1, but we want periodic so use nx
  user.ht = (user.tF-user.t0)/(user.nt-1);
  user.L  = user.xR - user.xL;
  Compute_linear_stiffness(user.A_time, user.A_space, user.A_standard, user.hx, user.ht);
  user.dm = dm;           /* <-- FormResidual/Jacobian read it */


  /* ------------ create SNES + work vectors/mats ------------ */
  
  Mat J;
  DMCreateMatrix(dm, &J);
  
  SNES snes;
  SNESCreate(PETSC_COMM_WORLD,&snes);
  SNESSetDM(snes,dm);                       /* tells SNES to use dm */
  SNESSetFunction(snes,NULL,FormResidual,&user);
  SNESSetJacobian(snes,J,J,FormJacobian,&user);
  SNESSetFromOptions(snes);                 /* allow command line options to change snes options */


  /* ------------ set up preconditioner  ------------ */
  KSP ksp;
  PC  pc;
  PetscBool use_shell = PETSC_FALSE;
  PetscOptionsGetBool(NULL,NULL,"-use_shell_pc",&use_shell,NULL);


  SNESGetKSP(snes, &ksp);
  KSPSetDM(ksp, dm);
  KSPSetDMActive(ksp, PETSC_FALSE);  
  KSPGetPC(ksp, &pc);

  if (use_shell) {
    PCSetType(pc, PCSHELL);

    SampleShellPC *shell;
    PetscNew(&shell);
    shell->user = &user;
    shell->dm   = user.dm;
    shell->snes = snes;
    shell->use_nonlinear = user.slab_pc_nonlinear;
    shell->use_ras = user.slab_pc_ras;

    PCShellSetContext(pc, shell);
    PCShellSetSetUp(pc, PCSetUp_SampleShell);
    PCShellSetApply(pc, PCApply_SampleShell);
    PCShellSetDestroy(pc, PCDestroy_SampleShell);
  }
  /* else: do nothing; -pc_type lu etc. will work */


  /* ------------ create vectors ------------ */
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);

  /* ------------ fill initial conditions on rank-local part ------------ */
  ierr = VecZeroEntries(U);CHKERRQ(ierr);


  /* ------------ solve ------------ */
  
	PetscPrintf(PETSC_COMM_WORLD, "\n=================== Viscous Burgers' Equation ===================\n\nUsing nx = %d, nt = %d\n", (int)user.nx, (int)user.nt);
	PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nt0 = %g, tF = %g\nviscosity = %g\n", user.xL, user.xR, user.t0, user.tF, (double)user.viscosity);
	PetscPrintf(PETSC_COMM_WORLD, "hx = %.8e, ht = %.8e\n\n", (double)user.hx, (double)user.ht);

    PetscPrintf(PETSC_COMM_WORLD,
      "=== Solve (T=[%.3f,%.3f], nt=%d) ===\n", (double)user.t0,(double)user.tF,user.nt);

    SNESSolve(snes,NULL,U); /* solution vector is U */

  /* Parallel-safe L2 and max error norms for u */
  {
    DMDALocalInfo info; DMDAGetLocalInfo(dm,&info);
    /* Local accumulators */
    PetscReal l2_u_local = 0.0;
    PetscReal max_u_local = 0.0;

    /* Access distributed array with dof */
    PetscScalar (**a)[1];  /* a[t][x][comp] */
    DMDAVecGetArrayRead(dm, U, &a);

    /* Global starts for physical coords */
    for (PetscInt tj = info.ys; tj < info.ys + info.ym; ++tj) {
      PetscReal t_phys = user.t0 + tj * user.ht;
      for (PetscInt xi = info.xs; xi < info.xs + info.xm; ++xi) {
        PetscReal x_phys = user.xL + xi * user.hx;

        /* analytic solutions */
        PetscReal psi_exact = PetscExpReal(-(x_phys-4*t_phys)*(x_phys-4*t_phys)/(4*user.viscosity*(t_phys+1))) + PetscExpReal(-(x_phys-4*t_phys-2*PETSC_PI)*(x_phys-4*t_phys-2*PETSC_PI)/(4*user.viscosity*(t_phys+1)));
        PetscReal psi_x_exact = -(-8*t_phys + 2*x_phys)*PetscExpReal(-(x_phys-4*t_phys)*(x_phys-4*t_phys)/(4*user.viscosity*(t_phys+1)))/(4*user.viscosity*(t_phys+1)) - (-8*t_phys + 2*x_phys - 4*PETSC_PI)*PetscExpReal(-(x_phys-4*t_phys-2*PETSC_PI)*(x_phys-4*t_phys-2*PETSC_PI)/(4*user.viscosity*(t_phys+1)))/(4*user.viscosity*(t_phys+1));
        PetscReal u_exact = -2*user.viscosity*psi_x_exact/psi_exact + 4.0;

        /* Extract solution values (dof 0 = u) */
        PetscReal diff_u = PetscRealPart(a[tj][xi][0]) - u_exact;

        l2_u_local += diff_u * diff_u;

        PetscReal abs_u = PetscAbsReal(diff_u);

        if (abs_u > max_u_local) max_u_local = abs_u;
      }
    }

    DMDAVecRestoreArrayRead(dm, U, &a);

    /* Global reductions */
    PetscReal l2_u_global = 0.0;
    PetscReal max_u_global = 0.0;

    MPIU_Allreduce(&l2_u_local, &l2_u_global, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    MPIU_Allreduce(&max_u_local, &max_u_global, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);

    /* Scale for discrete L2 norm over the (t,x) grid */
    PetscReal error_L2_u = PetscSqrtReal(l2_u_global * user.hx * user.ht);

    PetscPrintf(PETSC_COMM_WORLD, "U  L2 error norm  = %.8e\n", (double)error_L2_u);
    PetscPrintf(PETSC_COMM_WORLD, "U  max error norm = %.8e\n", (double)max_u_global);
  }




  /* ---------------------------------------------------------------------
              Dump solutions to binary for python later
    --------------------------------------------------------------------- */

  if (user.save) {
    /* U -> natural ordering (still distributed). x-fastest, then t */
    Vec Unat;
    DMDACreateNaturalVector(user.dm, &Unat);
    DMDAGlobalToNaturalBegin(user.dm, U, INSERT_VALUES, Unat);
    DMDAGlobalToNaturalEnd  (user.dm, U, INSERT_VALUES, Unat);
    VecSetBlockSize(Unat, 1); /* enforce bs=1: [u] */

    /* Manually extract components 0 (u) into distributed Vecs */
    PetscInt Nloc, Nglob; 
    VecGetLocalSize(Unat, &Nloc);
    VecGetSize(Unat, &Nglob);
    PetscInt nloc = Nloc/1, nglob = Nglob/1;  /* number of (x,t) points */

    Vec u;
    VecCreateMPI(PETSC_COMM_WORLD, nloc, nglob, &u);
    PetscObjectSetName((PetscObject)u, "u");

    const PetscScalar *ua;
    VecGetArrayRead(Unat, &ua);

    PetscInt urstart; VecGetOwnershipRange(Unat, &urstart, NULL);
    PetscInt bstart = urstart/1; /* global block index of local start */

    for (PetscInt i = 0; i < nloc; ++i) {
      PetscInt g = bstart + i;                /* global index in u/v (0..nx*nt-1) */
      PetscScalar vu = ua[i];
      VecSetValues(u, 1, &g, &vu, INSERT_VALUES);
    }
    VecRestoreArrayRead(Unat, &ua);
    VecAssemblyBegin(u); VecAssemblyEnd(u);

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
            "Burgers_visc-%.5g_1x%.5g_%dx%d.dat", (double)user.viscosity, (double)user.tF, (int)user.nx, (int)user.nt);		
    }


    PetscViewer viewer = NULL;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname , FILE_MODE_WRITE, &viewer);
    /* Write what you need */
    VecView(u,     viewer);
    VecView(vx,      viewer);
    VecView(vt,      viewer);

    PetscViewerDestroy(&viewer);

    /* 6) Cleanup */
    VecDestroy(&vx); VecDestroy(&vt);
    VecDestroy(&u);
    VecDestroy(&Unat);

    PetscPrintf(PETSC_COMM_WORLD,"Saved u, x, t to PETSc binary.\n");

  }





    /* ------------ clean up ------------ */
  SNESDestroy(&snes); VecDestroy(&U); MatDestroy(&J); DMDestroy(&dm);
  PetscFinalize();
  return 0;
}
