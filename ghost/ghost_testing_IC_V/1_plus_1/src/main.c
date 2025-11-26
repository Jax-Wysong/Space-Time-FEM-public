#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"
#include "energies.h"
#include "ic.h"
#include "fill_IC.h"
#include "saveSol.h"
#include "dm_create.h"
#include "extend_in_time.h"
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
  user.IC = 0;		user.C = 1.0; // kL/2pi = C
  user.ghost = -1.0;
  user.Loop = 0;
 
	/* ---- defaults for colored-noise -IC 3 ---- */
	user.n1   = 1;
	user.n2   = 64;
	user.ns   = -1.0;
	user.dphi = 0.0;
	PetscInt noise_seed = 12345; 
	
	/* ---- defaults for phase-correlated PW -IC 4 ---- */
	user.pw_r     = 1.0;
	user.pw_dphi  = 0.0;
	user.pw_sigma = -1;
	
	/* ---- defaults for Oscillon-like -IC 5 ---- */
	user.osc_x0    = 0.5*(user.xL + user.xR);
	user.osc_sigma = 0.05*(user.xR - user.xL);  
	user.osc_r     = 1.0;
	user.osc_dphi  = 0.0;
	user.osc_k0    = 0.0;   /* 0 -> no carrier */


  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
                    "Space-time solver options",NULL);
  PetscOptionsInt  ("-nx"   ,"# x-nodes"      ,"" ,user.nx   ,&user.nx   ,NULL);
  PetscOptionsInt  ("-nt"   ,"# t-nodes"      ,"" ,user.nt   ,&user.nt   ,NULL);
  PetscOptionsInt  ("-IC"   ,"# initial conditions","" ,user.IC   ,&user.IC,NULL);
  PetscOptionsInt  ("-Loop"   ,"Loop on or off","" ,user.Loop ,&user.Loop,NULL);
  PetscOptionsReal ("-Ghost"   ,"Ghost on/off"        ,"" ,user.ghost   ,&user.ghost   ,NULL);
  PetscOptionsReal ("-C"   ,"kL/2pi = C"        ,"" ,user.C   ,&user.C   ,NULL);
  PetscOptionsReal ("-xL"   ,"left  x"        ,"" ,user.xL   ,&user.xL   ,NULL);
  PetscOptionsReal ("-xR"   ,"right x"        ,"" ,user.xR   ,&user.xR   ,NULL);
  PetscOptionsReal ("-t0"   ,"initial t"      ,"" ,user.t0   ,&user.t0   ,NULL);
  PetscOptionsReal ("-tF"   ,"final   t"      ,"" ,user.tF   ,&user.tF   ,NULL);
  PetscOptionsReal ("-mphi2" ,"mass φ"         ,"" ,user.mphi2 ,&user.mphi2 ,NULL);
  PetscOptionsReal ("-mchi2" ,"mass χ"         ,"" ,user.mchi2 ,&user.mchi2 ,NULL);
  PetscOptionsReal ("-A"    ,"amplitude"      ,"" ,user.A    ,&user.A    ,NULL);
  PetscOptionsReal ("-lam22","λ₂₂"            ,"" ,user.lam22,&user.lam22,NULL);
	PetscOptionsInt ("-noise_n1","lowest mode n1","",user.n1,&user.n1,NULL);
	PetscOptionsInt ("-noise_n2","highest mode n2","",user.n2,&user.n2,NULL);
	PetscOptionsReal("-noise_ns","spectral tilt ns","",user.ns,&user.ns,NULL);
	PetscOptionsReal("-noise_dphi","phase offset Δφ for χ","",user.dphi,&user.dphi,NULL);
	PetscOptionsInt ("-noise_seed","random seed for phases","",noise_seed,&noise_seed,NULL);
  PetscOptionsInt ("-pw_sigma","plane wave sigma","",user.pw_sigma,&user.pw_sigma,NULL);
  PetscOptionsReal ("-pw_r","plane wave amplitude ratio","" ,user.pw_r,&user.pw_r,NULL);
  PetscOptionsReal ("-pw_dphi","plane wave phi offset","" ,user.pw_dphi,&user.pw_dphi,NULL);
	PetscOptionsReal ("-osc_sigma" ,"oscillon sigma","" ,user.osc_sigma,&user.osc_sigma    ,NULL);
	PetscOptionsReal ("-osc_r" ,"oscillon ratio","" ,user.osc_r ,&user.osc_r ,NULL);
	PetscOptionsReal ("-osc_dphi" ,"oscillon offset","" ,user.osc_dphi ,&user.osc_dphi ,NULL);
	PetscOptionsReal ("-osc_k0" ,"oscillon offset","" ,user.osc_k0 ,&user.osc_k0 ,NULL);
  PetscOptionsEnd();



  /* ------------ build the DM ------------ */
    DM  dm;
    SNES snes;
    Mat J;
    Vec U;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                            DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,
                            DMDA_STENCIL_BOX,
                            user.nx, user.nt, PETSC_DECIDE, PETSC_DECIDE,
                            4, 1, NULL, NULL, &dm));
    PetscCall(DMDASetStencilType(dm, DMDA_STENCIL_BOX));
    PetscCall(DMSetFromOptions(dm));   /* allow -da_* flags from command line  */
    PetscCall(DMSetUp(dm));

    user.hx = (user.xR - user.xL)/user.nx;
    user.ht = (user.tF - user.t0)/((PetscReal)user.nt - 1.0);
    user.L  = user.xR - user.xL;
    user.dm = dm;
    Compute_linear_stiffness(user.A_time, user.A_space, user.A_standard,
                            user.hx, user.ht);

    PetscCall(DMCreateMatrix(dm, &J));
    PetscCall(DMCreateGlobalVector(dm, &U));
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetDM(snes, dm));
    PetscCall(SNESSetFunction(snes, NULL, FormResidual, &user));
    PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &user));
    PetscCall(SNESSetFromOptions(snes));

    { /* Print statements of useful information to keep track of parameters in IC and mesh */

        

        PetscPrintf(PETSC_COMM_WORLD, "\n\nInitial condtion options (-IC #):\n1) Plane Wave\n2) Gaussian Packet\n3) Colored-Noise Spectra\n4) Phase-Correlated Two-Field Plane Waves\n5) Oscillon-Like, Time-Symmetric Seeds\n");

        if(user.IC == 0 || user.IC >= 6)
        {
            PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY INITIAL CONDITION !!!\n");
        }

        if(user.IC == 3)
        {
            PetscPrintf(PETSC_COMM_WORLD, "\nn1 = %d\nn2 = %d\nns = %g\ndphi = %g\n", (int)user.n1, (int)user.n2, (double)user.ns, (double)user.dphi);
            /* allocate and fill theta_n and s_n */
            PetscInt Nm = user.n2 - user.n1 + 1;
            PetscMalloc1(Nm,&user.theta);
            PetscMalloc1(Nm,&user.ssign);

            for (PetscInt i=0;i<Nm;++i) user.ssign[i] = -1;
            
            PetscMPIInt rank; MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
            if (rank == 0) {
                PetscRandom rng;
                PetscRandomCreate(PETSC_COMM_SELF,&rng);
                PetscRandomSetSeed(rng,(unsigned long)noise_seed);
                PetscRandomSetFromOptions(rng);
                for (PetscInt i=0;i<Nm;++i){ PetscReal r; PetscRandomGetValueReal(rng,&r); user.theta[i]=2.0*PETSC_PI*r; }
                PetscRandomDestroy(&rng);
            }
            /* broadcast to everyone */
            MPI_Bcast(user.theta, Nm, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(user.ssign, Nm, MPIU_INT,  0, PETSC_COMM_WORLD);
        }

        if(user.IC == 4)
        {
            PetscPrintf(PETSC_COMM_WORLD, "\nr = %g\nsigma = %d\ndphi = %g\n", (double)user.pw_r, (int)user.pw_sigma, (double)user.pw_dphi);
        }

        if(user.IC == 5)
        {
            PetscPrintf(PETSC_COMM_WORLD, "\nsigma = %g\ndphi = %g\nr = %g\ncarrier = %g\n", (double)user.osc_sigma, (double)user.osc_dphi, (double)user.osc_r, (double)user.osc_k0);
        }
        

        /* ------------ solve ------------ */
    
        PetscPrintf(PETSC_COMM_WORLD, "\n===================\nLam22 IC %d run!!!\n===================\n\nUsing nx = %d, nt = %d\nmphi2 = %g, mchi2 = %g\nA = %g\nkL/2pi = %g\nlam22 = %g\nGhost = %g\n", (int)user.IC, (int)user.nx, (int)user.nt, user.mphi2, user.mchi2, user.A, (double)user.C, user.lam22,user.ghost);
        PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nt0 = %g, tF = %g\n", user.xL, user.xR, user.t0, user.tF);
        PetscPrintf(PETSC_COMM_WORLD, "hx = %.8e, ht = %.8e\n\n", (double)user.hx, (double)user.ht);

        PetscPrintf(PETSC_COMM_WORLD,
        "=== Solve (T=[%.3f,%.3f], nt=%d) ===\n", (double)user.t0,(double)user.tF,user.nt);

    }

    /* ------------ fill initial conditions on rank-local part and solve ------------ */
    PetscCall(FillInitialConditions(dm,  U,  &user));
    SNESSolve(snes,NULL,U);
	
    
    /* (optional) if converged, solve for the next time slice */
	SNESConvergedReason reason0;
	SNESGetConvergedReason(snes,&reason0);

	Vec      Ugood = NULL;               /* will hold copy of last convergent solution */
	PetscInt Ugood_n = -1;               /* cached global size of Ugood for quick layout check */
	DM dm_good = NULL;                   /* DM matching Ugood layout */


	PetscReal tF       = user.tF;      /* start from CLI tF */
	PetscReal tF_step  = 1.0;          /* grow tF each loop */
	PetscInt  maxloops = 200;          /* safety cap */
	PetscInt  last_good_nt = -1;
	PetscReal last_good_tF = -1.0;
	PetscBool found_failure = PETSC_FALSE;

	

	if (reason0 > 0) {
        last_good_tF = tF;
        last_good_nt = user.nt;
        PetscCall(SaveUgood(U,&Ugood,&Ugood_n));              /* keep initial converged solution */
        /* keep a reference to dm so it outlives this block */
        dm_good = dm;
        PetscCall(PetscObjectReference((PetscObject)dm_good));
        
	} else if (reason0 < 0) {
        PetscPrintf(PETSC_COMM_WORLD,
            "\n>>> first tF to fail is: %.15g (SNES reason %d) <<<\n",
            (double)tF,(int)reason0);
        found_failure = PETSC_TRUE;
	}

	SNESDestroy(&snes); VecDestroy(&U); MatDestroy(&J); DMDestroy(&dm);

	/* If the very first solve failed, report the last convergent state (if any) and stop */
	if (found_failure) {
        if (last_good_tF >= 0.0)
            PetscPrintf(PETSC_COMM_WORLD,
            "Last convergent solution was found at tF = %.15g (nt = %d).\n",
            (double)last_good_tF, (int)last_good_nt);
        else
            PetscPrintf(PETSC_COMM_WORLD,
            "No convergent solution was found before failure.\n");
	}

    /* --------------------------------------------------------------------
    unless loop was turned off in CLI, increase tF by 1 and run again 
    -----------------------------------------------------------------------*/
    if (!found_failure && user.Loop) {
        PetscCall(ExtendInTimeUntilFailure(
            &user,
            tF_step,
            maxloops,
            &Ugood,
            &dm_good,
            &last_good_tF,
            &last_good_nt,
            &found_failure
        ));
    }


	/* ---------------------------------------------------------------------
	Dump energies and field snapshots to a MATLAB/Octave script
	--------------------------------------------------------------------- */
	PetscInt save = 1;
    if (save) {
    PetscCall(DumpSolutionAndEnergies(&user, dm_good, Ugood, last_good_nt, last_good_tF));
    }


  /* ------------ clean up ------------ */
  if (Ugood) VecDestroy(&Ugood);
  if (dm_good) DMDestroy(&dm_good);
  if(user.IC == 3){PetscFree(user.theta); PetscFree(user.ssign);}
  PetscFinalize();
  return 0;
}
