#include "extend_in_time.h"
#include "dm_create.h"
#include "fill_IC.h"
#include "saveSol.h"

PetscErrorCode ExtendInTimeUntilFailure(
    AppCtx      *user,
    PetscReal    tF_step,
    PetscInt     maxloops,
    Vec         *Ugood_out,
    DM          *dm_good_out,
    PetscReal   *last_good_tF_out,
    PetscInt    *last_good_nt_out,
    PetscBool   *found_failure_out
)
{
    PetscFunctionBeginUser;

    /* Initialize outputs */
    *Ugood_out          = NULL;
    *dm_good_out        = NULL;
    *last_good_tF_out   = -1.0;
    *last_good_nt_out   = -1;
    *found_failure_out  = PETSC_FALSE;

    Vec  Ugood = NULL;
    DM   dm_good = NULL;
    PetscInt Ugood_n = -1;

    for (PetscInt it = 0; it < maxloops; it++) {

        /* Increase tF and recompute nt */
        user->tF += tF_step;
        PetscInt nt = (PetscInt)(user->tF * user->nx) + 1;

        /* Create DM/SNES/J/U for this grid */
        DM   dm2;
        SNES snes2;
        Mat  J2;
        Vec  U2;

        PetscCall(CreateDMSNESForGrid(user, user->nx, nt,
                                      &dm2, &snes2, &J2, &U2));

        /* Fill ICs */
        PetscCall(FillInitialConditions(dm2, U2, user));

        PetscPrintf(PETSC_COMM_WORLD, "\n\n==========================================================\n\n");
        PetscPrintf(PETSC_COMM_WORLD, "\nLam22 IC %d run!!!!!!!\nUsing nx = %d, nt = %d\nmphi2 = %g, mchi2 = %g\nA = %g\nkL/2pi = %g\nlam22 = %g\n", (int)user->IC, (int)user->nx, (int)user->nt, user->mphi2, user->mchi2, user->A, (double)user->C, user->lam22);
        PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nt0 = %g, tF = %g\n", user->xL, user->xR, user->t0, user->tF);
        PetscPrintf(PETSC_COMM_WORLD, "hx = %.8e, ht = %.8e\n\n", (double)user->hx, (double)user->ht);

        PetscPrintf(PETSC_COMM_WORLD,
        "=== Solve (T=[%.3f,%.3f], nt=%d) ===\n", (double)user->t0,(double)user->tF,user->nt);

        /* Solve */
        SNESSolve(snes2, NULL, U2);

        SNESConvergedReason r;
        PetscCall(SNESGetConvergedReason(snes2, &r));

        if (r > 0) {
            /* Converged, update last-good solution */
            *last_good_tF_out = user->tF;
            *last_good_nt_out = nt;

            PetscCall(SaveUgood(U2, &Ugood, &Ugood_n));

            if (dm_good) { DMDestroy(&dm_good); dm_good = NULL; }
            dm_good = dm2;
            PetscCall(PetscObjectReference((PetscObject)dm_good));

            PetscPrintf(PETSC_COMM_WORLD, "\n\n==========================================================\n\n");
			PetscPrintf(PETSC_COMM_WORLD,"  converged (reason=%d). Growing tF -> %.6f\n",(int)r,(double)(user->tF + tF_step));


            /* Destroy per-iteration objects except dm_good */
            PetscCall(SNESDestroy(&snes2));
            PetscCall(VecDestroy(&U2));
            PetscCall(MatDestroy(&J2));
            /* DONT destroy dm2 (it is now dm_good) */
            continue;
        }

        else if (r < 0) {
            /* Failure */
            *found_failure_out = PETSC_TRUE;

            PetscPrintf(PETSC_COMM_WORLD,
                "\n>>> First failure at tF = %.15g (reason=%d) <<<\n",
                (double)user->tF, (int)r);

            /* Clean objects */
            PetscCall(SNESDestroy(&snes2));
            PetscCall(VecDestroy(&U2));
            PetscCall(MatDestroy(&J2));
            PetscCall(DMDestroy(&dm2));
            break;
        }

        else {
            PetscPrintf(PETSC_COMM_WORLD,
                "Unexpected SNES reason 0, stopping.\n");

            PetscCall(SNESDestroy(&snes2));
            PetscCall(VecDestroy(&U2));
            PetscCall(MatDestroy(&J2));
            PetscCall(DMDestroy(&dm2));
            break;
        }
    }

    *Ugood_out   = Ugood;
    *dm_good_out = dm_good;

    PetscFunctionReturn(0);
}
