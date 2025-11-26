#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=MMS         # job name  ← change if you wish
#SBATCH --partition=compute       # queue/partition
#SBATCH --nodes=1                 # total nodes          ← pick what you need
#SBATCH --ntasks-per-node=1      # MPI ranks per node   ← match cores/socket
#SBATCH --time=00-00:15:00         # wall-clock limit
#SBATCH --mem=15G
#SBATCH --output=MMS_1p1_100x101_refac_test.out    # stdout+stderr file (%j = job-id)
#SBATCH --error=MMS_1p1_100x101_refac_test_%j.err      # optional separate stderr

# needed for multi-node runs on innovator

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0


# Convenience: total MPI ranks Slurm just gave us
NTASKS=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))

# PETSc executable (already built with DMDA support)
EXE=.././ghost_MMS_1p1

# ---- run ----
    srun --mpi=pmix_v3 -n ${NTASKS} \
         ${EXE} \
         -dm_view \
       -snes_type newtonls -snes_linesearch_type basic \
       -snes_max_it 10000 \
       -ksp_converged_reason \
       -snes_rtol 1e-8 -snes_atol 1e-12 \
       -snes_monitor \
       -snes_converged_reason \
       -IC 1 \
       -xL -1.0 -xR 1.0 \
       -t0 0.0 -tF 2.0 \
       -mphi2 1.0 -mchi2 1.0 \
       -Ghost -1.0 \
       -A 1.0 \
       -lam22 1.0 \
       -nx 100 -nt 101 \
       -ksp_type fgmres -snes_ksp_ew -ksp_max_it 500 \
       -ksp_gmres_restart 50 \
       -memory_view \
       -log_view \
       -pc_type fieldsplit \
       -pc_fieldsplit_0_fields 2,3 \
       -pc_fieldsplit_1_fields 0,1 \
       -pc_fieldsplit_type schur \
       -pc_fieldsplit_schur_precondition a11 \
       -pc_fieldsplit_schur_fact_type upper \
       -fieldsplit_0_ksp_type preonly \
       -fieldsplit_0_pc_type lu \
       -fieldsplit_0_pc_factor_mat_solver_type mumps \
       -fieldsplit_0_pc_factor_shift_type NONZERO \
       -fieldsplit_0_pc_factor_shift_amount 1e-12 \
       -fieldsplit_1_ksp_type preonly \
       -fieldsplit_1_pc_type lu \
       -fieldsplit_1_pc_factor_mat_solver_type mumps \
       -fieldsplit_1_pc_factor_shift_type NONZERO \
       -fieldsplit_1_pc_factor_shift_amount 1e-12 \
