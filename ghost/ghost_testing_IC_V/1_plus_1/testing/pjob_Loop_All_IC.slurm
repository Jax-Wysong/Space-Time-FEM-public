#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=IC5         # job name  ← change if you wish
#SBATCH --partition=compute       # queue/partition
#SBATCH --nodes=1                 # total nodes          ← pick what you need
#SBATCH --ntasks-per-node=4      # MPI ranks per node   ← match cores/socket
#SBATCH --time=00-01:00:00         # wall-clock limit
#SBATCH --mem=20G
#SBATCH --output=lam22_IC_5_A1_sig005_r1_dphi0_k00.out    # stdout+stderr file (%j = job-id)
#SBATCH --error=lam22_IC_5_A1_sig005_r1_dphi0_k00_%j.err      # optional separate stderr


export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0


# Convenience: total MPI ranks Slurm just gave us
NTASKS=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))

# PETSc executable (already built with DMDA support)
EXE=.././ghost_test_IC_V
# ---- run ----
# Use srun (preferred) or mpirun; srun automatically inherits Slurm resources.
#srun --mpi=pmix_v3 -n ${NTASKS} \
    srun --mpi=pmix_v3 -n ${NTASKS} \
         ${EXE} \
         -dm_view \
       -snes_type newtonls -snes_linesearch_type basic \
       -snes_max_it 10000 \
       -ksp_converged_reason \
       -snes_rtol 1e-8 -snes_atol 1e-12 \
       -snes_monitor \
       -snes_converged_reason \
       -Loop 1 \
       -IC 5 \
       -noise_n2 64 \
       -noise_ns -1.0 \
       -pw_sigma -1 \
       -pw_r 1.0 \
       -pw_dphi 3.1415926535897932384626433832795028841971 \
       -osc_sigma 0.05 \
       -osc_r 1.0 \
       -osc_dphi 0.0 \
       -osc_k0 0.0 \
       -xL 0.0 -xR 1.0 \
       -t0 0.0 -tF 22.0 \
       -mphi2 1.0 -mchi2 1.0 \
       -C 1.0 \
       -Ghost -1.0 \
       -A 1.0 \
       -lam22 1.0 \
       -nx 100 -nt 2201 \
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
       -fieldsplit_1_pc_factor_shift_amount 1e-12
