#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=Heat_linear        # job name  ← change if you wish
#SBATCH --partition=compute       # queue/partition
#SBATCH --nodes=1                 # total nodes          ← pick what you need
#SBATCH --ntasks-per-node=1      # MPI ranks per node   ← match cores/socket
#SBATCH --time=00-00:10:00         # wall-clock limit
#SBATCH --mem=20G
#SBATCH --output=Heat_1p1_100x100_test_.out    # stdout+stderr file (%j = job-id)
#SBATCH --error=Heat_1p1_100x100_test_%j.err      # optional separate stderr

# needed for multi-node runs on innovator

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0


# Convenience: total MPI ranks Slurm just gave us
NTASKS=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))

# PETSc executable (already built with DMDA support)
EXE=.././heat_linear

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
       -xL 0.0 -xR 1.0 \
       -t0 0.0 -tF 1.0 \
       -nx 100 -nt 100 \
       -memory_view \
       -log_view \
       -ksp_type preonly \
       -pc_type lu \
       -factor_mat_solver_type mumps
