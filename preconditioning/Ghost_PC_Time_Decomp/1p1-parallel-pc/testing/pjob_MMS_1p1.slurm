#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=FS-LU         # job name  ← change if you wish
#SBATCH --partition=compute       # queue/partition
#SBATCH --nodes=1                 # total nodes          ← pick what you need
#SBATCH --ntasks-per-node=10      # MPI ranks per node   ← match cores/socket
#SBATCH --time=00-00:30:00         # wall-clock limit
#SBATCH --mem=40G
#SBATCH --output=Thesis-Results/FS-RAS-nsub-vary/MMS_1p1_199x200_FS-RAS_10ranks.out    # stdout+stderr file (%j = job-id)
#SBATCH --error=Thesis-Results/FS-RAS-nsub-vary/error_files/MMS_1p1_199x200_FS-RAS_10ranks.err      # optional separate stderr

# needed for multi-node runs on innovator

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0


# Convenience: total MPI ranks Slurm just gave us
NTASKS=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))
#   -ksp_monitor_true_residual \
    #-ksp_converged_reason \
    #  -pc_type shell \
#valgrind --tool=memcheck --leak-check=full --track-origins=yes \

# PETSc executable 
EXE=.././ghost_MMS_1p1_PC

# -------------- RAS with FS command line options ---------------------- #
srun --mpi=pmix_v3 -n ${NTASKS} \
  ${EXE} \
  -da_processors_x 1 \
  -dm_view \
  -snes_type newtonls -snes_linesearch_type bt \
  -snes_monitor \
  -snes_atol 1e-6 \
  -ksp_type gmres -ksp_gmres_restart 200 \
  -use_shell_pc 1 -slab_pc_ras 1 -slab_pc_nonlinear 0 \
  -snes_ksp_ew \
  -ksp_monitor_true_residual -ksp_converged_reason \
  -sub_start_pc_type fieldsplit \
  -sub_start_pc_fieldsplit_type additive \
  -sub_start_pc_fieldsplit_0_fields 0,1 \
  -sub_start_pc_fieldsplit_1_fields 2,3 \
  -sub_start_fieldsplit_0_ksp_type preonly \
  -sub_start_fieldsplit_0_pc_type lu \
  -sub_start_fieldsplit_1_ksp_type preonly \
  -sub_start_fieldsplit_1_pc_type lu \
  -sub_mid_pc_type fieldsplit \
  -sub_mid_pc_fieldsplit_type additive \
  -sub_mid_pc_fieldsplit_0_fields 0,1 \
  -sub_mid_pc_fieldsplit_1_fields 2,3 \
  -sub_mid_fieldsplit_0_ksp_type preonly \
  -sub_mid_fieldsplit_0_pc_type lu \
  -sub_mid_fieldsplit_1_ksp_type preonly \
  -sub_mid_fieldsplit_1_pc_type lu \
  -sub_end_pc_type fieldsplit \
  -sub_end_pc_fieldsplit_type additive \
  -sub_end_pc_fieldsplit_0_fields 0,1 \
  -sub_end_pc_fieldsplit_1_fields 2,3 \
  -sub_end_fieldsplit_0_ksp_type preonly \
  -sub_end_fieldsplit_0_pc_type lu \
  -sub_end_fieldsplit_1_ksp_type preonly \
  -sub_end_fieldsplit_1_pc_type lu \
  -IC 1 \
  -xL -1.0 -xR 1.0 \
  -t0 0.0 -tF 2.0 \
  -mphi2 1.0 -mchi2 1.0 \
  -Ghost -1.0 \
  -A 1.0 \
  -lam22 1.0 \
  -nx 199 -nt 200 \
  -Nsub 10 \
  -overlap 1 \
  -memory_view \
  -log_view :Thesis-Results/FS-RAS-nsub-vary/PC_1p1_199x200_FS-RAS_10ranks_log-view.txt

# # ------------- LU with FS (Additive) command line options -------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_ksp_ew \
#   -snes_monitor \
#   -snes_atol 1e-7 \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -pc_type fieldsplit \
#   -pc_fieldsplit_0_fields 0,1 \
#   -pc_fieldsplit_1_fields 2,3 \
#   -pc_fieldsplit_type additive \
#   -fieldsplit_0_ksp_type preonly \
#   -fieldsplit_0_pc_type lu \
#   -fieldsplit_0_pc_factor_mat_solver_type mumps \
#   -fieldsplit_1_ksp_type preonly \
#   -fieldsplit_1_pc_type lu \
#   -fieldsplit_1_pc_factor_mat_solver_type mumps \
#   -use_shell_pc 0 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -IC 1 \
#   -xL -1.0 -xR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -mphi2 1.0 -mchi2 1.0 \
#   -Ghost -1.0 \
#   -A 1.0 \
#   -lam22 1.0 \
#   -nx 1599 -nt 1600 \
#   -overlap 0 \
#   -memory_view \
#   -log_view :Thesis-Results/FS-LU-OFFICIAL/PC_1p1_1599x1600_FS-LU_4ranks_log-view.txt


# # ------------- LU command line options --------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_atol 1e-8 \
#   -ksp_type preonly \
#   -pc_type lu \
#   -pc_factor_mat_solver_type mumps \
#   -use_shell_pc 0 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -IC 1 \
#   -xL -1.0 -xR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -mphi2 1.0 -mchi2 1.0 \
#   -Ghost -1.0 \
#   -A 1.0 \
#   -lam22 1.0 \
#   -nx 399 -nt 400 \
#   -overlap 0 \
#   -memory_view \
#   -log_view :Thesis-Results/LU-mem-per-proc/PC_399x400_LU_32ranks_log-view.txt 


# # --------- RAS command line options ---------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_atol 1e-6 \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 1 \
#   -slab_pc_nonlinear 0 \
#   -snes_ksp_ew \
#   -ksp_max_it 100000 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -IC 1 \
#   -xL -1.0 -xR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -mphi2 1.0 -mchi2 1.0 \
#   -Ghost -1.0 \
#   -A 1.0 \
#   -lam22 1.0 \
#   -nx 199 -nt 200 \
#   -Nsub 20 \
#   -overlap 1 \
#   -memory_view \
#   -log_view :RAS-IC-0-nsub-vary/PC_1p1_199x200_RAS_20ranks_log-view.txt 



# ---------------- ASM command line options ---------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_rtol 5e-5 \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 0 \
#   -slab_pc_nonlinear 0 \
#   -snes_ksp_ew \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -IC 1 \
#   -xL -1.0 -xR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -mphi2 1.0 -mchi2 1.0 \
#   -Ghost -1.0 \
#   -A 1.0 \
#   -lam22 1.0 \
#   -nx 799 -nt 800 \
#   -Nsub 4 \
#   -overlap 1 \
#   -memory_view \
#   -log_view :ASM-nonlin-compare/PC_1p1_799x800_ASM_4ranks_log-view.txt


# ---------------- ASM FS command line options ---------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_monitor \
#   -snes_rtol 1e-5 \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 0 \
#   -slab_pc_nonlinear 0 \
#   -ksp_rtol 1e-3 \
#   -ksp_atol 1e-12 \
#   -snes_ksp_ew \
#   -ksp_error_if_not_converged false \
#   -ksp_pc_side right -ksp_norm_type unpreconditioned \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -sub_start_pc_type fieldsplit \
#   -sub_start_pc_fieldsplit_type additive \
#   -sub_start_pc_fieldsplit_0_fields 0,1 \
#   -sub_start_pc_fieldsplit_1_fields 2,3 \
#   -sub_start_fieldsplit_0_ksp_type preonly \
#   -sub_start_fieldsplit_0_pc_type lu \
#   -sub_start_fieldsplit_1_ksp_type preonly \
#   -sub_start_fieldsplit_1_pc_type lu \
#   -sub_mid_pc_type fieldsplit \
#   -sub_mid_pc_fieldsplit_type additive \
#   -sub_mid_pc_fieldsplit_0_fields 0,1 \
#   -sub_mid_pc_fieldsplit_1_fields 2,3 \
#   -sub_mid_fieldsplit_0_ksp_type preonly \
#   -sub_mid_fieldsplit_0_pc_type lu \
#   -sub_mid_fieldsplit_1_ksp_type preonly \
#   -sub_mid_fieldsplit_1_pc_type lu \
#   -sub_end_pc_type fieldsplit \
#   -sub_end_pc_fieldsplit_type additive \
#   -sub_end_pc_fieldsplit_0_fields 0,1 \
#   -sub_end_pc_fieldsplit_1_fields 2,3 \
#   -sub_end_fieldsplit_0_ksp_type preonly \
#   -sub_end_fieldsplit_0_pc_type lu \
#   -sub_end_fieldsplit_1_ksp_type preonly \
#   -sub_end_fieldsplit_1_pc_type lu \
#   -IC 1 \
#   -xL -1.0 -xR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -mphi2 1.0 -mchi2 1.0 \
#   -Ghost -1.0 \
#   -A 1.0 \
#   -lam22 1.0 \
#   -nx 1599 -nt 1600 \
#   -Nsub 10 \
#   -overlap 1 \
#   -memory_view \
#   -log_view :comparison-LU-ASM-RAS/10rank_e-5/MMS_1p1_1599x1600_ASM_FS_10ranks_log-view.txt

# ------------- LU command line options --------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_monitor \
#   -snes_rtol 1e-8 \
#   -ksp_type preonly \
#   -pc_type lu \
#   -pc_factor_mat_solver_type mumps \
#   -use_shell_pc 0 -slab_pc_nonlinear 0 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -IC 1 \
#   -xL -1.0 -xR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -mphi2 1.0 -mchi2 1.0 \
#   -Ghost -1.0 \
#   -A 1.0 \
#   -lam22 1.0 \
#   -nx 99 -nt 100 \
#   -memory_view \
#   -log_view :LU/MMS_1p1_99x100_LU_4ranks_log-view.txt 


