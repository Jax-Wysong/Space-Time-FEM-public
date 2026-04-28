#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=time         # job name  ← change if you wish
#SBATCH --partition=compute,quickq       # queue/partition
#SBATCH --nodes=1                 # total nodes          ← pick what you need
#SBATCH --ntasks-per-node=20     # MPI ranks per node   ← match cores/socket
#SBATCH --time=00-00:10:00         # wall-clock limit
#SBATCH --mem=10G
#SBATCH --output=Thesis-Results/pollution_percent_nx-200/p~1%/Wave_PC_1p1_200x3800_RAS_20ranks.out    # stdout+stderr file (%j = job-id)
#SBATCH --error=Thesis-Results/pollution_percent_nx-200/p~1%/error_files/Wave_PC_1p1_200x3800_RAS_20ranks.err      # optional separate stderr

# needed for multi-node runs on innovator

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0


# Convenience: total MPI ranks Slurm just gave us
NTASKS=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))
#   -ksp_monitor_true_residual \
    #-ksp_converged_reason \
    #  -pc_type shell \
#valgrind --tool=memcheck --leak-check=full --track-origins=yes \

# PETSc executable (already built with DMDA support)
EXE=.././wave_1p1_PC

# --------- RAS command line options ---------- #
srun --mpi=pmix_v3 -n ${NTASKS} \
  ${EXE} \
  -da_processors_x 1 \
  -dm_view \
  -snes_type ksponly \
  -ksp_type gmres -ksp_gmres_restart 200 \
  -use_shell_pc 1 \
  -slab_pc_ras 1 \
  -ksp_atol 1e-4 \
  -ksp_rtol 1e-50 \
  -ksp_view_final_residual \
  -ksp_converged_reason \
  -IC 1 \
  -xL -5.0 -xR 5.0 \
  -t0 0.0 -tF 10.0 \
  -nx 200 -nt 3800 \
  -Nsub 20 \
  -overlap 1 \
  -interface_width 1 \
  -dump_matrices 0 \
  -memory_view \
  -log_view :Thesis-Results/pollution_percent_nx-200/p~1%/Wave_PC_1p1_200x3800_RAS_20ranks_log-view.txt

# ###------------- LU command line options --------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type ksponly \
#   -ksp_type preonly \
#   -pc_type lu \
#   -pc_factor_mat_solver_type mumps \
#   -use_shell_pc 0 \
#   -slab_pc_ras 0 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -ksp_atol 1e-04 \
#   -IC 1 \
#   -xL -5.0 -xR 5.0 \
#   -t0 0.0 -tF 10.0 \
#   -nx 100 -nt 100 \
#   -overlap 0 \
#   -interface_width 0 \
#   -memory_view \
#   -log_view :../analysis/testing-testing/Wave_PC_1p1_100x100_LU_4ranks_log-view.txt 


# # # ------------- LU with FS command line options -------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type ksponly \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -pc_type fieldsplit \
#   -pc_fieldsplit_type schur \
#   -pc_fieldsplit_schur_fact_type full \
#   -pc_fieldsplit_schur_precondition self \
#   -fieldsplit_u_ksp_type preonly \
#   -fieldsplit_u_pc_type lu \
#   -fieldsplit_u_pc_factor_mat_solver_type mumps \
#   -fieldsplit_v_ksp_type preonly \
#   -fieldsplit_v_pc_type lu \
#   -fieldsplit_v_pc_factor_mat_solver_type mumps \
#   -use_shell_pc 0 \
#   -slab_pc_nonlinear 0 \
#   -slab_pc_ras 0 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -interface_BC_all 1 \
#   -IC 1 \
#   -xL -5.0 -xR 5.0 \
#   -t0 0.0 -tF 10.0 \
#   -nx 60 -nt 60 \
#   -memory_view \
#   -log_view :FS-LU/PC_1p1_60x60_FS-LU_4ranks_log-view.txt 


# ---------------- ASM command line options ---------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type ksponly \
#   -ksp_type gmres -ksp_gmres_restart 50 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 0 \
#   -slab_pc_nonlinear 0 \
#   -interface_BC_all 1 \
#   -interface_BC_dirichlet 0 \
#   -interface_BC_neumann 0 \
#   -interface_BC_none 0 \
#   -interface_D_N_alternate 0 \
#   -interface_BC_robin 0 \
#   -robin_alpha 1.0 \
#   -ksp_rtol 1e-50 \
#   -ksp_atol 1e-50 \
#   -ksp_max_it 500 \
#   -ksp_view_final_residual \
#   -ksp_converged_reason \
#   -IC 1 \
#   -xL -5.0 -xR 5.0 \
#   -t0 0.0 -tF 10.0 \
#   -nx 60 -nt 60 \
#   -Nsub 4 \
#   -overlap 1 \
#   -memory_view \
#   -log_view :ASM/PC_1p1_60x60_ASM_4ranks_500its-50rs_log-view.txt



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
#   -interface_BC_all 0 \
#   -interface_BC_dirichlet 1 \
#   -interface_BC_neumann 0 \
#   -interface_BC_none 0 \
#   -interface_D_N_alternate 0 \
#   -ksp_rtol 1e-3 \
#   -ksp_atol 1e-12 \
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
#   -nx 199 -nt 200 \
#   -Nsub 4 \
#   -overlap 1 \
#   -memory_view \
#   -log_view :BC-quick-test/PC_1p1_199x200_ASM_FS_dirichlet_4ranks_log-view.txt


# RAS with FS command line options #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_monitor \
#   -snes_rtol 1e-5 \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -use_shell_pc 1 -slab_pc_ras 1 -slab_pc_nonlinear 0 \
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
#   -nx 199 -nt 200 \
#   -Nsub 4 \
#   -overlap 1 \
#   -memory_view \
#   -log_view :quick-test/MMS_1p1_199x200_RAS_FS_4ranks_log-view.txt










