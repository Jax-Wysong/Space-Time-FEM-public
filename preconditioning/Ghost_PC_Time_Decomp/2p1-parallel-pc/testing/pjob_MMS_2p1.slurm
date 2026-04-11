#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=FS-LU         # job name  
#SBATCH --partition=compute       # queue/partition
#SBATCH --nodes=1                 # total nodes          
#SBATCH --ntasks-per-node=4      # MPI ranks per node   
#SBATCH --time=00-08:00:00         # wall-clock limit
#SBATCH --mem=20G
#SBATCH --output=FS-LU/MMS_2p1_FS-LU_15x15x16_test_4r_ov0.out    # stdout+stderr file (%j = job-id)
#SBATCH --error=FS-LU/error_files/MMS_2p1_FS-LU_15x15x16_test_4r_ov0.err      # optional separate stderr

# needed for multi-node runs on innovator

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0

# Convenience: total MPI ranks Slurm just gave us
NTASKS=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))

# PETSc executable (already built with DMDA support)
EXE=.././ghost_MMS_2p1_PC

###--------------- LU FS Command line ------------------- #
srun --mpi=pmix_v3 -n ${NTASKS} \
  ${EXE} \
  -dm_view \
  -snes_type newtonls -snes_linesearch_type bt \
  -snes_max_it 10000 \
  -ksp_converged_reason \
  -snes_atol 1e-4 \
  -snes_monitor \
  -snes_converged_reason \
  -xL -1.0 -xR 1.0 \
  -yL -1.0 -yR 1.0 \
  -t0 0.0 -tF 2.0 \
  -lam 1.0 \
  -nx 15 \
  -ny 15 \
  -nt 16 \
  -overlap 0 \
  -ksp_type gmres -snes_ksp_ew \
  -ksp_gmres_restart 200 \
  -memory_view \
  -log_view \
  -pc_type fieldsplit \
  -pc_fieldsplit_0_fields 0,1 \
  -pc_fieldsplit_1_fields 2,3 \
  -pc_fieldsplit_type additive \
  -fieldsplit_0_ksp_type preonly \
  -fieldsplit_0_pc_type lu \
  -fieldsplit_0_pc_factor_mat_solver_type mumps \
  -fieldsplit_1_ksp_type preonly \
  -fieldsplit_1_pc_type lu \
  -fieldsplit_1_pc_factor_mat_solver_type mumps \
  -memory_view \
  -log_view :FS-LU/PC_2p1_FS-LU_15x15x16_4r_ov0_log-view.txt 

# # --------------- FS - RAS Command line ------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 -da_processors_y 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_atol 1e-4 \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_ksp_ew \
#   -ksp_type gmres -ksp_gmres_restart 200 \
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
#   -xL -1.0 -xR 1.0 \
#   -yL -1.0 -yR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -lam 1.0 \
#   -nx 63 \
#   -ny 63 \
#   -nt 64 \
#   -Nsub 4 \
#   -overlap 1 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 1 \
#   -slab_pc_nonlinear 0 \
#   -memory_view \
#   -log_view :FS-RAS/PC_2p1_FS-RAS_63x63x64_4r_log-view.txt 

# # --------------- RAS Command line ------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 -da_processors_y 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_atol 1e-4 \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_ksp_ew \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -xL -1.0 -xR 1.0 \
#   -yL -1.0 -yR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -lam 1.0 \
#   -nx 63 \
#   -ny 63 \
#   -nt 64 \
#   -Nsub 4 \
#   -overlap 1 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 1 \
#   -slab_pc_nonlinear 0 \
#   -memory_view \
#   -log_view :RAS/PC_2p1_RAS_63x63x64_4r_log-view.txt 

# # --------------- LU Command line ------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 -da_processors_y 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_rtol 1e-5 \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_ksp_ew \
#   -ksp_type preonly \
#   -pc_type lu -pc_factor_mat_solver_type mumps \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -xL -1.0 -xR 1.0 \
#   -yL -1.0 -yR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -lam 1.0 \
#   -nx 31 \
#   -ny 31 \
#   -nt 32 \
#   -Nsub 4 \
#   -overlap 1 \
#   -use_shell_pc 0 \
#   -slab_pc_ras 0 \
#   -slab_pc_nonlinear 0 \
#   -memory_view \
#   -log_view :LU/PC_2p1_LU_31x31x32_4r_ov1_log-view.txt 

# --------------- ASM Command line ------------------- #
# srun --mpi=pmix_v3 -n ${NTASKS} \
#   ${EXE} \
#   -da_processors_x 1 -da_processors_y 1 \
#   -dm_view \
#   -snes_type newtonls -snes_linesearch_type bt \
#   -snes_rtol 1e-5 \
#   -snes_max_it 10000 \
#   -snes_monitor \
#   -snes_converged_reason \
#   -snes_ksp_ew \
#   -ksp_type gmres -ksp_gmres_restart 200 \
#   -ksp_rtol 1e-3 \
#   -ksp_atol 1e-12 \
#   -ksp_error_if_not_converged false \
#   -ksp_pc_side right -ksp_norm_type unpreconditioned \
#   -ksp_monitor_true_residual -ksp_converged_reason \
#   -xL -1.0 -xR 1.0 \
#   -yL -1.0 -yR 1.0 \
#   -t0 0.0 -tF 2.0 \
#   -lam 1.0 \
#   -nx 15 \
#   -ny 15 \
#   -nt 16 \
#   -Nsub 4 \
#   -overlap 1 \
#   -use_shell_pc 1 \
#   -slab_pc_ras 0 \
#   -slab_pc_nonlinear 0 \
#   -memory_view \
#   -log_view :quick-test/MMS_2p1_ASM_15x15x16_4r_ov1_silenceErr2_log-view.txt 


