#!/bin/bash
#  ---------- Slurm directives ----------
#SBATCH --job-name=Char_IC_Sweep
#SBATCH --partition=compute,quickq,bigmem
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=00-00:10:00
#SBATCH --mem=10G
#SBATCH --output=char-sweep-spike/sweep_nx200_nt200-ov1.out
#SBATCH --error=char-sweep-spike/error_files/sweep_nx200_nt200-ov1.err

NX=200
NT=200
OV=1
IW=1

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_coll_hcoll_enable=0

EXE=.././Advection_1p1_PC

COMMON="-da_processors_x 1 \
  -snes_type ksponly \
  -ksp_type gmres -ksp_gmres_restart 200 \
  -use_shell_pc 1 \
  -slab_pc_ras 1 \
  -ksp_atol 1e-6 \
  -ksp_rtol 1e-100 \
  -ksp_converged_reason \
  -sub_start_pc_factor_mat_solver_type mumps \
  -sub_mid_pc_factor_mat_solver_type mumps \
  -sub_end_pc_factor_mat_solver_type mumps \
  -IC 1 \
  -xL 0.0 -xR 1.0 \
  -t0 0.0 -tF 1.0 \
  -nx ${NX} -nt ${NT} \
  -speed 1.0 \
  -overlap ${OV} \
  -interface_width ${IW} \
  -save 0 \
  -dump_matrices 0"

echo ""
echo ""
echo ""
echo "========================================================"
echo " Characteristic IC Sweep: nx=$NX nt=$NT"
echo "========================================================"
echo ""

for NSUB in 1 2 4 8 10 20 40; do

  echo ""
  echo "-------- Nsub=$NSUB --------"

  echo "--- Baseline (no char features) ---"
  srun --mpi=pmix_v3 -n ${NSUB} \
    ${EXE} ${COMMON} -Nsub ${NSUB} \
    -use_char_ic 0 \
    -log_view :char-sweep-spike/log-files/char_IC-0/nx${NX}_nt${NT}-ov${OV}-Nsub-${NSUB}-halfD1-log-view.txt

  echo ""
  echo ""
  echo "--- char_ic (baseline char) ---"
  srun --mpi=pmix_v3 -n ${NSUB} \
    ${EXE} ${COMMON} -Nsub ${NSUB} \
    -use_char_ic 1 \
    -log_view :char-sweep-spike/log-files/char_IC-1/nx${NX}_nt${NT}-ov${OV}-Nsub-${NSUB}-halfD2-log-view.txt

done
