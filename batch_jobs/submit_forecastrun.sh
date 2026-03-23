#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=200G
#SBATCH --time=10:05:00

module load conda
module load cray-fftw/3.3.10.6
module load cray-mpich-abi
conda activate cGForecasts_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


for f in 1
#!$(seq 0 99)
do
	echo "Running forecast for file: python CubicGalileonForecasts.py ./ini_files/config_run_ell3000_k0p3.yaml"
    start_time=$(date +%s)
	srun -n 64 python CubicGalileonForecasts.py ./ini_files/config_run_ell3000_k0p3.yaml
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "Time taken for srun: $runtime seconds"
done

