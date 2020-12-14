#!/bin/bash -l
#SBATCH --job-name=DeepSphere_climate_equi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=climate-equi-normal-cpu-%j.log
#SBATCH --error=climate-equi-normal-cpu-%j-e.log

module load daint-gpu
module load cray-python/3.6.5.1
module load TensorFlow/1.11.0-CrayGNU-18.08-cuda-9.1-python3
module load PyExtensions/3.6.5.1-CrayGNU-18.08
module load h5py/2.8.0-CrayGNU-18.08-python3-serial

source ~/venv-3.6/bin/activate

cd $SCRATCH/PDMdeepsphere/Experiments/Climate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -n 1 -u python run_experiment_equiangular.py
echo -e "$SLURM_JOB_NAME finished on $(date)\n"
