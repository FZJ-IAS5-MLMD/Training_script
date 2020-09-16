#!/bin/sh

#SBATCH --time=00:10:00
#SBATCH --job-name=trainprof
#SBATCH --output=trainprof.%A.out
#SBATCH --error=trainprof.%A.out

##SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
##SBATCH --mem-per-cpu=1GB  # On ML partition, memory per CPU is fixed.

#SBATCH --partition ml

CODE_DIR_PATH="/scratch/ws/1/hpclab11-gpuH_1"

# Set up environment.
module load modenv/ml
module load matplotlib/3.0.3-fosscuda-2018b-Python-3.6.6
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4 Horovod/0.19.5-fosscuda-2019b-TensorFlow-2.2.0-Python-3.7.4 PyYAML/5.1.2-GCCcore-8.3.0

# NVIDIA Nsight modules.
#module use --append /sw/taurus/tools/nvidia/modulefiles
#module add nsys/2020.3.1.72-ppc64le

# No need to install PiNN if this was installed in this environment before.
#cd ${CODE_DIR_PATH}/PiNN
#pip install --user -e .

# Run training.
cd ${CODE_DIR_PATH}/Training_script

#python train.py
#nsys profile -f true -o acetone.profile -t cuda,mpi --stats=true python train.py
#nsys profile -f true -o acetone.profile -t cuda,nvtx --stats=true python train.py

srun python train.py

