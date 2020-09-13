#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --partition=ml
#SBATCH --gres=gpu:4
#SBATCH -A p_hack20-2 
#SBATCH --job-name=acetone_train

module load modenv/ml
ml Horovod/0.19.5-fosscuda-2019b-TensorFlow-2.2.0-Python-3.7.4 PyYAML
srun python train.py 
