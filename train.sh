#!/bin/sh

# bash script for training on Taurus

module purge
module load Python/3.7.2-GCCcore-8.2.0
module load ASE/3.18.1-foss-2019a-Python-3.7.2
module load PyYAML
module load TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4

rm train.out train.err
nohup python train.py > train.out 2> train.err &
