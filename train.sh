#!/bin/sh

# bash script for training on Taurus

module load Python/3.6.6-foss-2019a
module load SciPy-bundle/2019.03-foss-2019a
module load ASE/3.16.2-intel-2018a-Python-3.6.4
module load PyYAML
module load TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4

rm train.out train.err
nohup python train.py > train.out 2> train.err &
