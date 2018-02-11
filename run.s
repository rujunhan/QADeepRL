#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=50:59:00
#SBATCH --mem=20GB
##SBATCH --gres=gpu:1
##SBATCH --output=test.out
##SBATCH --error=test.err

module purge
module load pytorch/python3.5/0.2.0_3

python3 -u BiDAF_rep.py
