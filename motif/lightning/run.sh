#!/bin/bash
#SBATCH --nodelist=ink-ruby
#SBATCH --gres=gpu:8000:1
#SBATCH -t 720
#SBATCH -n 1
#SBATCH --export=ALL,LD_LIBRARY_PATH='/usr/local/cuda-10.1/lib64/'
#SBATCH -o outs/slurm/LF::%j.out
# CUDA_VISIBLE_DEVICES=0
source activate valley;

# echo "one vs one InfoCE"
# python run.py datamodule=one_vs_one

echo "one vs one Rank Loss"
python run.py datamodule=one_vs_one module.criterion=rankloss