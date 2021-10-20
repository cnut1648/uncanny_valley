#!/bin/bash
#SBATCH --nodelist=ink-nova
#SBATCH --gres=gpu:8000:1
#SBATCH --qos=general-8000
#SBATCH -n 1
#SBATCH --export=ALL,LD_LIBRARY_PATH='/usr/local/cuda-10.1/lib64/'
#SBATCH -o outs/slurm/LF::%j.out
# CUDA_VISIBLE_DEVICES=0
source activate valley;

python lm.py model='longformer-base'
# # 25
# python fine_tune_clf.py -m model='longformer-base' train.lr=0.00003 train.label_smooth=0.01
# # 32
# python fine_tune_clf.py -m model='roberta-large' train.lr=0.00003 train.label_smooth=0

# tune
# python fine_tune_clf.py -m model='longformer-base' train.lr='choice(4E-5,1E-6,8E-6,1E-4)' \
#     train.wd='choice(0.02,0.001)'

# python fine_tune_clf.py -m model='roberta-large' train.lr='choice(4E-5,1E-6,8E-6,1E-4)' \
#     train.wd='choice(0.02,0.001)'
# python fine_tune_clf.py -m model='roberta-large' train.lr='choice(1E-5,3E-5,6E-5)' train.label_smooth='choice(0.1,0)'