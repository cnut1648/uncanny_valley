#!/bin/bash
#SBATCH --gres=gpu:6000:1
#SBATCH -t 720
#SBATCH -n 1
#SBATCH --export=ALL,LD_LIBRARY_PATH='/usr/local/cuda-10.1/lib64/'
#SBATCH -o logs/slurm/fine-tune::%j.log
# CUDA_VISIBLE_DEVICES=0
source activate valley;

# echo "one vs one InfoCE"
# python run.py datamodule=one_vs_one


# rank
# echo "fine tune from rank loss"
# python run.py \
#     datamodule=fine_tune \
#     module=fine_tune_model \
#     module/model@module='roberta-large' \
#     module.arch_ckpt='/home/jiashu/uncanny_valley/motif/lightning/outs/ckpt/171/LM'

# echo "fine tune from info loss"
# python run.py \
#     datamodule=fine_tune \
#     module=fine_tune_model \
#     module/model@module='roberta-large' \
#     module.arch_ckpt='/home/jiashu/uncanny_valley/motif/lightning/outs/ckpt/170/LM'


# roberta
python run.py \
    datamodule=fine_tune \
    module=fine_tune_model \
    module/model@module='roberta-large' \
    module.arch_ckpt='/home/jiashu/uncanny_valley/motif/lightning/outs/ckpt/170/LM' \
    module.optim.lr=0.00003 \
    module.optim.weight_decay=0.02 \
    trainer.max_epochs=20

####################################################################################################
# lr search fine tune
####################################################################################################
# echo "lr search for fine tune from info loss"
# python run.py -m \
#     experiment=fine_tune \
#     hparams_search=lr \
#     module/model@module='roberta-large' \
#     module.arch_ckpt='/home/jiashu/uncanny_valley/motif/lightning/logs/ckpt/170/LM'


# echo "fine tune from rank loss"
# python run.py -m \
#     experiment=fine_tune \
#     hparams_search=lr \
#     module/model@module='roberta-large' \
#     module.arch_ckpt='/home/jiashu/uncanny_valley/motif/lightning/logs/ckpt/171/LM'
