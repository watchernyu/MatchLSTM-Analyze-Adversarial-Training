#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=front
#SBATCH --output=hpclogs/adv_front_%j.out
#SBATCH --error=hpclogs/adv_front_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd MatchLSTM-PyTorch/
source activate matchlstm
python train.py -name front_only -d advdata/advdata_front_only -h5 adv_front.h5

