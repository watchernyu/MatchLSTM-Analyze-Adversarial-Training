#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=any_3
#SBATCH --output=hpclogs/adv_any_3_%j.out
#SBATCH --error=hpclogs/adv_any_3_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd MatchLSTM-PyTorch/
source activate matchlstm
python train.py -name any_3 -d advdata/advdata_any_3 -h5 adv_any_3.h5

