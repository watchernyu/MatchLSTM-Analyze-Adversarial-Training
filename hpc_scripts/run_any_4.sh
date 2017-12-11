#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=any_4
#SBATCH --output=hpclogs/adv_any_4_%j.out
#SBATCH --error=hpclogs/adv_any_4_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd MatchLSTM-PyTorch/
source activate matchlstm
python train.py -name any_4 -d advdata/advdata_any_4 -h5 adv_any_4.h5

