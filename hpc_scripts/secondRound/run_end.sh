#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=end
#SBATCH --output=hpclogs/adv_end_%j.out
#SBATCH --error=hpclogs/adv_end_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cw1681@nyu.edu

cd MatchLSTM-PyTorch/
source activate matchlstm
python train.py -name end_only -d advdata/advdata_end_only -h5 adv_end.h5 -r -e 12

