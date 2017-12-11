#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=run_original
#SBATCH --output=hpclogs/run_original_1_%j.out
#SBATCH --error=hpclogs/run_original_1_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd MatchLSTM-PyTorch/
source activate matchlstm
python train.py -name original -d tokenized_squad_v1.1.2 -h5 original.h5

