#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=small_test_2
#SBATCH --output=hpclogs/small_test_2_%j.out
#SBATCH --error=hpclogs/small_test_2_%j.err
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --mem=12GB

cd MatchLSTM-PyTorch/
source activate matchlstm
python train.py -name small_test_2 -t

