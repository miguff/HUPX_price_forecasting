#!/bin/bash
#SBATCH --job-name=optuna_catboost
#SBATCH --output=logs/out_catboost_%A_%a.txt
#SBATCH --error=logs/err_catboost_%A_%a.txt
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


cd $SLURM_SUBMIT_DIR

module load cray-python
source venv/bin/activate

python runs_paralel.py --model catboost --country HR