#!/bin/bash
#SBATCH --job-name=optuna_lightgbm
#SBATCH --output=logs/out_lgbm_%A_%a.txt
#SBATCH --error=logs/err_lgbm_%A_%a.txt
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-16

cd $SLURM_SUBMIT_DIR

module load cray-python
source venv/bin/activate

python runs_paralel.py --model lightgbm --country HR