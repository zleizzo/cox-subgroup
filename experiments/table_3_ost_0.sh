#!/bin/bash
#SBATCH --job-name=table_3_ost
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-599
#SBATCH --output=/home/ml/zach/survival-subgroup-refactored/results/slurm_logs/table_3_ost_%A_%a.out
#SBATCH --error=/home/ml/zach/survival-subgroup-refactored/results/slurm_logs/table_3_ost_%A_%a.err

OFFSET=0

# Print this sub-job's task ID
echo "My experiment number is " $((SLURM_ARRAY_TASK_ID + OFFSET))

conda init
conda activate survival
python run_slurm.py table_3_ost $((SLURM_ARRAY_TASK_ID + OFFSET))
