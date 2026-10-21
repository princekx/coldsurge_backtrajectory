#!/bin/bash
#SBATCH --job-name=name_backtraj
#SBATCH --output=logs/name_backtraj_%j.out
#SBATCH --error=logs/name_backtraj_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=2

# Usage (from this folder): sbatch run_backtraj.sh --dates-file cs_onsets.txt --members 4
module load scitools/default-current
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
python run_backtraj.py "$@"
