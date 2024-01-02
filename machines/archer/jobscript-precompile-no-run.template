#!/usr/bin/env bash

#SBATCH --mem=64G
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --output=PRECOMPILEDIRslurm-%j.out

set -e

cd $SLURM_SUBMIT_DIR

# Get setup for Julia
source julia.env

# Workaround as cpus-per-task no longer inherited by srun from sbatch.
# See https://docs.archer2.ac.uk/faq/upgrade-2023/
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "precompiling $(date)"

bin/julia --project -O3 --check-bounds=no precompile-no-run.jl

echo "finished! $(date)"
