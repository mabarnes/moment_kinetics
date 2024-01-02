#!/usr/bin/env bash

#SBATCH --ntasks=48
#SBATCH --time=1:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=skl_fua_dbg
#SBATCH --output=PRECOMPILEDIRslurm-%j.out

set -e

cd $SLURM_SUBMIT_DIR

# Get setup for Julia
source julia.env

echo "precompiling $(date)"

bin/julia --project -O3 --check-bounds=no precompile.jl

echo "finished! $(date)"
