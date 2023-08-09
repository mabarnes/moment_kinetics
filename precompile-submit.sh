#!/bin/bash

set -e

# job settings for the run
##########################

# Get setup for Julia
source julia.env

echo "Submitting precompile job..."

JOBINFO=($(util/get-precompile-info.jl))
MACHINE=${JOBINFO[0]}
ACCOUNT=${JOBINFO[1]}

PRECOMPILEDIR=precompile-temp/
mkdir -p $PRECOMPILEDIR

# Create a submission script for the run
JOBSCRIPT=${PRECOMPILEDIR}precompile.job
sed -e "s|ACCOUNT|$ACCOUNT|" -e "s|PRECOMPILEDIR|$PRECOMPILEDIR|" machines/$MACHINE/jobscript-precompile.template > $JOBSCRIPT

JOBID=$(sbatch --parsable $JOBSCRIPT)
echo "Precompile: $JOBID"
echo "In the queue" > $PRECOMPILEDIR/slurm-$JOBID.out

echo "Done"
