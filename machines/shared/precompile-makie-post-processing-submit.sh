#!/usr/bin/env bash

set -e

# Parse command line options
while getopts "h" opt; do
  case $opt in
    h)
      echo "Submit job to precompile moment kinetics
-h             Print help and exit"
      exit 1
      ;;
  esac
done

# job settings for the run
##########################

# Get setup for Julia
source julia.env

JOBINFO=($(util/get-precompile-info.jl))
MACHINE=${JOBINFO[0]}
ACCOUNT=${JOBINFO[1]}

PRECOMPILEDIR=precompile-temp/
mkdir -p $PRECOMPILEDIR

# Create a submission script for the post-processing precompilation
POSTPROCESSINGJOBSCRIPT=${PRECOMPILEDIR}precompile-makie-post-processing.job
sed -e "s|ACCOUNT|$ACCOUNT|" -e "s|PRECOMPILEDIR|$PRECOMPILEDIR|" machines/$MACHINE/jobscript-precompile-makie-post-processing.template > $POSTPROCESSINGJOBSCRIPT

JOBID=$(sbatch --parsable $POSTPROCESSINGJOBSCRIPT)
echo "Precompile makie_post_processing: $JOBID"
echo "In the queue" > $PRECOMPILEDIR/slurm-$JOBID.out

echo "Done"
