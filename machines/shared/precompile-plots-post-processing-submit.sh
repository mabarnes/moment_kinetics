#!/usr/bin/env bash

set -e

# Parse command line options
ONLY_JOB_ID=1
while getopts "hj" opt; do
  case $opt in
    h)
      echo "Submit job to precompile moment kinetics
-h             Print help and exit"
      exit 1
      ;;
    j)
      ONLY_JOB_ID=0
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
POSTPROCESSINGJOBSCRIPT=${PRECOMPILEDIR}precompile-plots-post-processing.job
sed -e "s|ACCOUNT|$ACCOUNT|" -e "s|PRECOMPILEDIR|$PRECOMPILEDIR|" machines/$MACHINE/jobscript-precompile-plots-post-processing.template > $POSTPROCESSINGJOBSCRIPT

JOBID=$(sbatch --parsable $POSTPROCESSINGJOBSCRIPT)
if [[ "$ONLY_JOB_ID" -eq 1 ]]; then
  echo "Precompile plots_post_processing: $JOBID"
else
  echo "$JOBID"
fi
echo "In the queue" > $PRECOMPILEDIR/slurm-$JOBID.out

if [[ "$ONLY_JOB_ID" -eq 1 ]]; then
  echo "Done"
fi
