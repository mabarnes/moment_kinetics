#!/bin/bash

set -e

# Parse command line options
NO_PRECOMPILE_RUN=1
while getopts "hn" opt; do
  case $opt in
    h)
      echo "Submit job to precompile moment kinetics
-h             Print help and exit
-n             No 'precompile run' - i.e. call precompile-no-run.jl instead of precompile.jl"
      exit 1
      ;;
    n)
      NO_PRECOMPILE_RUN=0
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

# Create a submission script for the precompilation
JOBSCRIPT=${PRECOMPILEDIR}precompile.job
if [[ NO_PRECOMPILE_RUN -eq 0 ]]; then
  echo "Submitting precompile job (no precompile run)..."
  TEMPLATE_NAME=jobscript-precompile-no-run.template
else
  echo "Submitting precompile job..."
  TEMPLATE_NAME=jobscript-precompile.template
fi
sed -e "s|ACCOUNT|$ACCOUNT|" -e "s|PRECOMPILEDIR|$PRECOMPILEDIR|" machines/$MACHINE/$TEMPLATE_NAME > $JOBSCRIPT

JOBID=$(sbatch --parsable $JOBSCRIPT)
echo "Precompile: $JOBID"
echo "In the queue" > $PRECOMPILEDIR/slurm-$JOBID.out

echo "Done"
