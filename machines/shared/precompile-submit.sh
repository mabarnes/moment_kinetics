#!/usr/bin/env bash

set -e

# Parse command line options
ONLY_JOB_ID=1
NO_PRECOMPILE_RUN=1
NO_POSTPROC=1
while getopts "hjno" opt; do
  case $opt in
    h)
      echo "Submit job to precompile moment kinetics
-h             Print help and exit
-j             Only print job id (so it can be captured easily by another script)
-n             No 'precompile run' - i.e. call precompile-no-run.jl instead of precompile.jl
-o             Only compile moment_kinetics.so, skipping any available post-processing packages"
      exit 1
      ;;
    j)
      ONLY_JOB_ID=0
      ;;
    n)
      NO_PRECOMPILE_RUN=0
      ;;
    o)
      NO_POSTPROC=0
  esac
done

# job settings for the run
##########################

# Get setup for Julia
source julia.env

JOBINFO=($(util/get-precompile-info.jl))
MACHINE=${JOBINFO[0]}
ACCOUNT=${JOBINFO[1]}
MAKIE_AVAILABLE=${JOBINFO[2]}
PLOTS_AVAILABLE=${JOBINFO[3]}

PRECOMPILEDIR=precompile-temp/
mkdir -p $PRECOMPILEDIR

# Create a submission script for the precompilation
JOBSCRIPT=${PRECOMPILEDIR}precompile.job
if [[ NO_PRECOMPILE_RUN -eq 0 ]]; then
  if [[ "$ONLY_JOB_ID" -eq 1 ]]; then
    echo "Submitting precompile job (no precompile run)..."
  fi
  TEMPLATE_NAME=jobscript-precompile-no-run.template
else
  if [[ "$ONLY_JOB_ID" -eq 1 ]]; then
    echo "Submitting precompile job..."
  fi
  TEMPLATE_NAME=jobscript-precompile.template
fi
sed -e "s|ACCOUNT|$ACCOUNT|" -e "s|PRECOMPILEDIR|$PRECOMPILEDIR|" machines/$MACHINE/$TEMPLATE_NAME > $JOBSCRIPT

JOBID=$(sbatch --parsable $JOBSCRIPT)
if [[ "$ONLY_JOB_ID" -eq 1 ]]; then
  echo "Precompile: $JOBID"
else
  echo "$JOBID"
fi
echo "In the queue" > $PRECOMPILEDIR/slurm-$JOBID.out

if [[ "$NO_POSTPROC" -eq 1 && "$MAKIE_AVAILABLE" == "y" ]]; then
  ./precompile-makie-post-processing-submit.sh
fi
if [[ "$NO_POSTPROC" -eq 1 && "$PLOTS_AVAILABLE" == "y" ]]; then
  ./precompile-plots-post-processing-submit.sh
fi

if [[ "$ONLY_JOB_ID" -eq 1 ]]; then
  echo "Done"
fi
