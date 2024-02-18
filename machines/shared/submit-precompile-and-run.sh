#!/usr/bin/env bash

set -e

POSTPROC=0
MAKIEPOSTPROCESS=1
while getopts "haf:g:m:n:p:oq:st:u:w" opt; do
  case $opt in
    h)
      echo "Precompile the system image(s) and then run a simulation. Arguments are passed through to submit-run.sh:"
      echo
      ./submit-run.sh -h
      exit 1
      ;;
    a)
      POSTPROC=1
      ;;
    o)
      MAKIEPOSTPROCESS=0
      ;;
  esac
done

source julia.env

JOBINFO=($(util/get-precompile-info.jl))
MACHINE=${JOBINFO[0]}
ACCOUNT=${JOBINFO[1]}
MAKIE_AVAILABLE=${JOBINFO[2]}
PLOTS_AVAILABLE=${JOBINFO[3]}

if [[ "$MAKIE_AVAILABLE" == "n" && "$PLOTS_AVAILABLE" == "y" ]]; then
  # No Makie post-processing available, so always use Plots post-processing
  MAKIEPOSTPROCESS=0
fi

MK_SYSIMAGE_JOB_ID=$(./precompile-submit.sh -o -j)
echo "Precompile moment_kinetics: $MK_SYSIMAGE_JOB_ID"

POSTPROC_ARGUMENT=""
if [[ $POSTPROC -eq 0 && ("$MAKIE_AVAILABLE" == "y" || "$PLOTS_AVAILABLE" == "y") ]]; then
  # Create a submission script for post-processing
  if [[ MAKIEPOSTPROCESS -eq 1 ]]; then
    POSTPROC_SYSIMAGE_JOB_ID=$(./precompile-makie-post-processing-submit.sh -j)
    echo "Precompile makie_post_processing: $POSTPROC_SYSIMAGE_JOB_ID"
  else
    POSTPROC_SYSIMAGE_JOB_ID=$(./precompile-plots-post-processing-submit.sh -j)
    echo "Precompile plots_post_processing: $POSTPROC_SYSIMAGE_JOB_ID"
  fi

  POSTPROC_ARGUMENT="-g $POSTPROC_SYSIMAGE_JOB_ID"
fi

./submit-run.sh -f $MK_SYSIMAGE_JOB_ID $POSTPROC_ARGUMENT -w $@
