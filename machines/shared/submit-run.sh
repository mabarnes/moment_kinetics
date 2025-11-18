#!/usr/bin/env bash

set -e

# job settings for the run
##########################

# Get setup for Julia
source julia.env

JOBINFO=($(util/get-job-info.jl))
MACHINE=${JOBINFO[0]}
ACCOUNT=${JOBINFO[1]}
RUNTIME=${JOBINFO[2]}
NODES=${JOBINFO[3]}
POSTPROCTIME=${JOBINFO[4]}
POSTPROCMEMORY=${JOBINFO[5]}
PARTITION=${JOBINFO[6]}
QOS=${JOBINFO[7]}
MAKIE_AVAILABLE=${JOBINFO[8]}
PLOTS_AVAILABLE=${JOBINFO[9]}

# Parse command line options
# [See e.g. https://www.stackchief.com/tutorials/Bash%20Tutorial%3A%20getopts
# for examples of how to use Bash's getopts]
POSTPROC=0
SUBMIT=0
FOLLOWFROM=""
POSTPROC_FOLLOWFROM=""
MAKIEPOSTPROCESS=1
WARN_OLD_SYSIMAGE=0
PERFORMANCE_RUN=1
while getopts "haf:g:m:n:p:oq:st:u:wx" opt; do
  case $opt in
    h)
      echo "Submit jobs for a simulation (using INPUT_FILE for input) and post-processing to the queue
Usage: submit-run.sh [option] INPUT_FILE
-h             Print help and exit
-a             Do not submit post-processing job after the run
-f JOBID       Make this job start after JOBID finishes successfully
-g JOBID       Make the post processing job start after JOBID finishes successfully (as well as after the simulation run finishes)
-m MEM         The requested memory for post-processing
-n NODES       The number of nodes to use for the simulation
-o             Use original post_processing, instead of makie_post_processing, for the post-processing job when both are available
-p PARTITION   The 'partition' (passed to 'sbatch --partition')
-q QOS         The 'quality of service' (passed to 'sbatch --qos')
-s             Only create submission scripts, do not actually submit jobs
-t TIME        The run time, e.g. 24:00:00
-u TIME        The run time for the post-processing, e.g. 1:00:00
-w             Suppress the warning given when system image(s) are older than source code files
-x             Run a 'performance test' using performance-tests/run_for_performance_test.jl instead run_moment_kinetics.jl."
      exit 1
      ;;
    a)
      POSTPROC=1
      ;;
    f)
      FOLLOWFROM="-d afterok:$OPTARG"
      ;;
    g)
      POSTPROC_FOLLOWFROM=",afterok:$OPTARG"
      ;;
    m)
      POSTPROCMEM=$OPTARG
      ;;
    n)
      NODES=$OPTARG
      ;;
    o)
      MAKIEPOSTPROCESS=0
      ;;
    p)
      PARTITION=$OPTARG
      ;;
    q)
      QOS=$OPTARG
      ;;
    s)
      SUBMIT=1
      ;;
    t)
      RUNTIME=$OPTARG
      ;;
    u)
      POSTPROCTIME=$OPTARG
      ;;
    w)
      WARN_OLD_SYSIMAGE=1
      ;;
    x)
      PERFORMANCE_RUN=0
      ;;
  esac
done

if [[ "$MAKIE_AVAILABLE" == "n" && "$PLOTS_AVAILABLE" == "y" ]]; then
  # No Makie post-processing available, so always use Plots post-processing
  MAKIEPOSTPROCESS=0
fi

# Get the positional argument as INPUTFILE
# [See https://stackoverflow.com/a/13400237]
INPUTFILE=${@:$OPTIND:1}
if [ -z "$INPUTFILE" ]; then
  echo "No input file passed - must pass an input file name as a positional argument"
  exit 1
fi

if [[ $POSTPROC -eq 0 ]]; then
  if [[ $SUBMIT -eq 0 ]]; then
    echo "Submitting $INPUTFILE for run and post-processing..."
  else
    echo "Creating submission script from $INPUTFILE for run and post-processing..."
  fi
else
  if [[ $SUBMIT -eq 0 ]]; then
    echo "Submitting $INPUTFILE for run..."
  else
    echo "Creating submission script from $INPUTFILE for run..."
  fi
fi

RUNNAME=$(util/get-run-name.jl $INPUTFILE)

RUNDIR=runs/$RUNNAME/
mkdir -p $RUNDIR

if [[ $PERFORMANCE_RUN -eq 0 ]]; then
  RUNSCRIPT=performance-tests/run_for_performance_test.jl
else
  RUNSCRIPT=run_moment_kinetics.jl
fi

# Create a submission script for the run
RUNJOBSCRIPT=${RUNDIR}$RUNNAME.job
sed -e "s|NODES|$NODES|" -e "s|RUNTIME|$RUNTIME|" -e "s|ACCOUNT|$ACCOUNT|" -e "s|PARTITION|$PARTITION|" -e "s|QOS|$QOS|" -e "s|RUNDIR|$RUNDIR|" -e "s|INPUTFILE|$INPUTFILE|" -e "s|RUNSCRIPT|$RUNSCRIPT|" machines/$MACHINE/jobscript-run.template > $RUNJOBSCRIPT

if [[ "$WARN_OLD_SYSIMAGE" -eq 0 ]]; then
  # Check that source code has not been changed since moment_kinetics.so was created
  bin/julia --project -O3 moment_kinetics/src/check_so_newer_than_code.jl moment_kinetics.so
fi

if [[ $SUBMIT -eq 0 ]]; then
  JOBID=$(sbatch $FOLLOWFROM --parsable $RUNJOBSCRIPT)
  echo "Run: $JOBID"
  echo "In the queue" > ${RUNDIR}slurm-$JOBID.out
fi

if [[ $POSTPROC -eq 0 && "$MAKIE_AVAILABLE" == "n" && "$PLOTS_AVAILABLE" == "n" ]]; then
  echo "No post-processing packages available, so no post-processing job submitted"
elif [[ $POSTPROC -eq 0 ]]; then
  # Create a submission script for post-processing
  POSTPROCJOBSCRIPT=${RUNDIR}$RUNNAME-post.job
  if [[ MAKIEPOSTPROCESS -eq 1 ]]; then
    POSTPROCESSTEMPLATE=jobscript-postprocess.template

    if [[ "$WARN_OLD_SYSIMAGE" -eq 0 ]]; then
      # Check that source code has not been changed since makie_postproc.so was created
      bin/julia --project=makie_post_processing -O3 moment_kinetics/src/check_so_newer_than_code.jl makie_postproc.so
    fi
  else
    POSTPROCESSTEMPLATE=jobscript-postprocess-plotsjl.template

    if [[ "$WARN_OLD_SYSIMAGE" -eq 0 ]]; then
      # Check that source code has not been changed since plots_postproc.so was created
      bin/julia --project=plots_post_processing -O3 moment_kinetics/src/check_so_newer_than_code.jl plots_postproc.so
    fi
  fi
  sed -e "s|POSTPROCMEMORY|$POSTPROCMEMORY|" -e "s|POSTPROCTIME|$POSTPROCTIME|" -e "s|ACCOUNT|$ACCOUNT|" -e "s|RUNDIR|$RUNDIR|" machines/$MACHINE/$POSTPROCESSTEMPLATE > $POSTPROCJOBSCRIPT

  if [[ $SUBMIT -eq 0 ]]; then
    POSTID=$(sbatch -d afterany:$JOBID$POSTPROC_FOLLOWFROM --parsable $POSTPROCJOBSCRIPT)
    echo "Postprocess: $POSTID"
    echo "In the queue" > ${RUNDIR}slurm-post-$POSTID.out
  fi
fi

echo "Done"
