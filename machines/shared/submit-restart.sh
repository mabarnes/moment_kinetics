#!/bin/bash

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

# Parse command line options
# [See e.g. https://www.stackchief.com/tutorials/Bash%20Tutorial%3A%20getopts
# for examples of how to use Bash's getopts]
POSTPROC=0
SUBMIT=0
RESTARTFROM=""
FOLLOWFROM=""
MAKIEPOSTPROCESS=1
while getopts "haf:m:n:p:oq:r:st:u:" opt; do
  case $opt in
    h)
      echo "Submit jobs for a simulation (using INPUT_FILE for input) and post-processing to the queue
Usage: submit-run.sh [option] INPUT_FILE
-h             Print help and exit
-a             Do not submit post-processing job after the run
-f JOBID       Make this job start after JOBID finishes successfully
-m MEM         The requested memory for post-processing
-n NODES       The number of nodes to use for the simulation
-o             Use original post_processing, instead of makie_post_processing, for the post-processing job
-p PARTITION   The 'partition' (passed to 'sbatch --partition')
-q QOS         The 'quality of service' (passed to 'sbatch --qos')
-r FILE        The output file to restart from (defaults to latest output in the run directory)
-s             Only create submission scripts, do not actually submit jobs
-t TIME        The run time, e.g. 24:00:00
-u TIME        The run time for the post-processing, e.g. 1:00:00"
      exit 1
      ;;
    a)
      POSTPROC=1
      ;;
    f)
      FOLLOWFROM="-d afterok:$OPTARG"
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
    r)
      RESTARTFROM=$OPTARG
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
  esac
done

# Get the positional argument as INPUTFILE
# [See https://stackoverflow.com/a/13400237]
INPUTFILE=${@:$OPTIND:1}
if [ -z "$INPUTFILE" ]; then
  echo "No input file passed - must pass an input file name as a positional argument"
  exit 1
fi

RUNNAME=$(util/get-run-name.jl $INPUTFILE)

RUNDIR=runs/$RUNNAME/
mkdir -p $RUNDIR

if [[ $POSTPROC -eq 0 ]]; then
  echo "Submitting $INPUTFILE for restart from '$RESTARTFROM' and post-processing..."
else
  echo "Submitting $INPUTFILE for restart from '$RESTARTFROM'..."
fi

# Create a submission script for the run
RESTARTJOBSCRIPT=${RUNDIR}$RUNNAME-restart.job
sed -e "s|NODES|$NODES|" -e "s|RUNTIME|$RUNTIME|" -e "s|ACCOUNT|$ACCOUNT|" -e "s|PARTITION|$PARTITION|" -e "s|QOS|$QOS|" -e "s|RUNDIR|$RUNDIR|" -e "s|INPUTFILE|$INPUTFILE|" -e "s|RESTARTFROM|$RESTARTFROM|" machines/$MACHINE/jobscript-restart.template > $RESTARTJOBSCRIPT

if [[ $SUBMIT -eq 0 ]]; then
  JOBID=$(sbatch $FOLLOWFROM --parsable $RESTARTJOBSCRIPT)
  echo "Restart: $JOBID"
  echo "In the queue" > ${RUNDIR}slurm-$JOBID.out
fi

if [[ $POSTPROC -eq 0 ]]; then
  # Create a submission script for post-processing
  POSTPROCJOBSCRIPT=${RUNDIR}$RUNNAME-post.job
  if [[ MAKIEPOSTPROCESS -eq 1 ]]; then
    POSTPROCESSTEMPLATE=jobscript-postprocess.template
  else
    POSTPROCESSTEMPLATE=jobscript-postprocess-plotsjl.template
  fi
  sed -e "s|POSTPROCMEMORY|$POSTPROCMEMORY|" -e "s|POSTPROCTIME|$POSTPROCTIME|" -e "s|ACCOUNT|$ACCOUNT|" -e "s|RUNDIR|$RUNDIR|" machines/$MACHINE/$POSTPROCESSTEMPLATE > $POSTPROCJOBSCRIPT

  if [[ $SUBMIT -eq 0 ]]; then
    POSTID=$(sbatch -d afterany:$JOBID --parsable $POSTPROCJOBSCRIPT)
    echo "Postprocess: $POSTID"
    echo "In the queue" > ${RUNDIR}slurm-post-$POSTID.out
  fi
fi

echo "Done"
