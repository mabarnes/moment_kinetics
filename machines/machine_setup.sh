#!/usr/bin/env bash

set -e

# Parse command line options
# [See e.g. https://www.stackchief.com/tutorials/Bash%20Tutorial%3A%20getopts
# for examples of how to use Bash's getopts]
while getopts "h" opt; do
  case $opt in
    h)
      echo "Setup moment_kinetics to run on a known machine"
      echo "Usage: machine_setup.sh [/path/to/julia]"
      echo
      echo "If the path to julia is not given, you will be prompted for it."
      exit 1
      ;;
  esac
done

# Get the positional argument as INPUTFILE
# [See https://stackoverflow.com/a/13400237]
JULIA=${@:$OPTIND:1}

# Make sure $JULIA is set
# Note [ -z "$VAR" ] tests if $VAR is empty. Need the quotes to ensure that the
# contents of $VAR are not evaluated if it is not empty.
if [ -z "$JULIA" ]; then
  # Try to find a sensible default if there is one in the $PATH
  if [ $(command -v julia) ]; then
    JULIA=$(command -v julia)
  fi
  echo "Please enter the path to the Julia executable to be used ['$JULIA']:"
  # Use '-e' option to get path auto-completion
  read -e -p "> " input
  if [ ! -z "$input" ]; then
    JULIA=$input
  fi
fi

# Make sure $JULIA is actually an executable. If not, prompt the user to try
# again.
# Note 'command -v foo' returns success if foo is an executable and failure
# otherwise, except that if no argument is passed 'command -v' returns success,
# so we also need to check that $JULIA is not empty.
while [[ -z "$JULIA" || ! $(command -v $JULIA) ]]; do
  echo
  echo "Error: '$JULIA' is not an executable."
  echo
  echo "Please enter the path to the Julia executable to be used:"
  # Use '-e' option to get path auto-completion
  read -e -p "> "  JULIA
done
# Convert $JULIA (which might be a relative path) to an absolute path.
# '-m' to avoid erroring if the directory does not exist already.
# '-s' to skip resolving symlinks (if we did resolve symlinks, it might make
#      the path look different than expected).
JULIA=$(realpath -m -s $JULIA)
echo
echo "Using Julia at $JULIA"
echo

while true; do
  echo "Enter name of the machine to set up:"
  read -p "> "  MACHINE
  echo

  # Get default values for this machine from the machine_setup.jl script
  # Note: the brackets() around the command execution turn the result into an
  # 'array' that we can get the separate elements from below.
  # This command will fail (and print a helpful error message) if $MACHINE is
  # not a known value, so `break` only when it succeeds.
  DEFAULTS=($($JULIA machines/shared/machine_setup.jl --defaults $MACHINE)) && break
done

echo "Setting up for '$MACHINE'"
echo

# Note that the $(eval echo <thing>)) is needed to remove quotes around
# arguments. Adding the quotes in the Julia script is necessary so that if an
# argument is empty it is not lost when parsing the Julia script output into
# $DEFAULTS. Note $DEFAULTS is a Bash array.
DEFAULT_RUN_TIME=$(eval echo ${DEFAULTS[0]})
DEFAULT_NODES=$(eval echo ${DEFAULTS[1]})
DEFAULT_POSTPROC_TIME=$(eval echo ${DEFAULTS[2]})
DEFAULT_POSTPROC_MEMORY=$(eval echo ${DEFAULTS[3]})
DEFAULT_PARTITION=$(eval echo ${DEFAULTS[4]})
DEFAULT_QOS=$(eval echo ${DEFAULTS[5]})

# Get the account to submit jobs with
echo "Enter the account code used to submit jobs []:"
read -p "> "  ACCOUNT
echo
echo "Account code used is $ACCOUNT"
echo

# Get the location for the .julia directory, in case this has to have a
# non-default value, e.g. because the user's home directory is not accessible
# from compute nodes.
# Use $JULIA_DEPOT_PATH as the default if it has already been set
JULIA_DIRECTORY=$JULIA_DEPOT_PATH
echo "Enter location that should be used for the .julia directory (this can be empty"
echo "if the default location is OK) [$JULIA_DIRECTORY]:"
# Use '-e' option to get path auto-completion
read -e -p "> "  input
if [ ! -z "$input" ]; then
  JULIA_DIRECTORY=$input
fi
# Convert input (which might be a relative path) to an absolute path.
# '-m' to avoid erroring if the directory does not exist already.
# '-s' to skip resolving symlinks (if we did resolve symlinks, it might make
#      the path look different than expected).
JULIA_DIRECTORY=$(realpath -m -s $JULIA_DIRECTORY)
echo
echo "Using julia_directory=$JULIA_DIRECTORY"
echo

# Get the setting for the default run time
echo "Enter the default value for the time limit for simulation jobs [$DEFAULT_RUN_TIME]:"
read -p "> "  input
if [ ! -z "$input" ]; then
  DEFAULT_RUN_TIME=$input
fi
echo
echo "Default simulation time limit is $DEFAULT_RUN_TIME"
echo

# Get the setting for the default number of nodes
echo "Enter the default value for the number of nodes for a run [$DEFAULT_NODES]:"
read -p "> "  input
if [ ! -z "$input" ]; then
  DEFAULT_NODES=$input
fi
echo
echo "Default number of nodes is $DEFAULT_NODES"
echo

# Get the setting for the default postproc time
echo "Enter the default value for the time limit for post-processing jobs [$DEFAULT_POSTPROC_TIME]:"
read -p "> "  input
if [ ! -z "$input" ]; then
  DEFAULT_POSTPROC_TIME=$input
fi
echo
echo "Default post-processing time limit is $DEFAULT_POSTPROC_TIME"
echo

# Get the setting for the default postproc memory
echo "Enter the default value for the memory requested for post-processing jobs [$DEFAULT_POSTPROC_MEMORY]:"
read -p "> "  input
if [ ! -z "$input" ]; then
  DEFAULT_POSTPROC_MEMORY=$input
fi
echo
echo "Default post-processing memory requested is $DEFAULT_POSTPROC_MEMORY"
echo

# Get the setting for the default partition
echo "Enter the default value for the partition for simulation jobs [$DEFAULT_PARTITION]:"
read -p "> "  input
if [ ! -z "$input" ]; then
  DEFAULT_PARTITION=$input
fi
echo
echo "Default partiion for simulations is $DEFAULT_PARTITION"
echo

# Get the setting for the default qos
echo "Enter the default value for the QOS for simulation jobs [$DEFAULT_QOS]:"
read -p "> "  input
if [ ! -z "$input" ]; then
  DEFAULT_QOS=$input
fi
echo
echo "Default QOS for simulations is $DEFAULT_QOS"
echo

# Now we have a 'julia' executable and the settings, can call a Julia script
# (machines/shared/machine_setup.jl) to create LocalPreferences.toml,
# julia.env, bin/julia, and some machine-specific symlinks.
echo "Doing initial setup"
echo
# Pass JULIA_DEPOT_PATH explicitly here because this script creates julia.env,
# so we cannot source it before running this script.
JULIA_DEPOT_PATH=$JULIA_DIRECTORY $JULIA machines/shared/machine_setup.jl "$MACHINE" "$ACCOUNT" "$JULIA_DIRECTORY" "$DEFAULT_RUN_TIME" "$DEFAULT_NODES" "$DEFAULT_POSTPROC_TIME" "$DEFAULT_POSTPROC_MEMORY" "$DEFAULT_PARTITION" "$DEFAULT_QOS"

# Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
source julia.env

# [ -f <path> ] tests if <path> exists and is a file
if [ -f machines/shared/compile_dependencies.sh ]; then
  # Need to compile some dependencies
  echo
  echo "Compliing dependencies"
  machines/shared/compile_dependencies.sh
fi

# [ -f <path> ] tests if <path> exists and is a file
if [ -f machines/shared/machine_setup_stage_two.jl ]; then
  # A second setup stage exists, so run it.
  # This does setup for HDF5 and MPI, possibly other things if necessary.
  echo
  echo "Doing stage two setup"
  bin/julia --project machines/shared/machine_setup_stage_two.jl
fi

echo "Do you want to submit a serial (or debug) job to precompile, creating the"
echo "moment_kinetics.so image (this is required in order to use the job submission"
echo "scripts and templates provided)? [y]/n:"
read -p "> "  input

while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
  # $input must be empty, 'y' or 'n'. It is none of these, so ask for input
  # again until we get a valid response.
  echo
  echo "$input is not a valid response: [y]/n"
  read -p "> "  input
done
if [[ -z $input || $input == "y" ]]; then
  # This script launches a job that runs precompile.jl to create the
  # moment_kinetics.so image.
  ./precompile-submit.sh
fi

echo
echo "Finished!"
echo "Now run \`source julia.env\` to set up your environment, and/or add it to your .bashrc"

exit 0
