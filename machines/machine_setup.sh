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
  # Try to find a sensible default if there is one in the path
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
# Make sure $JULIA is actually an executable
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
  # This command will fail if $MACHINE is not a know value, so `break` only
  # when it succeeds.
  DEFAULTS=($($JULIA machines/shared/machine_setup.jl --defaults $MACHINE)) && break
done

echo "Setting up for '$MACHINE'"
echo

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

echo "Doing initial setup"
echo
$JULIA machines/shared/machine_setup.jl "$MACHINE" "$ACCOUNT" "$JULIA_DIRECTORY"

source julia.env

if [ -f machines/shared/machine_setup_stage_two.jl ]; then
  # A second setup stage exists, so run it
  echo
  echo "Doing stage two setup"
  bin/julia --project machines/shared/machine_setup_stage_two.jl
fi

echo "Do you want to submit a serial (or debug) job to precompile, creating the"
echo "moment_kinetics.so image (this is required in order to use the job submission"
echo "scripts and templates provided)? [y]/n:"
read -p "> "  input
while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
  echo
  echo "$input is not a valid response: [y]/n"
  read -p "> "  input
done
if [[ -z $input || $input == "y" ]]; then
  ./precompile-submit.sh
fi

echo
echo "Finished!"
echo "Now run \`source julia.env\` to set up your environment, and/or add it to your .bashrc"

exit 0
