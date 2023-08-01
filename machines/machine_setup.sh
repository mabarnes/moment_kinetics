#!/usr/bin/env bash

set -e

# Parse command line options
# [See e.g. https://www.stackchief.com/tutorials/Bash%20Tutorial%3A%20getopts
# for examples of how to use Bash's getopts]
FORCE_DOWNLOAD=1
while getopts "hd" opt; do
  case $opt in
    h)
      echo "Setup moment_kinetics to run on a known machine"
      echo "Usage: machine_setup.sh [-d] [/path/to/julia]"
      echo
      echo "If the path to julia is not given, you will be prompted for it."
      echo "  -d  Force Julia to be downloaded even if julia executable is"
      echo "      present in \$PATH. Ignored if </path/to/julia> is passed"
      echo "      explicitly."
      exit 1
      ;;
    d)
      FORCE_DOWNLOAD=0
      ;;
  esac
done

# Get the positional argument as INPUTFILE
# [See https://stackoverflow.com/a/13400237]
JULIA=${@:$OPTIND:1}

# Apply heuristics to try and get a default value for MACHINE.
# Note these are machine-specific guesses, and the tests may well be broken by
# changes to machine configuration, etc.
if ls /marconi > /dev/null 2>&1; then
  DEFAULT_MACHINE=marconi
elif module avail 2>&1 | grep -q epcc; then
  DEFAULT_MACHINE=archer
else
  DEFAULT_MACHINE=
fi

# Make sure $JULIA is set
# Note [ -z "$VAR" ] tests if $VAR is empty. Need the quotes to ensure that the
# contents of $VAR are not evaluated if it is not empty.
if [ -z "$JULIA" ]; then
  # Try to find a sensible default if there is one in the $PATH
  if [ $(command -v julia) ]; then
    JULIA=$(command -v julia)
  fi
  # REQUEST_INPUT will be set to 1 if the path to Julia is not needed because
  # Julia was downloaded. If it stays at zero, request a path to Julia after
  # looking for a good default value in this block.
  REQUEST_INPUT=0
  if [[ -z "$JULIA" || $FORCE_DOWNLOAD -eq 0 ]]; then
    if [[ $FORCE_DOWNLOAD -eq 0 ]]; then
      # '-d' flag was passed, so carry on and download Julia without prompting
      input="y"
    else
      echo "No 'julia' found in \$PATH. Would you like to download Julia? [y]/n"
      read -p "> " input
      echo
    fi
    while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
      # $input must be empty, 'y' or 'n'. It is none of these, so ask for input
      # again until we get a valid response.
      echo
      echo "$input is not a valid response: [y]/n"
      read -p "> "  input
      echo
    done

    if [[ -z $input || $input == "y" ]]; then
      # Need machine name to get the right get-julia.sh script.
      # Not ideal to get input here as in the 'proper' place below we can check
      # for a valid input for $MACHINE and list the allowed values (although
      # there will be an error here if the input is not valid as
      # 'machines/$MACHINE/get-julia.sh would not exist). However the checking
      # requires Julia, so cannot do it here as we need the value in order to
      # download Julia (need to know the machine so that we download the right
      # set of binaries for the OS, architecture, etc.
      while [ -z $MACHINE ]; do
        echo "Enter name of the machine to set up [$DEFAULT_MACHINE]:"
        read -p "> "  MACHINE
        echo
        if [ -z $MACHINE ]; then
          MACHINE=$DEFAULT_MACHINE
        fi
      done

      # Download a version of Julia that is correct for this machine.
      # Here the user can specify which version of Julia they want to use in
      # case the latest stable versionis not wanted.
      echo "Enter the version of Julia to download [latest]:"
      read -p "> "  version

      # The get-julia.sh script for $MACHINE should download and extract Julia
      # (into machines/artifacts/). It prints the path to the 'julia'
      # executable, which we store in $JULIA.
      JULIA=$(machines/$MACHINE/get-julia.sh $version)

      # Make sure $JULIA is actually an executable
      # Note 'command -v foo' returns success if foo is an executable and failure
      # otherwise, except that if no argument is passed 'command -v' returns success,
      # so we also need to check that $JULIA is not empty.
      if [[ -z "$JULIA" || ! $(command -v $JULIA) ]]; then
        echo "Failed to download Julia. '$JULIA' is not an executable"
        exit 1
      fi
      REQUEST_INPUT=1
    fi
  fi
  if [ $REQUEST_INPUT -eq 0 ]; then
    # Have not downloaded Julia, so need to get the path to a 'julia'
    # executable from the user. If 'julia' was found in $PATH, its location is
    # stored in $JULIA at the moment, so use this as the default value.
    echo "Please enter the path to the Julia executable to be used ['$JULIA']:"
    # Use '-e' option to get path auto-completion
    read -e -p "> " input
    if [ ! -z "$input" ]; then
      JULIA=$input
    fi
  fi
fi
# Ensure we have the resolved path for $JULIA, to avoid potentially creating a
# self-referencing symlink at bin/julia
JULIA=$(realpath $JULIA)

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

# If Julia was downloaded by this script, $MACHINE is already set. Otherwise,
# need to get and check its value here.
if [ -z "$MACHINE" ]; then
  # Make first attempt at getting the name of the machine to setup
  echo "Enter name of the machine to set up [$DEFAULT_MACHINE]:"
  read -p "> "  MACHINE
  echo
  if [ -z $MACHINE ]; then
    MACHINE=$DEFAULT_MACHINE
  fi
fi
while true; do
  # Get default values for this machine from the machine_setup.jl script
  # Note: the brackets() around the command execution turn the result into an
  # 'array' that we can get the separate elements from below.
  # This command will fail (and print a helpful error message) if $MACHINE is
  # not a known value, so `break` only when it succeeds.
  DEFAULTS=($($JULIA machines/shared/machine_setup.jl --defaults $MACHINE)) && break

  # If we did not get a good value for $MACHINE yet, prompt for a new one.
  echo "Enter name of the machine to set up:"
  read -p "> "  MACHINE
  echo
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
if [ ! -z "$JULIA_DIRECTORY" ]; then
  JULIA_DIRECTORY=$(realpath -m -s $JULIA_DIRECTORY)
fi
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

# Create a Python venv, ensure it contains matplotlib, and append its
# activation command to julia.env.
# Use the `--system-site-packages` option to let the venv include any packages
# already installed by the system.
PYTHON_VENV_PATH=$PWD/machines/artifacts/mk_venv
python -m venv --system-site-packages $PYTHON_VENV_PATH
source $PYTHON_VENV_PATH/bin/activate
pip install matplotlib
echo "source $PYTHON_VENV_PATH/bin/activate" >> julia.env

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
