#!/usr/bin/env bash

set -e

# Parse command line options
# [See e.g. https://www.stackchief.com/tutorials/Bash%20Tutorial%3A%20getopts
# for examples of how to use Bash's getopts]
while getopts "h" opt; do
  case $opt in
    h)
      echo "Setup moment_kinetics to run on a known machine"
      exit 1
      ;;
  esac
done

# Ensure a Project.toml exists in the top-level directory, which defines a
# project environment that we can install packages to.
touch Project.toml

# If `julia` executable has been set up before, then that will be used as the
# default selection (unless -d flag is passed explicitly to download `julia`).
if [[ -f .julia_default.txt ]]; then
  JULIA=$(cat .julia_default.txt)
  if [[ ! -f $JULIA ]]; then
    # Previously-used path to `julia` no longer exists, so ignore it and remove
    # the stored path.
    JULIA=""
    echo > .julia_default.txt
  fi
fi

if [[ -f .this_machine_name.txt ]]; then
  DEFAULT_MACHINE=$(cat .this_machine_name.txt)
else
  # Apply heuristics to try and get a default value for MACHINE.
  # Note these are machine-specific guesses, and the tests may well be broken by
  # changes to machine configuration, etc.
  if ls /pitagora > /dev/null 2>&1; then
    DEFAULT_MACHINE=pitagora
  elif module avail 2>&1 | grep -q epcc; then
    DEFAULT_MACHINE=archer
  elif $(command -v module avail); then
    DEFAULT_MACHINE=generic-batch
  else
    DEFAULT_MACHINE=generic-pc
  fi
fi

# Create directory to set up Python venv and download/compile dependencies in
mkdir -p machines/artifacts

# Get name of 'machine'
while [[ -z $MACHINE || !( $MACHINE == "generic-pc" || $MACHINE == "generic-batch" || $MACHINE == "archer" || $MACHINE == "pitagora" ) ]]; do
  echo "Enter name of the machine to set up (must be one of 'generic-pc',"
  echo "'generic-batch', 'archer', 'pitagora') [$DEFAULT_MACHINE]:"
  read -p "> "  MACHINE
  echo
  if [ -z $MACHINE ]; then
    MACHINE=$DEFAULT_MACHINE
  fi
done

if [[ $MACHINE == "generic-batch" && ! -d machines/generic-batch ]]; then
  echo "To use 'generic-batch' you must copy 'machines/generic-batch-template' to 'machines/generic-batch' and:"
  echo "* Edit the modules in 'machines/generic-batch/julia.env' (see comments in that file)"
  echo "* Edit the 'jobscript-*.template' files for precompilation or post-processing jobswith the correct serial or"
  echo "  debug queue for your machine."
  echo "* If you want to use a system-provided HDF5 you can delete 'machines/generic-batch/compile_dependencies.sh',"
  echo "  and uncomment the 'hdf5_library_setting = \"system\"' option in 'machines/generic-batch/machine_settings.toml'"
  echo "* If 'MPIPreferences.use_system_binary()' cannot auto-detect your MPI library and/or if 'mpirun' is not the "
  echo "  right command to launch MPI processes, then you need to set the 'mpi_library_names' and 'mpiexec' settings in"
  echo "  'machines/generic-batch/machine_settings.toml' (note if either of these settings is set, then both must be)"
  echo "* If 'mpirun' is not the right command to launch MPI processes, you may need to edit the 'jobscript-run.template'"
  echo "  and 'jobscript-restart.template' files in 'machines/generic-batch/' and set the setting in"
  echo "  'machines/generic-batch/machine_settings.toml'"
  echo "Note that 'generic-batch' is set up assuming a Linux, x86_64 based machine that uses the 'module' system and a"
  echo "SLURM job queue."
  exit 1
fi

echo "Setting up for '$MACHINE'"
echo

# Save the machine name so we can use it as the default if we re-run
# machine_setup.sh.
echo $MACHINE > .this_machine_name.txt

if [[ $MACHINE == "generic-pc" ]]; then
  BATCH_SYSTEM=1
else
  BATCH_SYSTEM=0
fi

# Make sure $JULIA is set
# Note [ -z "$VAR" ] tests if $VAR is empty. Need the quotes to ensure that the
# contents of $VAR are not evaluated if it is not empty.
if [ -z "$JULIA"  ]; then
  # Try to find a sensible default if there is one in the $PATH
  if [ $(command -v julia) ]; then
    JULIA=$(command -v julia)
  fi
fi
# REQUEST_INPUT will be set to 1 if the path to Julia is not needed because
# Julia was downloaded. If it stays at zero, request a path to Julia.
REQUEST_INPUT=0
DOWNLOAD_JULIA=1
if [[ -z "$JULIA" ]]; then
  echo "No previously-used 'julia' or 'julia' found in \$PATH. Would you like"
  echo "to download Julia? [y]/n"
  read -p "> " input
  echo
  while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
    # $input must be empty, 'y' or 'n'. It is none of these, so ask for input
    # again until we get a valid response.
    echo
    echo "$input is not a valid response: [y]/n"
    read -p "> "  input
    echo
  done
  if [[ -z $input || $input == "y" ]]; then
    DOWNLOAD_JULIA=0
  fi
else
  echo "Existing 'julia' is available ($JULIA). Would you like to download"
  echo "Julia anyway? y/[n]"
  read -p "> " input
  echo
  while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
    # $input must be empty, 'y' or 'n'. It is none of these, so ask for input
    # again until we get a valid response.
    echo
    echo "$input is not a valid response: y/[n]"
    read -p "> "  input
    echo
  done
  if [[ $input == "y" ]]; then
    DOWNLOAD_JULIA=0
  fi
fi

if [[ $DOWNLOAD_JULIA -eq 0 ]]; then
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
if [ $REQUEST_INPUT -eq 0 ]; then
  # Have not downloaded Julia, so need to get the path to a 'julia'
  # executable from the user. If 'julia' was specified before, or found in
  # $PATH, its location is stored in $JULIA at the moment, so use this as the
  # default value.
  echo "Please enter the path to the Julia executable to be used ['$JULIA']:"
  # Use '-e' option to get path auto-completion
  read -e -p "> " input
  if [ ! -z "$input" ]; then
    JULIA=$input
  fi
fi
# Ensure we have the resolved path for $JULIA, to avoid potentially creating a
# self-referencing symlink at bin/julia.
# Use Python's `os.path` module instead of GNU coreutils `realpath` as
# `realpath` may not be available on all systems (e.g. some MacOS versions),
# but we already assume Python is available.
JULIA=$(/usr/bin/env python3 -c "import os; juliapath = os.path.realpath('$JULIA'); not os.path.isfile(juliapath) and exit(1); print(juliapath)")

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
# Use Python's `os.path` module instead of GNU coreutils `realpath` as
# `realpath` may not be available on all systems (e.g. some MacOS versions),
# but we already assume Python is available.
# Use `os.path.abspath` rather than `os.path.realpath` to skip resolving
# symlinks (if we did resolve symlinks, it might make the path look different
# than expected).
JULIA=$(/usr/bin/env python3 -c "import os; juliapath = os.path.abspath('$JULIA'); not os.path.isfile(juliapath) and exit(1); print(juliapath)")
echo
echo "Using Julia at $JULIA"
echo
echo "$JULIA" > .julia_default.txt

if [ x$MACHINE == xarcher -o x$MACHINE == xpitagora ]; then
  JULIA_DEPOT_IN_TMP=true

  # Julia depot is created in /tmp/ so that it can be copied into /tmp/ on each
  # of the compute nodes for parallel jobs, to work around an issue where Julia
  # hangs while starting up on >~2048 MPI processes.
  # Here we set $JULIA_DEPOT_PATH explicitly so that it gets used within this
  # script to set up the depot under /tmp, but do not set JULIA_DIRECTORY so
  # that when julia is run on the login nodes, the default JULIA_DIRECTORY is
  # used (which will be in the user's home directory).
  export JULIA_DEPOT_PATH=/tmp/$USER/compute-node-temp.julia

  mkdir -p /tmp/$USER/

  # Clean up the temporary directory when this script exits for any reason
  # (this might not be done automatically on login nodes).
  # https://stackoverflow.com/a/53063602
  trap "rm -rf /tmp/$USER" EXIT SIGHUP SIGINT SIGQUIT SIGILL SIGTRAP SIGABRT SIGBUS SIGFPE SIGKILL SIGUSR1 SIGSEGV SIGUSR2 SIGTERM SIGSTOP SIGTSTP

  # Re-use existing tar'ed depot directory if it exists.
  if [ -e compute-note-temp.julia.tar.bz ]; then
    tar -xjf compute-note-temp.julia.tar.bz -C /tmp/$USER/
  fi
else
  # Get the location for the .julia directory, in case this has to have a
  # non-default value, e.g. because the user's home directory is not accessible
  # from compute nodes.
  if [[ -f .julia_directory_default.txt ]]; then
    JULIA_DIRECTORY=$(cat .julia_directory_default.txt)
  else
    # If we do not have an existing setting, try using $JULIA_DEPOT_PATH (if that
    # is not set, then we end up with an empty default, which means use the
    # standard default location).
    JULIA_DIRECTORY=$JULIA_DEPOT_PATH
  fi
  echo "It can be useful or necessary to set a non-default location for the "
  echo ".julia directory. Leave this empty if the default location is OK."
  echo "Enter a name for a subdirectory of the current directory, e.g. "
  echo "'.julia', to isolate the julia used for this instance of "
  echo "moment_kinetics - this might be useful to ensure a 'clean' install or "
  echo "to check whether some error is related to conflicting or corrupted "
  echo "dependencies or cached precompilation files, etc."
  echo "Enter location that should be used for the .julia directory [$JULIA_DIRECTORY]:"
  # Use '-e' option to get path auto-completion
  read -e -p "> "  input
  if [ ! -z "$input" ]; then
    JULIA_DIRECTORY=$input
  fi
  # Convert input (which might be a relative path) to an absolute path.
  # Note that here we do not require the directory to exist already - if it does
  # not exist then Julia will create it.
  # Use Python's `os.path` module instead of GNU coreutils `realpath` as
  # `realpath` may not be available on all systems (e.g. some MacOS versions),
  # but we already assume Python is available.
  # Use `os.path.abspath` rather than `os.path.realpath` to skip resolving
  # symlinks (if we did resolve symlinks, it might make the path look different
  # than expected).
  if [ ! -z "$JULIA_DIRECTORY" ]; then
    JULIA_DIRECTORY=$(/usr/bin/env python3 -c "import os; print(os.path.abspath('$JULIA_DIRECTORY'))")
  fi
  echo
fi
echo "Using julia_directory=$JULIA_DIRECTORY"
echo
echo $JULIA_DIRECTORY > .julia_directory_default.txt

# Now we have a 'julia' executable and the settings, can call a Julia script
# (machines/shared/machine_setup.jl) to create LocalPreferences.toml,
# julia.env, bin/julia, and some machine-specific symlinks.
echo "Doing initial setup"
echo
# Pass JULIA_DEPOT_PATH explicitly here because this script creates bin/julia
# or julia.env, so we cannot use either before running this script.
# export JULIA_DEPOT_PATH instead of just passing as a prefix to the julia
# command, because passing as a prefix does not work (sometimes??) within a
# bash script (even though as far as JTO knows it should work).
if [ ! -z "$JULIA_DIRECTORY" ]; then
  export JULIA_DEPOT_PATH=$JULIA_DIRECTORY
fi
$JULIA --project machines/shared/machine_setup.jl "$MACHINE"

if [ -f julia.env ]; then
  # Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
  source julia.env
fi

SEPARATE_POSTPROC_PROJECTS=$(bin/julia --project machines/shared/get_mk_preference.jl separate_postproc_projects)
if [[ $BATCH_SYSTEM -eq 0 || $SEPARATE_POSTPROC_PROJECTS == "y" ]]; then
  # Batch systems can (conveniently) use different optimization flags for
  # running simulations and for post-processing.
  OPTIMIZATION_FLAGS="-O3"
  POSTPROC_OPTIMIZATION_FLAGS="-O3"
else
  # On interactive systems which use the same project for running simulations
  # and for post-processing, both should use the same optimization flags to
  # avoid invalidating precompiled dependencies.
  OPTIMIZATION_FLAGS="-O3"
  POSTPROC_OPTIMIZATION_FLAGS=$OPTIMIZATION_FLAGS
fi

# [ -f <path> ] tests if <path> exists and is a file
if [ -f machines/shared/compile_dependencies.sh ]; then
  # Need to compile some dependencies
  echo
  echo "Compiling dependencies"
  machines/shared/compile_dependencies.sh
fi

bin/julia --project $OPTIMIZATION_FLAGS machines/shared/add_dependencies_to_project.jl
if [ x$JULIA_DEPOT_IN_TMP == xtrue ]; then
  # tar up the Julia depot directory from /tmp/$USER so that we can save it and
  # unpack it in future on compute nodes.
  tar cjf compute-node-temp.julia.tar.bz -C /tmp/$USER/ compute-node-temp.julia/
fi
# Don't use bin/julia for machine_setup_stage_two.jl because that script modifies bin/julia.
# It is OK to not use it here, because JULIA_DEPOT_PATH has been set within this script
$JULIA --project $OPTIMIZATION_FLAGS machines/shared/machine_setup_stage_two.jl
bin/julia --project $POSTPROC_OPTIMIZATION_FLAGS machines/shared/makie_post_processing_setup.jl
bin/julia --project $POSTPROC_OPTIMIZATION_FLAGS machines/shared/plots_post_processing_setup.jl

SUBMIT_PRECOMPILATION=$(bin/julia --project machines/shared/get_mk_preference.jl submit_precompilation)
USE_MAKIE=$(bin/julia --project machines/shared/get_mk_preference.jl use_makie)
USE_PLOTS=$(bin/julia --project machines/shared/get_mk_preference.jl use_plots)
if [[ $USE_MAKIE == "y" || $USE_PLOTS == "y" ]]; then
  if [ x$JULIA_DEPOT_IN_TMP == xtrue ]; then
    # More packages have been added to the depot when the plotting package(s)
    # were added, so re-create the tar.
    tar cjf compute-node-temp.julia.tar.bz -C /tmp/$USER/ compute-node-temp.julia/
  fi

  if [[ $SUBMIT_PRECOMPILATION == "y" ]]; then
    if [[ $USE_MAKIE == "y" ]]; then
      ./precompile-makie-post-processing-submit.sh
    fi
    if [[ $USE_PLOTS == "y" ]]; then
      ./precompile-plots-post-processing-submit.sh
    fi
  fi
fi

echo
echo "Finished!"
if [[ $BATCH_SYSTEM -eq 0 ]]; then
  echo "Now run \`source julia.env\` to set up your environment, and/or add it to your .bashrc"
fi
echo
if [[ $BATCH_SYSTEM -eq 0 ]]; then
  echo "To run simulations interactively, start julia like:"
  echo "$ bin/julia --project $OPTIMIZATION_FLAGS"
  echo
  echo "To run post-processing interactively, start julia like:"
  echo "$ bin/julia --project=makie_post_processing $POSTPROC_OPTIMIZATION_FLAGS"
  echo "or"
  echo "$ bin/julia --project=plots_post_processing $POSTPROC_OPTIMIZATION_FLAGS"
  echo
  echo "Note that if you change the optimization flags '$OPTIMIZATION_FLAGS' or '$POSTPROC_OPTIMIZATION_FLAGS' precompilation may need to be repeated using the new flags (which is slow)."
elif [[ $SEPARATE_POSTPROC_PROJECTS == "y" ]]; then
  echo "To run simulations, start julia like:"
  echo "$ bin/julia --project $OPTIMIZATION_FLAGS"
  echo
  echo "To run post-processing, start julia like:"
  echo "$ bin/julia --project=makie_post_processing $POSTPROC_OPTIMIZATION_FLAGS"
  echo "or"
  echo "$ bin/julia --project=plots_post_processing $POSTPROC_OPTIMIZATION_FLAGS"
  echo
  echo "Note that if you change the optimization flags '$OPTIMIZATION_FLAGS' or '$POSTPROC_OPTIMIZATION_FLAGS' precompilation may need to be repeated using the new flags (which is slow)."
else
  echo "To run simulations or do post-processing, start julia like:"
  echo "$ bin/julia --project $OPTIMIZATION_FLAGS"
  echo
  echo "Note that if you change the optimization flags '$OPTIMIZATION_FLAGS' precompilation may need to be repeated using the new flags (which is slow)."
fi

exit 0
