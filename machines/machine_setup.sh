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
  if ls /marconi > /dev/null 2>&1; then
    DEFAULT_MACHINE=marconi
  elif module avail 2>&1 | grep -q epcc; then
    DEFAULT_MACHINE=archer
  else
    DEFAULT_MACHINE=generic-pc
  fi
fi

# Create directory to set up Python venv and download/compile dependencies in
mkdir -p machines/artifacts

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

# Get name of 'machine'
while [[ -z $MACHINE || !( $MACHINE == "generic-pc" || $MACHINE == "archer" || $MACHINE == "marconi" ) ]]; do
  echo "Enter name of the machine to set up (must be one of 'generic-pc',"
  echo "'archer', or 'marconi') [$DEFAULT_MACHINE]:"
  read -p "> "  MACHINE
  echo
  if [ -z $MACHINE ]; then
    MACHINE=$DEFAULT_MACHINE
  fi
done

echo "Setting up for '$MACHINE'"
echo

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
# '-s' to skip resolving symlinks (if we did resolve symlinks, it might make
#      the path look different than expected).
JULIA=$(realpath -s $JULIA)
echo
echo "Using Julia at $JULIA"
echo
echo "$JULIA" > .julia_default.txt

# Save the machine name so we can use it as the default if we re-run
# machine_setup.sh.
echo $MACHINE > .this_machine_name.txt

if [[ $MACHINE == "generic-pc" ]]; then
  BATCH_SYSTEM=1
else
  BATCH_SYSTEM=0
fi

if [[ $BATCH_SYSTEM -eq 0 ]]; then
  # Batch systems can (conveniently) use different optimization flags for
  # running simulations and for post-processing.
  OPTIMIZATION_FLAGS="-O3 --check-bounds=no"
  POSTPROC_OPTIMIZATION_FLAGS="-O3"
else
  # On interactive systems which use the same project for running simulations
  # and for post-processing, both should use the same optimization flags to
  # avoid invalidating precompiled dependencies.
  OPTIMIZATION_FLAGS="-O3"
  POSTPROC_OPTIMIZATION_FLAGS=$OPTIMIZATION_FLAGS
fi

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
# '-m' to avoid erroring if the directory does not exist already.
# '-s' to skip resolving symlinks (if we did resolve symlinks, it might make
#      the path look different than expected).
if [ ! -z "$JULIA_DIRECTORY" ]; then
  JULIA_DIRECTORY=$(realpath -m -s $JULIA_DIRECTORY)
fi
echo
echo "Using julia_directory=$JULIA_DIRECTORY"
echo
echo $JULIA_DIRECTORY > .julia_directory_default.txt

if [[ $BATCH_SYSTEM -eq 0 ]]; then
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
    SUBMIT_PRECOMPILATION=0
  else
    SUBMIT_PRECOMPILATION=1
  fi
else
  SUBMIT_PRECOMPILATION=1
fi

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
export JULIA_DEPOT_PATH=$JULIA_DIRECTORY
$JULIA machines/shared/machine_setup.jl "$MACHINE"

if [ -f julia.env ]; then
  # Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
  source julia.env
fi

if [[ $USE_PLOTS_POSTPROC -eq 0 ]]; then
  # Create a Python venv, ensure it contains matplotlib, and append its
  # activation command to julia.env.
  # Use the `--system-site-packages` option to let the venv include any packages
  # already installed by the system.
  PYTHON_VENV_PATH=$PWD/machines/artifacts/mk_venv
  python -m venv --system-site-packages $PYTHON_VENV_PATH
  source $PYTHON_VENV_PATH/bin/activate
  # Use 'PYTHONNOUSERSITE=1' so that pip ignores any packages in ~/.local (which
  # may not be accessible from compute nodes on some clusters) and therefore
  # definitely installs matplotlib and its dependencies into mk_venv.
  PYTHONNOUSERSITE=1 pip install matplotlib
  if [[ $BATCH_SYSTEM -eq 0 ]]; then
    echo "source $PYTHON_VENV_PATH/bin/activate" >> julia.env
  else
    # Re-write bin/julia to add activation of the Python venv
    LAST_LINES=$(tail -n 2 bin/julia)
    echo '#!/usr/bin/env bash' > bin/julia # It is necessary to use single-quotes not double quotes here, but don't understand why
    echo "source $PYTHON_VENV_PATH/bin/activate" >> bin/julia
    echo "$LAST_LINES" >> bin/julia
  fi
fi

# [ -f <path> ] tests if <path> exists and is a file
if [ -f machines/shared/compile_dependencies.sh ]; then
  # Need to compile some dependencies
  echo
  echo "Compiling dependencies"
  machines/shared/compile_dependencies.sh
fi

# We want to always add a couple of dependencies that are required to run the
# tests in the top-level environment by just 'include()'ing the test scripts.
# We also run the 'stage two' setup now if it is required.
SETUP_COMMAND="bin/julia --project $OPTIMIZATION_FLAGS -e 'import Pkg"
if [[ $USE_NETCDF -eq 1 || $ENABLE_MMS -eq 1 ]]; then
  # Remove packages used by non-selected extensions in case they were installed previously
  if [[ $USE_NETCDF -eq 1 ]]; then
    SETUP_COMMAND="$SETUP_COMMAND; try Pkg.rm(\"NCDatasets\") catch end"
  fi
  if [[ $ENABLE_MMS -eq 1 ]]; then
    SETUP_COMMAND="$SETUP_COMMAND; try Pkg.rm([\"Symbolics\", \"IfElse\"]) catch end"
  fi
fi
SETUP_COMMAND="$SETUP_COMMAND; Pkg.add([\"HDF5\", \"MPI\", \"MPIPreferences\", \"SpecialFunctions\""
if [[ $USE_NETCDF -eq 0 ]]; then
  # Install NetCDF interface package NCDatasets to enable file_io_netcdf extension
  SETUP_COMMAND="$SETUP_COMMAND, \"NCDatasets\""
fi
if [[ $ENABLE_MMS -eq 0 ]]; then
  # Install Symbolics and IfElse packages required by manufactured_solns_ext extension
  SETUP_COMMAND="$SETUP_COMMAND, \"Symbolics\", \"IfElse\""
fi
SETUP_COMMAND="$SETUP_COMMAND]); include(\"machines/shared/machine_setup_stage_two.jl\")'"
eval "$SETUP_COMMAND"

# Add moment_kinetics package to the working project
bin/julia --project $OPTIMIZATION_FLAGS -e 'import Pkg; Pkg.develop(path="moment_kinetics"); Pkg.precompile()'

if [[ $USE_MAKIE_POSTPROC -eq 0 ]]; then
  echo "Setting up makie_post_processing"
  if [[ $BATCH_SYSTEM -eq 0 ]]; then
    bin/julia --project=makie_post_processing/ $POSTPROC_OPTIMIZATION_FLAGS -e 'import Pkg; Pkg.develop(path="moment_kinetics/"); Pkg.develop(path="makie_post_processing/makie_post_processing"); Pkg.precompile()'
  else
    bin/julia --project -e $OPTIMIZATION_FLAGS 'import Pkg; Pkg.develop(path="makie_post_processing/makie_post_processing")'
  fi
else
  if [[ $BATCH_SYSTEM -eq 1 ]]; then
    bin/julia --project $OPTIMIZATION_FLAGS -e 'import Pkg; try Pkg.rm("makie_post_processing") catch end'
  fi
fi

if [[ $USE_PLOTS_POSTPROC -eq 0 ]]; then
  echo "Setting up plots_post_processing"
  if [[ $BATCH_SYSTEM -eq 0 ]]; then
    bin/julia --project=plots_post_processing/ $POSTPROC_OPTIMIZATION_FLAGS -e 'import Pkg; Pkg.develop(path="moment_kinetics/"); Pkg.develop(path="plots_post_processing/plots_post_processing"); Pkg.precompile()'
  else
    bin/julia --project $OPTIMIZATION_FLAGS -e 'import Pkg; Pkg.develop(path="plots_post_processing/plots_post_processing")'
  fi
else
  if [[ $BATCH_SYSTEM -eq 1 ]]; then
    bin/julia --project $OPTIMIZATION_FLAGS -e 'import Pkg; try Pkg.rm("plots_post_processing") catch end'
  fi
fi

if [[ $BATCH_SYSTEM -eq 0 ]]; then
  # Make symlinks to submission scripts
  ln -s machines/shared/precompile-submit.sh
  ln -s machines/shared/submit-run.sh
  ln -s machines/shared/submit-restart.sh
  if [[ $USE_MAKIE_POSTPROC -eq 0 ]]; then
    ln -s machines/shared/precompile-makie-post-processing-submit.sh
  fi
  if [[ $USE_PLOTS_POSTPROC -eq 0 ]]; then
    ln -s machines/shared/precompile-plots-post-processing-submit.sh
  fi
fi

if [[ $SUBMIT_PRECOMPILATION -eq 0 ]]; then
  # This script launches a job that runs precompile.jl to create the
  # moment_kinetics.so image.
  ./precompile-submit.sh

  if [[ $USE_MAKIE_POSTPROC -eq 0 ]]; then
    ./precompile-makie-post-processing-submit.sh
  fi
  if [[ $USE_PLOTS_POSTPROC -eq 0 ]]; then
    ./precompile-plots-post-processing-submit.sh
  fi
fi

echo
echo "Finished!"
if [[ $BATCH_SYSTEM -eq 0 ]]; then
  echo "Now run \`source julia.env\` to set up your environment, and/or add it to your .bashrc"
fi

exit 0
