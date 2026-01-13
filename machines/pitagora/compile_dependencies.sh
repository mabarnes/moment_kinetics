#!/usr/bin/env bash

set -e

if [ -f julia.env ]; then
  # Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
  source julia.env
fi
cd machines/artifacts/
export ARTIFACT_DIR=$PWD

# Note CMake module is required for compiling ADIOS2, but would cause linking
# errors if loaded while running moment_kinetics, so just load/unload
# explicitly here.
CMAKE_LOADED=$(module is-loaded cmake)
if [ ! $CMAKE_LOADED ]; then
  module load cmake
fi
../shared/default_compile_adios.sh
if [ ! $CMAKE_LOADED ]; then
  module unload cmake
fi

exit 0
