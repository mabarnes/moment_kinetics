#!/usr/bin/env bash

# Download and compile dependencies
###################################
#
# Note that you can delete this script if you only want to use a
# system-provided HDF5 library.

set -e

if [ -f julia.env ]; then
  # Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
  source julia.env
fi
cd machines/artifacts/
export ARTIFACT_DIR=$PWD

../shared/default_compile_hdf5.sh

exit 0
