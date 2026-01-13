#!/usr/bin/env bash

set -e

if [ -f julia.env ]; then
  # Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
  source julia.env
fi
cd machines/artifacts/
export ARTIFACT_DIR=$PWD

../shared/default_compile_hdf5.sh
../shared/default_compile_adios.sh

exit 0
