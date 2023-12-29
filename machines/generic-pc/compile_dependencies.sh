#!/usr/bin/env bash

set -e

if [ -f julia.env ]; then
  # Set up modules, JULIA_DEPOT_PATH, etc. to use for the rest of this script
  source julia.env
fi
cd machines/artifacts/
ARTIFACT_DIR=$PWD

# HDF5
######

# Get default response for whether to download/build HDF5
DEFAULT_BUILDHDF5=$(../../bin/julia ../shared/get_mk_preference.jl build_hdf5 0)

if [[ $DEFAULT_BUILDHDF5 -eq 0 ]]; then
  echo "Do you want to download, and compile a local version of HDF5 (if you do"
  echo "not do this, you will be given the option to choose an HDF5 library to"
  echo "link later)? [y]/n"
  read -p "> " input
  while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
    echo
    echo "$input is not a valid response: y/[n]"
    read -p "> " input
  done
  if [[ -z $input || $input == "y" ]]; then
    BUILDHDF5=0
  else
    BUILDHDF5=1
  fi
else
  echo "Do you want to download, and compile a local version of HDF5 (if you do"
  echo "not do this, you will be given the option to choose an HDF5 library to"
  echo "link later)? y/[n]"
  read -p "> " input
  while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
    echo
    echo "$input is not a valid response: y/[n]"
    read -p "> " input
  done
  if [[ -z $input || $input == "n" ]]; then
    BUILDHDF5=1
  else
    BUILDHDF5=0
  fi
fi

# Save current response for whether to download/build HDF5 as default
../../bin/julia ../shared/set_mk_preference.jl build_hdf5 $BUILDHDF5

if [[ BUILDHDF5 -eq 1 && -d hdf5-build ]]; then
  rm -r hdf5-build
elif [[ BUILDHDF5 -eq 0 && -d hdf5-build ]]; then
  echo "HDF5 appears to have been downloaded, compiled and installed already."
  echo "Do you want to download, compile and install again, overwriting the existing "
  echo "version? y/[n]"
  read -p "> " input
  while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
    echo
    echo "$input is not a valid response: y/[n]"
    read -p "> " input
  done
  if [[ -z $input || $input == "n" ]]; then
    BUILDHDF5=1
  fi
fi

if [ $BUILDHDF5 -eq 0 ]; then
  HDF5=hdf5-1.14.3
  # Download and extract the source
  wget -O ${HDF5}.tar.bz2 https://www.hdfgroup.org/package/hdf5-1-14-3-tar-bz2/?wpdmdl=18469
  tar xjf ${HDF5}.tar.bz2

  cd $HDF5

  # Configure and compile
  CC=mpicc ./configure --enable-parallel --prefix=$ARTIFACT_DIR/hdf5-build/
  make -j 4
  make install
fi

exit 0
