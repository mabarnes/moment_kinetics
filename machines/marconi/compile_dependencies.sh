#!/usr/bin/env bash

set -e

cd machines/artifacts/
ARTIFACT_DIR=$PWD

# HDF5
######

BUILDHDF5=0
if [ -d hdf5-build ]; then
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
  else
    # Remove the install directory if it exists already
    if [[ -d hdf5-build ]]; then
      rm -r hdf5-build
    fi
  fi
fi

if [ $BUILDHDF5 -eq 0 ]; then
  HDF5=hdf5-hdf5_1.14.6
  # Download and extract the source
  wget -O ${HDF5}.tar.gz https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz
  tar xzf ${HDF5}.tar.gz

  cd $HDF5

  # Configure and compile
  CC=mpicc ./configure --enable-parallel --prefix=$ARTIFACT_DIR/hdf5-build/ | tee config.log
  make -j 16 | tee make.log
  make install | tee -a make.log
fi

exit 0
