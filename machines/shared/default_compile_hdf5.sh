#!/bin/bash

SYSTEM_MPI=$(../../bin/julia --project ../shared/get_mk_preference.jl use_system_mpi)

if [[ $SYSTEM_MPI == "y" ]]; then
  # Get default response for whether to download/build HDF5
  DEFAULT_BUILDHDF5=$(../../bin/julia --project ../shared/get_mk_preference.jl build_hdf5 "y")

  if [[ $DEFAULT_BUILDHDF5 == "y" ]]; then
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
      BUILDHDF5="y"
    else
      BUILDHDF5="n"
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
      BUILDHDF5="n"
    else
      BUILDHDF5="y"
    fi
  fi

  # Save current response for whether to download/build HDF5 as default
  ../../bin/julia --project ../shared/set_mk_preference.jl build_hdf5 $BUILDHDF5

  if [[ $BUILDHDF5 == "y" && -d hdf5-build ]]; then
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
      BUILDHDF5="n"
    else
      # Remove the install directory if it exists already
      if [[ -d hdf5-build ]]; then
        rm -r hdf5-build
      fi
    fi
  fi
else
  BUILD_HDF5="n"
fi

if [[ $BUILDHDF5 == "y" ]]; then
  HDF5=hdf5-hdf5_1.14.6
  # Download and extract the source
  wget -O ${HDF5}.tar.gz https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz
  tar xzf ${HDF5}.tar.gz

  cd $HDF5

  # Configure and compile
  CC=mpicc ./configure --enable-parallel --prefix=$ARTIFACT_DIR/hdf5-build/ | tee config.log
  make -j 4 | tee make.log
  make install | tee -a make.log
fi

exit 0
