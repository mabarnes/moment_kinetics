#!/bin/bash

SYSTEM_MPI=$(../../bin/julia --project ../shared/get_mk_preference.jl use_system_mpi)
USE_ADIOS=$(../../bin/julia --project ../shared/get_mk_preference.jl use_adios)

if [[ $USE_ADIOS == "n" ]]; then
  BUILD_ADIOS2="n"
elif [[ $SYSTEM_MPI == "y" ]]; then
  # Get default response for whether to download/build ADIOS2
  DEFAULT_BUILDADIOS2=$(../../bin/julia --project ../shared/get_mk_preference.jl build_adios "y")

  if [[ $DEFAULT_BUILDADIOS2 == "y" ]]; then
    echo "Do you want to download, and compile a local version of ADIOS2 (if you do"
    echo "not do this, you will be given the option to choose an ADIOS2 library to"
    echo "link later)? [y]/n"
    read -p "> " input
    while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
      echo
      echo "$input is not a valid response: y/[n]"
      read -p "> " input
    done
    if [[ -z $input || $input == "y" ]]; then
      BUILDADIOS2="y"
    else
      BUILDADIOS2="n"
    fi
  else
    echo "Do you want to download, and compile a local version of ADIOS2 (if you do"
    echo "not do this, you will be given the option to choose an ADIOS2 library to"
    echo "link later)? y/[n]"
    read -p "> " input
    while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
      echo
      echo "$input is not a valid response: y/[n]"
      read -p "> " input
    done
    if [[ -z $input || $input == "n" ]]; then
      BUILDADIOS2="n"
    else
      BUILDADIOS2="y"
    fi
  fi

  # Save current response for whether to download/build ADIOS2 as default
  ../../bin/julia --project ../shared/set_mk_preference.jl build_adios $BUILDADIOS2

  if [[ $BUILDADIOS2 == "y" && -d adios-build ]]; then
    echo "ADIOS2 appears to have been downloaded, compiled and installed already."
    echo "Do you want to download, compile and install again, overwriting the existing "
    echo "version? y/[n]"
    read -p "> " input
    while [[ ! -z $input && !( $input == "y" || $input == "n" ) ]]; do
      echo
      echo "$input is not a valid response: y/[n]"
      read -p "> " input
    done
    if [[ -z $input || $input == "n" ]]; then
      BUILDADIOS2="n"
    else
      # Remove the install directory if it exists already
      if [[ -d adios-build ]]; then
        rm -r adios-build
      fi
    fi
  fi
else
  BUILD_ADIOS2="n"
fi

if [[ $BUILDADIOS2 == "y" ]]; then
  # Clone the source code.
  # The version probably needs to match the one that would be provided by
  # ADIOS2_jll, as this will be compatible with ADIOS2.jl
  git clone https://github.com/ornladios/ADIOS2.git -b v2.10.2

  cd ADIOS2/

  # Build using CMake.
  # Exclude some optional features that sometimes cause linking errors when
  # system versions of libraries conflict with the Julia package manager's
  # versions.
  cmake -B build -DCMAKE_INSTALL_PREFIX=$ARTIFACT_DIR/adios-build -DADIOS2_USE_SZ=OFF -DADIOS2_USE_UCX=OFF | tee cmake_config.log
  cmake --build build --parallel 8 | tee cmake_build.log
  cmake --install build | tee cmake_install.log
fi

exit 0
