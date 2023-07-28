#!/bin/bash

ARTIFACT_DIR=machines/artifacts/

# Download Julia binary distribution
if [ -z "$1" ]; then
  # No version argument passed
  JULIA_TAR=$(machines/shared/get-julia.py --output-dir $ARTIFACT_DIR --os linux --arch x86_64)
else
  JULIA_TAR=$(machines/shared/get-julia.py --output-dir $ARTIFACT_DIR --os linux --arch x86_64 --version $1)
fi

cd $ARTIFACT_DIR

# Get the directory name from the tar file, see
# https://unix.stackexchange.com/a/246807
JULIA_DIR=$(tar tf $JULIA_TAR | head -n 1)

# Assuming Julia's archives will always be gzipped (.tar.gz) here
tar -xzf $JULIA_TAR

# echo the path to the Julia binary so we can use it in another script
echo $ARTIFACT_DIR/$JULIA_DIR/bin/julia

exit 0
