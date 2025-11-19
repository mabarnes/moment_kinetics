#!/usr/bin/env bash

ARTIFACT_DIR=machines/artifacts/

ARCH=""
while [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; do
  echo "Which architecture does your Mac use - 'x86_64' (Intel processor)" > /dev/tty
  echo "or 'aarch64' (Apple processor)? x86_64/[aarch64]" > /dev/tty
  read -p "> " ARCH
  if [[ -z $ARCH ]]; then
    ARCH=aarch64
  fi
done

# Download Julia binary distribution
if [ -z "$1" ]; then
  # No version argument passed
  JULIA_TAR=$(machines/shared/get-julia.py --output-dir $ARTIFACT_DIR --os mac --arch $ARCH)
else
  JULIA_TAR=$(machines/shared/get-julia.py --output-dir $ARTIFACT_DIR --os mac --arch $ARCH --version $1)
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
