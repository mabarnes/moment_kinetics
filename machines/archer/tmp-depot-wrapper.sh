#!/bin/bash

# Only create/delete the temporary directory on one process on each node
if [ $SLURM_LOCALID -eq 0 ]; then
  # compute-node-temp.julia.tar.bz should have already been copied to /tmp by `sbcast` in the job submission script
  mkdir /tmp/$USER/
  mv /tmp/compute-node-temp.julia.tar.bz /tmp/$USER/compute-node-temp.julia.tar.bz
  tar -C /tmp/$USER -xjf compute-node-temp.julia.tar.bz

  # Don't need the tar file any more, so delete to save memory
  rm /tmp/$USER/compute-node-temp.julia.tar.bz

  touch /tmp/$USER/ready
else
  # https://unix.stackexchange.com/a/185370
  until [ -e /tmp/$USER/ready ]; do
    sleep 0.1
  done
fi

# Execute the arguments to this wrapper script
"$@"

if [ $SLURM_LOCALID -eq 0 ]; then
  # /tmp/ should be cleaned up automatically on ARCHER2, but just in case, delete our temporary depot.
  rm -rf /tmp/$USER/
fi
