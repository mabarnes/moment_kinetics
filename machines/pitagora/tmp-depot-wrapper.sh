#!/bin/bash

# Only create/delete the temporary directory on one process on each node
if [ $SLURM_LOCALID -eq 0 ]; then
  # compute-node-temp.julia.tar.bz should have already been copied to $TMPDIR by `sbcast` in the job submission script
  mkdir $TMPDIR/$USER/
  mv $TMPDIR/compute-node-temp.julia.tar.bz $TMPDIR/$USER/compute-node-temp.julia.tar.bz
  tar -C $TMPDIR/$USER -xjf compute-node-temp.julia.tar.bz

  # Don't need the tar file any more, so delete to save memory
  rm $TMPDIR/$USER/compute-node-temp.julia.tar.bz

  # Move julia executable and .so library into the subdirectory so that they
  # get cleaned up at the end.
  mv $TMPDIR/julia-dir $TMPDIR/$USER/
  mv $TMPDIR/*.so $TMPDIR/$USER/

  touch $TMPDIR/$USER/ready
else
  # https://unix.stackexchange.com/a/185370
  until [ -e $TMPDIR/$USER/ready ]; do
    sleep 0.1
  done
fi

export JULIA_DEPOT_PATH=$TMPDIR/$USER/compute-node-temp.julia

# Execute the arguments to this wrapper script
"$@"

if [ $SLURM_LOCALID -eq 0 ]; then
  # $TMPDIR/ should be cleaned up automatically on Pitagora, but just in case, delete our temporary depot.
  rm -rf $TMPDIR/$USER/
fi
