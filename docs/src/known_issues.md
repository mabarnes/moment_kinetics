Known Issues
============

Slow startup with very large numbers of MPI processes
-----------------------------------------------------

When using a very large number of MPI processes (e.g. more than 2048 on
ARCHER2), Julia may be slow to start
https://discourse.julialang.org/t/speeding-up-julia-startup-with-mpi-using-scratch-space-to-mitigate-hdd-access-delay/.

The work around for this is to copy the Julia depot (the directory usually
called `.julia`) onto storage local to each compute node, so that only the
number of processes from a single node access each copy. This not a
straightforward thing to do, because (at least up to Julia-1.10), the path to
the depot is not allowed to change between when it is generated and when it is
used.

### ARCHER2

For ARCHER2, the `machines/machine_setup.sh` setup and accompanying submission
scripts take care of setting up and copying the depot directory. An unfortunate
side-effect is that `bin/julia` cannot be used on the login nodes. In addition,
ARCHER2 has no node-local disk space. Fortunately the `/tmp/` directory exists,
providing a filesystem that lives in RAM (so is very fast), and is deleted when
the batch job ends (which makes clean-up much simpler in the event a job fails,
etc.).

The process is:
* `machines/machine_setup.sh` creates a depot directory at the path
  `/tmp/$USER/compute-node-temp.julia`. Once precompilation is finished, the
  directory is compressed into an archive `compute-node-temp.julia.tar.bz`,
  which is stored in the top level of the repo.
* The templates for job submission scripts in `machines/archer` include a
  command that uses `sbcast` to copy the .tar.bz into /tmp on each compute node
  (as suggested in
  https://docs.archer2.ac.uk/user-guide/scheduler/#large-jobs).
* The wrapper script `machines/archer/tmp-depot-wrapper.sh` (used within the
  submission script templates) moves the .tar.bz from `/tmp/` to `/tmp/$USER/`,
  and unpacks it, using the 0'th process on each node, while the remaining
  processes wait. Once the job has finished, the 0'th process deletes the
  `/tmp/$USER/` temporary directory - this should not be necessary, as `/tmp/`
  is cleaned up automatically at the end of the job, but is done just in case.

If new dependency packages are installed or if any dependencies are updated, it
is probably best to re-run `machines/machine_setup.sh` so that these packages
get precompiled and saved in `compute-node-temp.julia.tar.bz`.

Since we are already copying the depot into `/tmp/$USER` we also copy across
the `julia` executable and the `moment_kinetics.so` (or `makie_postproc.so` or
`plots_postproc.so`) library file, so that they can be accessed from the
node-local RAM for efficiency, as suggested by
https://docs.archer2.ac.uk/user-guide/scheduler/#large-jobs.
