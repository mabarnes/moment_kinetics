To use `generic-batch` you must copy `machines/generic-batch-template` to
`machines/generic-batch` and:

* Edit the modules in `machines/generic-batch/julia.env` (see comments in that
  file)
* Edit the `jobscript-*.template` files for precompilation or post-processing
  jobs with the correct serial or debug queue for your machine.
* If you want to use a system-provided HDF5 you can delete
  `machines/generic-batch/compile_dependencies.sh`, and uncomment the
  `hdf5_library_setting = "system"` option in
  `machines/generic-batch/machine_settings.toml`
* If `MPIPreferences.use_system_binary()` cannot auto-detect your MPI library,
  then you may need to set the `mpi_library_names` setting in
  `machines/generic-batch/machine_settings.toml`
* If `MPIPreferences.use_system_binary()` cannot auto-detect your MPI library
  and/or if `mpirun` is not the right command to launch MPI processes, then you
  need to set the `mpi_library_names` and `mpiexec` settings in
  `machines/generic-batch/machine_settings.toml` (note if either of these
  settings is set, then both must be)
* If `mpirun` is not the right command to launch MPI processes, you may need to
  edit the `jobscript-run.template` and `jobscript-restart.template` files in
  `machines/generic-batch/` and set the `mpiexec` setting in
  `machines/generic-batch/machine_settings.toml`

Note that `generic-batch` is set up assuming a Linux, x86\_64 based machine
that uses the 'module' system and a SLURM job queue.
