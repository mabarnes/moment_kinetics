`machine_setup` notes
=====================

The `machines` subdirectory provides scripts to set up Julia and
`moment_kinetics` to run on laptops/desktops or on clusters. If the cluster is
not one of the currently supported machines, then some additional manual setup
is required.

Currently supported:
* "generic-pc" - A generic personal computer (i.e. laptop or desktop machine).
  Set up for interactive use, rather than for submitting jobs to a batch queue.
* "generic-batch" - A generic cluster using a batch queue. Requires some manual setup
  first, see `machines/generic-batch-template/README.md`.
* "archer" - the UK supercomputer [ARCHER2](https://www.archer2.ac.uk/)
* "pitagora" - the EUROfusion supercomputer
  [Pitagora](https://docs.hpc.cineca.it/hpc/pitagora.html)

The usage is described in [Getting started](@ref). This page contains some
extra technical information.

## Notes on some prompts from the script

You will be prompted to enter a location for your `.julia` directory. If you
are installing on a personal computer or on a cluster which allows access to
your home directory from compute nodes, it is fine to leave this as the
default. If not (e.g. on ARCHER2), you need to set a path which is accessible
from the compute nodes.  If you want to create a completely self-contained
install (e.g. for reproducibility or for debugging some dependency conflicts),
you might want to put `.julia` within the `moment_kinetics` directory (i.e.
enter `.julia` at the prompt).

## Defaults for prompts

The default value for each of the settings that the user is prompted for
interactively are first taken from some sensible, machine-specific defaults.
When `machines/machine_setup.sh` is run, the values chosen by the user are
saved in the `[moment_kinetics]` section of `LocalPreferences.toml`, and these
values are used as the defaults next time `machines/machine_setup.sh` is run,
in order to make it easier to re-run the setup, e.g. because some dependencies
need updating, or to change just one or a few settings.

A few settings (which are needed before Julia can be started for the first
time) are saved into hidden files (`.julia_default.txt`,
`.this_machine_name.txt`, and `.julia_directory_default.txt`) instead of into
`LocalPreferences.toml`, to avoid needing to parse a TOML file using `bash`.

## `bin/julia`

A symlink or script is created at `bin/julia` to call the chosen julia
executable. On HPC systems we create a file `julia.env` which must be
`source`'d (to load the right modules, etc.) before `julia` can be used - in
this case `julia.env` can be used to set up any environment variables, etc. so
a symlink is sufficient. On laptops/desktops that will be used interactively,
it is inconvenient to have to remember to `source julia.env`, especially if you
have multiple instances of `moment_kinetics`, so instead the necessary setup
(of the `JULIA_DEPOT_PATH` environment variable, if needed, and a Python venv
if the Plots-based post processing is enabled) is done by making `bin/julia` a
  small bash script, which does that setup and then calls the chosen `julia`
  executable, passing through to it any command line arguments.

## `julia.env`

A `julia.env` file is used on HPC systems to set up the environment (modules
and environment variables). On laptop/desktop systems this would be
inconvenient (especially if there are multiple instances of `moment_kinetics`)
so a `julia.env` is not used for these.

The `julia.env` is created from a template `julia.env` which is located in the
subdirectory of `machines/` for the specific machine being set up.

If you need to run `julia` interactively (for `moment_kinetics`) on a machine
that uses `julia.env`, either run `source julia.env` in each terminal session
where you want to use `moment_kinetics`, or add it to your `.bashrc` (if this
does not conflict with any other projects).

!!! warning
    Note that `julia.env` runs `module purge` to remove any already loaded
    modules (to get a clean environment). It is therefore very likely to
    interfere with other projects.

## Setup of post processing packages

See [Post processing packages](@ref).

## Use of system images

On HPC clusters, creating system images `moment_kinetics.so` and for post
processing `makie_postproc.so` and/or `plots_postproc.so` is required. This is
to avoid (as far as practical) wasting CPU hours doing identical compilation in
large parallel jobs. If you wanted to remove this requirement for some reason
(although this is not recommended), you would need to go to the subdirectory of
`machines/` for the machine you are working on, and edit the
`jobscript-run.template`, `jobscript-restart.template`,
`jobscript-postprocess.template`, and `jobscript-postprocess-plotsjl.template`
files to remove the `-J*.so` argument.  If you do do this, please do not commit
those changes and merge them to the `master` branch of `moment_kinetics`.

## Operations done by `machines/machine_setup.sh`

The convenience script `machine_setup.sh` is provide because the actual setup
happens in multiple stages, with Julia being restarted in between (as this is
required on some machines), and because the script is able to download Julia if
Julia is not already installed.

The steps done by `machines/machine_setp.sh` are:
1. Get the name of the 'machine' ('generic-pc', 'archer', etc.) so that
   machine-dependent setup can be done and machine-specific defaults can be
   used. ()
1. Download a Julia executable, or prompt the user for the location of one
   (defaulting to any `julia` found in `$PATH`).
1. Get the location of the `.julia` directory to be used by (this copy of)
   `moment_kinetics`. At this point we have enough setup to start using
   `julia`. Export `JULIA_DEPOT_PATH` so that this is used any time `julia` is
   run in the rest of the script.
1. Run `machines/shared/machine_setup.jl`. This script (whose functions are
   documented in [API documentation](@ref machine_setup_api_documentation)):
   * prompts the user for most of the settings (and saves them to
     `LocalPreferences.toml` from where they can be accessed by other scripts
     later and used as defaults if `machines/machine_setup` is re-run)
   * creates the `julia.env` file (from the template for the given machine) on
     HPC systems
   * creates the `bin/julia` symlink or script (see [bin/julia](@ref))
   * creates a link to the `compile_dependencies.sh` script for the machine (if
     there is one).
   * installs the `Revise` package and adds `using Revise` to the `startup.jl`
     file (on laptop/desktop systems, and if the user did not de-select this).

   It is necessary to restart `julia` after this script has been run, so that
   we can first `source julia.env` (if it exists) or use the script at
   `bin/julia` in order to use the environment settings in them.
1. If `julia.env` exists, run `source julia.env` to load modules, etc.
1. Set the optimization flags that will be used for running simulations or for
   running post processing. These need to be set the same as will be used
   eventually so that precompilation of dependencies and packages that happens
   while running `machines/machine_setup.sh` does not need to be overwritten
   (due to different optimization flags) later, as this would be a waste of
   time (although it should work fine).
1. Add various dependencies to the top-level project, by calling
   `machines/shared/add_dependencies_to_project.jl`. This will set up MPI and
   HDF5 to link to the correct libraries. `julia` needs to be restarted after
   the setup of MPI and HDF5 is done, which is why this is a separate script
   from the following one (this separation also allows
   `add_dependencies_to_project.jl` to be re-used in
   `makie_post_processing_setup.jl` and `plots_post_processing_setup.jl` if
   these are to be set up as separate projects from the top-level one).
1. Complete the setup by running `machines/shared/machine_setup_stage_two.jl`,
   which creates a Python venv with matplotlib installed (if
   `plots_post_processing` is enabled), creates symlinks at the top level to
   scripts to submit batch jobs (if setting up for an HPC cluster), and submits
   a job to compile a system image for `moment_kinetics` (if setting up for an
   HPC cluster, and if the user did not de-select this).
1. Set up `makie_post_processing` (if enabled) by running
   `machines/shared/makie_post_processing_setup.jl` and/or
   `plots_post_processing (if enabled)` by running
   `machines/shared/plots_post_processing_setup.jl`. These scripts also submit
   jobs to create system images for `makie_post_processing` or
   `plots_post_processing` (if setting up for an HPC cluster, and if the user
   did not de-select this).
1. Print a message indicating which optimization flags to use for running
   simulations or for post-processing.

[API documentation](@id machine_setup_api_documentation)
--------------------------------------------------------

```@autodocs
Modules = [moment_kinetics.machine_setup]
```
