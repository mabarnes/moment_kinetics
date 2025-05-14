Introduction
============

The inputs and scripts in this directory were used to produce the figures for
the 1D1V benchmarking paper. This README explains how to reproduce the runs and
plots. Note that the inputs in this directory are not tested automatically, so
they may be broken by future updates to `moment_kinetics`.  They were tested
using ???, using julia-1.11.3 and the Manifest.toml file that was temporarily
committed in ???.

Sound wave
==========

Runs
----

All the runs for the sound-wave plots can be done quickly in serial. The
parameter scan functionality allows multiple processes to be used to launch
several runs simultaneously (using Distributed.jl rather than MPI). Here we use
8 processes as an example.

In the top-level `moment_kinetics` directory, run
```julia
$ julia --project -p 8
julia> include("publication_inputs/1D1V-paper/sound-wave/run-all-scans.jl")
```

Plots
-----

Note the steps here should be run after launching `julia` in the directory
containing this README.md, not in the top level `moment_kinetics` directory.

### Setup

The analysis script in this directory uses James Cook's
PlasmaDispersionFunctions.jl package, as this is more accurate than a naive
implementation of the plasma dispersion function using SpecialFuncions.jl.

We also have to add the `makie_post_processing` package from under the top
level, which is `../..` relative to this directory.

To set everything up, do
```julia
$ julia --project
julia>]
(1D1V-paper) pkg> add https://github.com/jwscook/PlasmaDispersionFunctions.jl
(1D1V-paper) pkg> dev ../../makie_post_processing/makie_post_processing/
(1D1V-paper) pkg>^D
$
```

You might also need to repeat any setup steps for MPI and HDF5 (or just copy
the `LocalPreferences.toml` file from the top-level `moment_kinetics`
directory).

### Usage

Import the script, do some post-processing, and run `make_plots()` (we again
use `-p 8` as the `post_process_all_scans()` step can run post processing for
several directories simultaneously, although `make_plots()` is serial)
```
$ julia --project -p 8
julia> include("sound-wave/post_process_scans.jl")
julia> post_process_all_scans()
julia> include("sound-wave/plot_dispersion_relation.jl")
julia> make_plots()
```

Wall BC
=======

Runs
----

[You might want to use MPI to speed up the runs, see the main `moment_kinetics`
documentation.]

We run the 'full-f' version from scratch, then restart the moment-kinetic
version from that to save time.

In the top-level `moment_kinetics` directory, run
```julia
$ julia --project
julia> using moment_kinetics
julia> run_moment_kinetics("publication_inputs/1D1V-paper/wall-bc/wall-bc_recyclefraction0.5.toml")
julia> run_moment_kinetics("publication_inputs/1D1V-paper/wall-bc/wall-bc_recyclefraction0.5_split1.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
julia> run_moment_kinetics("publication_inputs/1D1V-paper/wall-bc/wall-bc_recyclefraction0.5_split2.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
julia> run_moment_kinetics("publication_inputs/1D1V-paper/wall-bc/wall-bc_recyclefraction0.5_split3.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
```

Plots
-----

In the directory containing this README.md (after already doing the setup from
the 'Sound wave' section) run
```julia
$ julia --project
julia> include("wall-bc/wall-bc-plots.jl")
```
