# Manual setup

If you want or need to set up `moment_kinetics` without using
`machines/machine_setup.sh`, you will need to follow at least the steps in the
following sections.

## Install Julia

Download Julia from <https://julialang.org/downloads/>, and add it to your
`$PATH` so you can execute it from the command line.

## Add `moment_kinetics` packages

Create a 'project' in the top-level directory by creating an empty
`Project.toml` file, e.g.
```
$ touch Project.toml
```
Without this `Project.toml` file, running `julia --project` will activate a
global project, not one linked specificially to the repo you are in, which is
likely to cause confusion if you have more than one copy of the
`moment_kinetics` repo (experience suggests you are likely to end up with
multiple copies eventually!).

To add the `moment_kinetics` package to your project, start Julia, enter
'Package mode' by pressing ']' at the prompt and use `develop` (to exit
'Package mode' and return to the usual `julia>` prompt, press backspace):
```julia
$ julia --project
julia> ]
pkg> develop ./moment_kinetics/
```
To allow post-processing add one or both of the post processing packages
```julia
pkg> develop ./makie_post_processing/makie_post_processing/
```
and/or
```julia
pkg> develop ./plots_post_processing/plots_post_processing/
```

To use the `run_moment_kinetics.jl` script, you will need to install `MPI`
into the top-level project
```julia
pkg> add MPI
```

## Set up MPI

You probably want to use your system's MPI rather than a Julia-provided
version. To do this add (in 'Package mode') the `MPIPreferences` package
([documentation
here](https://juliaparallel.org/MPI.jl/stable/configuration/#using_system_mpi))
and then use its `use_system_binary()` function.
```julia
pkg> add MPIPreferences
pkg> <press 'backspace'>
julia> using MPIPreferences
julia> MPIPreferences.use_system_binary()
```
Normally this should 'just work'. Sometimes, for example if the MPI library
file is named something other than `libmpi.so`, you might have to pass some
keyword arguments to `use_system_binary()` - see
<https://juliaparallel.org/MPI.jl/stable/reference/mpipreferences/#MPIPreferences.use_system_binary>.

## Link HDF5

To enable parallel I/O, you need to get HDF5.jl to use the system HDF5 library
(which must be MPI-enabled and compiled using the same MPI as you run Julia
with). To do this (see [the HDF5.jl
docs](https://juliaio.github.io/HDF5.jl/stable/#Using-custom-or-system-provided-HDF5-binaries))
add the `HDF5` package and use its `HDF5.API.set_libraries!()` function
```julia
pkg> add HDF5
pkg> <press backspace>
julia> using HDF5
julia> HDF5.API.set_libraries!("/path/to/your/hdf5/directory/libhdf5.so", "/path/to/your/hdf5/directory/libhdf5_hl.so")
```
JTO also found that (on a Linux laptop) it was necessary to compile HDF5 from
source. The system-provided, MPI-linked libhdf5 depended on libcurl, and Julia
links to an incompatible libcurl, causing an error. When compiled from source
(enabling MPI!), HDF5 does not require libcurl (guess it is an optional
dependency), avoiding the problem.

## Enable MMS features

To enable the "method of manufactured solutions" (MMS) features, install the
`Symbolics` package (for more explanation, see [Optional dependencies](@ref))
```julia
pkg> add Symbolics
```

## Enable NetCDF output

If you want the option to output to NetCDF instead of HDF5, install the
`NCDatasets` package (for more explanation, see [Optional dependencies](@ref))
```julia
pkg> add NCDatasets
```

## Set up `Plots`-based plotting routines

The `plots_post_processing` package has some functions that have to use `PyPlot`
directly to access features not available through the `Plots` wrapper. This
means that Julia has to be able to access an instance of Python which has
`matplotlib` available. If you are going to use `plots_post_processing` you may
well want to use the same Python that you use outside Julia (e.g. a
system-provided Python) - to do so:
* Check that `matplotlib` is installed, e.g. check that
  ```python
  $ python
  >>> import matplotlib
  ```
  completes without an error. If not, install matplotlib, for example with a
  command like
  ```shell
  pip install --user matplotlib
  ```
* Set up Julia to use your chosen Python
  ```julia
  $ which python
  /your/python/location
  $ julia -O3 --project
  julia> ENV["PYTHON"]="/your/python/location"
  julia> using Pkg; Pkg.build("PyCall")
  ```
