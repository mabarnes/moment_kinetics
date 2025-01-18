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

## Using the native Julia MPI

As an alternative, you can also use the MPI that is shipped with Julia.
```julia
julia> using MPI; MPI.install_mpiexecjl(force=true)
julia> using MPIPreferences; MPIPreferences.use_jll_binary()
```
The executable for the Julia MPI can be called from the root folder of the project with
```
.julia/bin/mpiexecjl --project=./ -n N julia --project -O3 run_your_script.jl
```
where `N` is the number of cores used.

## Miscellaneous required packages

For full functionality, including precompilation with
```
$ julia --project -O3 precompile.jl
```
and running of tests by
```
$ julia --project -O3 -e 'include("moment_kinetics/test/runtests.jl")'
```
we require to install the following packages
```
$ julia --project -O3
julia> ]
pkg> add PackageCompiler StatsBase SpecialFunctions Test 
```

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
## An example manual setup script

We include an example manual setup script below to show the process
of carrying out the above steps together, for a case where no post processing is required,
and we only desire to verify the install with an MPI test. We use the 
native Julia MPI, and do not link to HDF5.
```
#!/bin/bash
# simple moment_kinetics install script
# not supporting parallel HDF5
# not supporting diagnostics

# first time use, uncomment this
# otherwise, use from the moment_kinetics_install (root) folder 
git clone https://github.com/mabarnes/moment_kinetics.git moment_kinetics_test_install
cd moment_kinetics_test_install

# set up modules and environment variables
# need this everytime you use Julia
# e.g. module load julia/1.10.2
# this will be specific to your system
# JULIA_DEPOT_PATH must be set to be the same for each use of a specific install
export JULIA_DEPOT_PATH=$(pwd)/.julia

# develop moment_kinetics, no plots, no symbolic function tests
touch Project.toml
julia --project -O3 -e 'using Pkg; Pkg.develop(path="./moment_kinetics")'
julia --project -O3 -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project -O3 -e 'using Pkg; Pkg.add("MPI")'
julia --project -O3 -e 'using Pkg; Pkg.add("Test")'
julia --project -O3 -e 'using Pkg; Pkg.add("SpecialFunctions")'
julia --project -O3 -e 'using Pkg; Pkg.add("PackageCompiler")'
julia --project -O3 -e 'using Pkg; Pkg.add("StatsBase")'

# setup MPI preferences and binary
julia --project -O3 -e 'using Pkg; Pkg.instantiate()'
julia --project -O3 -e 'using Pkg; Pkg.resolve()'
julia --project -O3 -e 'using MPI; MPI.install_mpiexecjl(force=true)'
julia --project -O3 -e 'using MPIPreferences; MPIPreferences.use_jll_binary()
julia --project -O3 -e 'using Pkg; Pkg.instantiate()'
julia --project -O3 -e 'using Pkg; Pkg.resolve()'

# generate moment_kinetics.so
julia --project -O3 precompile.jl

# check install with tests
echo "MPI test with precompiled moment_kinetics.so"
.julia/bin/mpiexecjl --project=./ -n 2 julia --project -O3 -Jmoment_kinetics.so -e 'include("moment_kinetics/test/runtests.jl")'
```


