# Developing


## Dependencies

If you need to add a dependency, start the REPL with the `moment_kinetics` package activated (see [above](#moment_kinetics)), enter `pkg>` mode (press `]`) and then to add, for example, the `FFTW.jl` package enter
```
(moment_kinetics) pkg> add FFTW
```
This should take care of adding the package (`FFTW`) to the `Project.toml` and `Manifest.toml` files.


## Revise.jl

When working on the code, one way to avoid waiting for everything to recompile frequently is to load the Revise.jl package
```
julia> using Revise
```
`Revise.jl` will recompile each edited function/method as needed, so it is possible to keep a REPL session open and avoid long recompilation. `moment_kinetics` can be run fairly conveniently from the REPL
```
julia> using moment_kinetics
julia> run_moment_kinetics(input)
```
where `input` is a `Dict()` containing any non-default options desired. Input can also be loaded from a TOML file passing the filaname as a String to the second argument, e.g.
```
julia> run_moment_kinetics("input.toml")
```
It might be convenient to add `using Revise` to your `startup.jl` file (`~/julia/config/startup.jl`) so it's always loaded.


## Input options and defaults

The input is read from a `.toml` file. It is also written to the output HDF5
(or NetCDF) file, after all defaults are applied, both as a TOML-formatted
String and as a tree of HDF5 variables.

!!! warning
    Neither TOML nor HDF5 have a 'null' type, so there is no convenient way to
    store Julia's `nothing` when writing to TOML or HDF5.  Therefore `nothing`
    should not be used as a default for any input option. If the code should
    use `nothing` as a default for some setting, that is fine, but must be done
    after the input is read, and not stored in the `input_dict`.

!!! warning "Parallel I/O consistency"
    To ensure consistency between all MPI ranks in the order of reads and/or
    writes when using Parallel I/O, all dictionary types used to store options
    must be either `OrderedDict` or `SortedDict`, so that their order of
    entries is deterministic (which is not the case for `Dict`, which instead
    optimises for look-up speed). This should mostly be taken care of by using
    `moment_kinetics`'s `OptionsDict` type (which is an alias for
    `OrderedDict{String,Any}`). We also need to sort the input after it is read
    by `TOML`, which is taken care of by
    [`moment_kinetics.input_structs.convert_to_sorted_nested_OptionsDict`](@ref).
    See also [Parallel I/O](@ref parallel_io_section).


## Array types

Most arrays in `moment_kinetics` are declared using a custom array type
[`moment_kinetics.communication.MPISharedArray`](@ref). Most of the time this
type is just an alias for `Array`, and so it needs the same template parameters
(see [Julia's Array
documentation](https://docs.julialang.org/en/v1/manual/arrays/)) - the data
type and the number of dimensions, e.g. `MPISharedArray{mk_float,3}`. Although
these arrays use shared memory, Julia does not know about this. We use
`MPI.Win_allocate_shared()` to allocate the shared memory, then wrap it in an
`Array` in [`moment_kinetics.communication.allocate_shared`](@ref).

The reason for using the alias, is that when the shared-memory debugging mode
is activated, we instead create arrays using a type `DebugMPISharedArray`,
which allows us to track some debugging information along with the array, see
[Shared memory debugging](@ref), and make `MPISharedArray` an alias for
`DebugMPISharedArray` instead. The reason for the alias is that if we declared
our structs with just `Array` type, then when debugging is activated we would
not be able to store `DebugMPISharedArray` instances in those structs, and if
we declared the structs with `AbstractArray`, they would not be concretely
typed, which could impact performance by creating code that is not 'type
stable' (i.e. all concrete types are known at compile time).


## Timings

Checking the timings of different parts of the code can be useful to check that
performance problems are not introduced. Excessive allocations can also be a
sign of type instability (or other problems) that could impact performance. To
monitor these things, `moment_kinetics` uses a `TimerOutput` object
[`moment_kinetics.timer_utils.global_timer`](@ref).

The timings and allocation counts from the rank-0 MPI process are printed to
the terminal at the end of a run. The same information is also saved to the
output file as a string for quick reference - one way to view this is
```bash
$ h5dump -d /timing_data/global_timer_string my_output_file.moments.h5
```

More detailed timing information is saved for each MPI rank into subgroups
`rank<i>` of the `timing_data` group in the output file. This information can
be plotted using [`makie_post_processing.timing_data`](@ref). The plots contain
many curves. Filtering out the ones you are not interested in (using the
`include_patterns`, `exclude_patterns`, and/or `ranks` arguments) can help, but
it still may be useful to have interactive plots which show the label and MPI
rank when you hover over a curve. For example
```julia
julia> using makie_post_processing, GLMakie
julia> ri = get_run_info("runs/my_example_run/")
julia> timing_data(ri; interactive_figs=:times);
```
Here `using GLMakie` selects the `Makie` backend that provides interactive
plots, and the `interactive_figs` argument specifies that `timing_data()`
should make an interactive plot (in this case for the execution times).

Lower level timing data, for example timing MPI and linear-algebra calls, can
be enabled by activating 'debug timing'. This can be done by re-defining the
function [`moment_kinetics.timer_utils.timeit_debug_enabled`](@ref) to return
`true` - not the most user-friendly interface (!) but this feature is probably
only needed while developing/profiling/debugging.


## Parallelization

The code is parallelized at the moment using MPI and shared-memory arrays. Arrays representing the pdf, moments, etc. are shared between all processes. Using shared memory means, for example, we can take derivatives along one dimension while parallelising the other for any dimension without having to communicate to re-distribute the arrays. Using shared memory instead of (in future as well as) distributed memory parallelism has the advantage that it is easier to split up the points within each element between processors, giving a finer-grained parallelism which should let the code use larger numbers of processors efficiently.

It is possible to use a REPL workflow with parallel code:
* Recommended option is to use [tmpi](https://github.com/Azrael3000/tmpi). This utility (it's a bash script that uses `tmux`) starts an mpi program with each process in a separate pane in a single terminal, and mirrors input to all processes simultaneously (which is normally what you want, there are also commands to 'zoom in' on a single process).
* Another 'low-tech' possibilty is to use something like `mpirun -np 4 xterm -e julia --project`, but that will start each process in a separate xterm and you would have to enter commands separately in each one. Occasionally useful for debugging when nothing else is available.

There is no restriction on the number of processes or number of grid points, although load-balancing may be affected - if there are only very few points per process, and a small fraction of processes have an extra grid point (e.g. splitting 5 points over 4 processes, so 3 process have 1 point but 1 process has 2 points), many processes will spend time waiting for the few with an extra point.

Parallelism is implemented through macros that get the local ranges of points that each process should handle. The inner-most level of nested loops is typically not parallelized, to allow efficient FFTs for derivatives, etc. A loop over one (possibly parallelized) dimension can be written as, for example,
```
@loop_s is begin
    f[is] = ...
end
```
These macros can be nested as needed for relatively complex loops
```
@loop_s is begin
    some_setup(is)
    @loop_z iz begin
        @views do_something(f[:,iz,is])
    end
    @loop_z iz begin
        @views do_something_else(f[:,iz,is])
    end
end
```
Simpler nested loops can (optionally) be written more compactly
```
@loop_s_z_vpa is iz ivpa begin
    f[ivpa,iz,is] = ...
end
```
Which dimensions are actually parallelized by these macros is controlled by the 'region' that the code is currently in, as set by the `begin_<dims>_region()` functions, where <dims> are the dimensions that will be parallelized in the following region. For example, after calling `begin_s_z_region()` loops over species and z will be divided up over the processes in a 'block' (currently there is only one block, which contains the whole grid and all the processes being used, as we have not yet implemented distributed-memory parallelism). Every process will loop over all points in the remaining dimensions if the loop macros for those dimensions are called.
* The recommended place to put `begin_*_region()` calls is at the beginning of a function whose contents should use loops parallelised according to the settings for that region.
    * Each `begin_*_region()` function checks if the region it would set is already active, and if so returns immediately (doing nothing). This means that `begin_*_region()` can (and should) be used to mark a block of code as belonging to that region, and if `moment_kinetics` is already in that region type, the call will have essentially zero cost.
    * In some places it may be necessary to change the region type half way through a function, etc. This is fine.
    * When choosing which region type to select, note that all 'parallelised dimensions' must be looped over for each operation (otherwise some points may be written more than once), unless some special handling is used (e.g. species dimension `s` is parallelised, but a conditional like `if 1 in loop_ranges[].s` is wrapped around code to be executed so that only processes which should handle the point at `s=1` do anything). It may be more optimal in some places to choose region types that do not parallelise all possible dimensions, to reduce the number of synchronisations that are needed.
    * As a matter of style, it is recommended to place `begin_*_region()` calls within functions where the loops are (or at most one level above), so that it is not necessary to search back along the execution path of the code to find the most recent `begin_*_region()` call, and therefore know what region type is active.
* In a region after `begin_serial_region()`, the rank 0 process in each block will loop over all points in every dimension, and all other ranks will not loop over any.
* Inside serial regions, the macro `@serial_region` can also be used to wrap blocks of code so that they only run on rank 0 of the block. This is useful for example to allow the use of array-broadcast expressions during initialization where performance is not critical.
* To help show how these macros work, a script is provided that print a set of examples where the loop macros are expanded. It can be run from the Julia REPL
  ```
  $ julia --project
                 _
     _       _ _(_)_     |  Documentation: https://docs.julialang.org
    (_)     | (_) (_)    |
     _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
    | | | | | | |/ _` |  |
    | | |_| | | | (_| |  |  Version 1.7.0 (2021-11-30)
   _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
  |__/                   |

  julia> include("util/print-macros.jl")
  ```
  or on the command line
  ```
  $ julia --project util/print-macros.jl
  ```

The ranges used are stored in a `LoopRanges` struct in the `Ref` variable `loop_ranges` (which is exported by the `looping` module). The range for each dimension is stored in a member with the same name as the dimension, e.g. `loop_ranges[].s` for the species. Occasionally it is useful to access the range directly. There are different `LoopRanges` instances for different parallelization patterns - the instance stored in `loop_ranges` is updated when `begin_*_region()` is called. It is possible to find out the current region type (i.e. which dimensions are being parallelized) by looking at `loop_ranges[].parallel_dims`.

!!! note

    The square brackets `[]` after `loop_ranges[]` are needed because `loop_ranges` is a reference to a `LoopRanges` object `Ref{LoopRanges}` (a bit like a pointer) - it allows `loop_ranges` to be a `const` variable, so its type is always known at compile time, but the actual `LoopRanges` can be set/modified at run-time.

It is also possible to run a block of code in serial (on just the rank-0 member of each block of processes) by wrapping it in a `@serial_region` macro. This is mostly useful for initialization or file I/O where performance is not critical. For example
```
@serial_region begin
    # Do some initialization
    f .= 0.0
end
```

Internally, when the `begin_*_region()` functions need to change the region type (i.e. the requested region is not already active), they call `_block_synchronize()`, which calls `MPI.Barrier()`. They also switch over the `LoopRanges` struct contained in `looping.loop_ranges` as noted above. For optimization, the `_block_synchronize()` call can be skipped - when it is correct to do so - by passing the argument `no_synchronize=true` (or some more complicated conditional expression if synchronization is necessary when using some options but not for others).

### Collision operator and `anyv` region

The Fokker-Planck collision operator requires a special approach to
shared-memory parallelisation. There is an outer loop over spatial points (and
potentially over species). Inside that outer loop there are operations that can
benefit from parallelisation over $v_{\perp}$, or over $v_{\parallel}$, or over
both $v_{\perp}$ and $v_{\parallel}$, as well as some that do not parallelise
over velocity space at all. To deal with this, it is beneficial to parallelise
the outer loop over species and spatial dimensions as much as possible, and
then within that allow changes between different ways of parallelizing over
velocity space.

The mechanism introduced to allow the type of parallelization just described is
the 'anyv' (read any-$v$) region. Before the outer loop of the collision
operator `begin_s_r_z_anyv_region()` is used to start the 'anyv'
parallelization. Then within the `@loop is ir iz begin...` the functions
`begin_anyv_region()` (for no parallelization over velocity space),
`begin_anyv_vperp_region()`, `begin_anyv_vpa_region()` and
`begin_anyv_vperp_vpa_region()` can be used to parallelize over neither
velocity space dimension, either velocity space dimension individually, or over
both velocity space dimensions together. This is possible because 'subblocks'
of processes are defined. Each subblock shares the same range of species and
spatial indices, which stay the same throughout the `begin_s_r_z_anyv_region()`
section, and are not shared with any other subblock of processes. Because the
subblock has an independent set of species- and spatial-indices, when changing
the velocity-space parallelization only the processes in the sub-block need to
be synchronized which is done by
[`moment_kinetics.communication._anyv_subblock_synchronize`](@ref), which is
called when necessary within the `begin_anyv*_region()` functions (the whole
shared-memory block does not need to be synchronized at once, as would be done
by [`moment_kinetics.communication._block_synchronize`](@ref)). The processes
that share an anyv subblock are all part of the `comm_anyv_subblock[]`
communicator (which is a subset of the processes in the full block, whose
communicator is `comm_block[]`).

See also notes on debugging the 'anyv' parallelisation: [Collision operator and
'anyv' region](@ref).

## Bounds checking

For best performance (i.e. 'production' runs), it is important that bounds
checks not be included on array accesses. It should be possible to do this by
running `julia` with the flag `--check-bounds=no`, but this flag has negative
effects on the core Julia code and compiler, and works less well in Julia
versions 1.10 and 1.11. As a workaround/alternative, the `@loop_*` macros
described in the previous section wrap the contained code with an `@inbounds`
macro (which disables bounds checks within the block, but the effect of
`@inbounds` does not propagate down into functions called within the block). If
performance-critical code that you write is within an `@loop`, then you do not
need to do anything. However if it is not within an `@loop`, then you should
add `@inbounds begin ... end` around any performance critical code. You can see
examples of this being done in
[`moment_kinetics.fokker_planck_calculus`](@ref).

## [Parallel I/O](@id parallel_io_section)

The code provides an option to use parallel I/O, which allows all output to be
written to a single output file even when using distributed-MPI parallelism -
this is the default option when the linked HDF5 library is compiled with
parallel-I/O support.

There are a few things to be aware of to ensure parallel I/O works correctly:
* Some operations have to be called simultaneously on all the MPI ranks that
  have the output file open. Roughly, these are any operations that change the
  'metadata' of the file, for example opening/closing files, creating
  variables, extending dimensions of variables, changing attributes of
  variables. Reading or writing the data from a variable does not have to be
  done collectively - actually when we write data we ensure that every rank
  that is writing writes a non-overlapping slice of the array to avoid
  contention that could slow down the I/O (because one rank has to wait for
  another) and to avoid slight inconsistencies because it is uncertain which
  rank writes the data last. For more details see the [HDF5.jl
  documentation](https://juliaio.github.io/HDF5.jl/stable/mpi/#Reading-and-writing-data-in-parallel)
  and the [HDF5
  documentation](https://support.hdfgroup.org/archive/support/HDF5/doc/RM/CollectiveCalls.html).
* One important subtlety is that the `Dict` type does not guarantee a
  deterministic order of entries. When you iterate over a `Dict`, you can get
  the results in a different order at different times or on different MPI
  ranks. If we iterated over a `Dict` to create variables to write to an output
  file, or to read from a file, then different MPI ranks might (sometimes) get
  the variables in a different order, causing errors. We therefore use either
  `OrderedDict` or `SortedDict` types for anything that might be written to or
  read from an HDF5 file.

If the collective operations are not done perfectly consistently, the errors
can be extremely non-obvious. The inconsistent operations may appear to execute
correctly, for example because the same number of variables are created, and
the metadata may only actually be written from the rank-0 process, but the
inconsistency may cause errors later. [JTO, 3/11/2024: my best guess as to the
reason for this is that it puts HDF5's 'metadata cache' in inconsistent states
on different ranks, and this means that at some later time the ranks will cycle
some metadata out of the cache in different orders, and then some ranks will be
able to get the metadata from the cache, while others have to read it from the
file. The reading from the file requires some collective MPI call, which is
only called from some ranks and not others, causing the code to hang.]

## Package structure

The structure of the packages in the `moment_kinetics` repo is set up so that
some features, which depend on 'heavy' external packages (such as `Makie`,
`Plots`, and `Symbolics`, which take a long time to precompile and load) can be
optional.

The structure is set up by the `machines/machine_setup.sh` script, which
prompts the user for input to decide which optional components to include (as
well as some settings related to batch job submission on HPC clusters).
`machine_setup.sh` calls several other scripts to do the setup (written as far
as possible in Julia). The structure of these scripts is explained in
[`machine_setup` notes](@ref).

The intention is that a top-level 'project' (defined by a `Project.toml` file,
which is created and populated by `machines/machine_setup.sh`) is set up in the
top-level directory of the repository. The `moment_kinetics` package itself
(which is in the `moment_kinetics/` subdirectory, defined by its own
`Project.toml` file which is tracked by git), and optionally other
post-processing packages, are added to this top-level project using
`Pkg.develop()`.

### Optional dependencies

Some capabilities that require optional dependencies are provided using
'package extensions' ([a new feature of Julia in
v1.9.0](https://julialang.org/blog/2023/04/julia-1.9-highlights/#package_extensions)).

The way we use package extensions is a bit of a hack. Extensions are intended
to be activated when an optional dependency (called a 'weakdep' by Julia) is
loaded, e.g. `using moment_kinetics, NCDatasets`. This usage pattern is not the
most convenient for the way we use `moment_kinetics` where we would rather just
load `moment_kinetics` and then specify for example `binary_format = "netcdf"`
in the input TOML file. To work around this, the optional dependencies are
loaded automatically if they are installed (by calling `Base.requires()` in the
`__init__()` function of an appropriate sub-module). This is not the way
package extensions were intended to be used, and it may be a bit fragile - at
the time of writing in January 2024 there would be an error on precompilation
if the optional dependencies were added in one order, which went away when the
order was reversed. If this causes problems, we might need to consider an
alternative, for example adding the optional dependencies to the `startup.jl`
file, instead of trying to auto-load them from within the `moment_kinetics`
package.

The optional capabilities at the moment are:
* Method of manufactured solutions (MMS) testing - this requires the
  `Symbolics` package which is heavy and has a large number of dependencies. It
  is convenient not to require `Symbolics` when MMS capability is not being
  used. The functionality is provided by the `manufactured_solns_ext`
  extension. The extension also requires the `IfElse` package, which is not
  needed elsewhere in `moment_kinetics` and so is included as a 'weakdep'
  although `IfElse` is not a heavy dependency.
* NetCDF output - this requires the `NCDatasets` package. Although not as heavy
  as `Symbolics` or the plotting packages, NetCDF output is not required and
  not used by default, so it does not hurt to make the dependency optional. As
  a bonus, importing `NCDatasets` can sometimes cause linking errors when a
  local or system installation of HDF5 (i.e. one not provided by the Julia
  package manager) is used, as `NCDatasets` (sometimes?) seems to try to link a
  different version of the library. These errors can be avoided by not enabling
  NetCDF outut (when HDF5 output is preferred), or allowing Julia to use the
  HDF5 library provided by its package manager (when NetCDF is preferred,
  although this would mean that parallel I/O functionality is not available).

### Post processing packages

Post processing functionality is provided by separate packages
(`makie_post_processing` and `plots_post_processing`) rather than by
extensions. Extensions are not allowed to define new modules, functions, etc.
within the main package, they can only add new methods (i.e. new
implementations of the function for a different number of arguments, or
different types of the arguments) to functions already defined in the main
package. For post-processing, we want to add a lot of new functions, so to use
extensions instead of separate packages we would need to define all the
function names in the main package, and then separately the implementations in
the extension, which would be inconvenient and harder to maintain.

There are two suggested ways of setting up the post-processing packages:

1. For interactive use/development on a local machine, one or both
   post-processing packages can be added to the top-level project using
   `Pkg.develop()`. This is convenient as there is only one project to deal
   with. Both simulations and post-processing are run using
   ```
   $ bin/julia --project -O3 <...>
   ```
2. For optimized use on an HPC cluster it is better to set up a separate
   project for the post-processing package(s). This allows different
   optimization flags to be used for running simulations (`-O3
   --check-bounds=no`) and for post-processing (`-O3`). [Note, in particular
   Makie.jl can have performance problems if run with `--check-bounds=no`, see
   [here](https://github.com/MakieOrg/Makie.jl/issues/3132).] Simulations
   should be run with
   ```
   $ bin/julia --project -O3 --check-bounds=no <...>
   ```
   and post-processing with
   ```
   $ bin/julia --project=makie_post_processing -O3 <...>
   ```
   or
   ```
   $ bin/julia --project=plots_post_processing -O3 <...>
   ```
   This option can also be used on a local machine, if you want to optimise
   your simulation runs as much as possible by using the `--check-bounds=no`
   flag. To do this answer `y` to the prompt "Would you like to set up separate
   packages for post processing..." from `machines/machine_setup.sh`.

To support option 2, the post-processing packages are located in
sub-sub-directories (`makie_post_processing/makie_post_processing/` and
`plots_post_processing/plots_post_processing/`), so that the separate projects
can be created in the sub-directories (`makie_post_processing/` and
`plots_post_processing`). `moment_kinetics` and the other dependencies must
also be added to the separate projects (the `machine_setup.sh` script takes
care of this).
