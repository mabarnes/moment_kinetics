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
