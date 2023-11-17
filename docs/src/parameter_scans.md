Parameter scans
===============

Running a scan
--------------

Parameter scans can be run using the `run_parameter_scan.jl` script. To run from the REPL
```julia
$ julia -p 8 --project -O3
julia> include("run_parameter_scan.jl")
julia> run_parameter_scan("path/to/an/input/file.toml")
```
or to run a single scan from the command line
```shell
$ julia -p 8 --project -O3 run_parameter_scan.jl path/to/an/input/file.toml
```
The `-p 8` argument passed to julia in these examples is optional. It indicates
that julia should use 8 processes for parallelism. In this case we are not
using MPI - each run in the scan is run in serial, but up to 8 (in this
example) runs from the scan can be performed simultaneously (using the
`@distributed` macro).

The runs can use MPI - in this case call julia using `mpirun`, etc. as usual
but do not pass the `-p` argument. Mixing MPI and `@distributed` would cause
oversubscription and slow everything down. The runs will run one after the
other, and each run will be MPI parallelised.

The inputs (see [`moment_kinetics.parameter_scans.get_scan_inputs`](@ref)) can
be passed to the function in a Dict, or read from a TOML file.

`run_parameter_scan` can also be passed a directory (either as an argument to
the function or from the command line), in which case it will perform a run for
every input file contained in that directory.

Post processing a scan
----------------------

[`moment_kinetics.makie_post_processing.makie_post_process`](@ref) can be
called for each run in a scan. For example to post process the scan in
`runs/scan_example` from the REPL
```julia
$ julia -p 8 --project -O3
julia> include("post_process_parameter_scan.jl")
julia> post_process_parameter_scan("runs/scan_example/")
```
or to from the command line
```shell
$ julia -p 8 --project -O3 post_process_parameter_scan.jl runs/scan_example/
```
Again the `-p 8` argument passed to julia in these examples is optional. It
indicates that julia should use 8 processes for parallelism. Each run in the
scan is post-processed in serial, but up to 8 (in this example) runs from the
scan can be post-processed simultaneously (using the `@distributed` macro).

API
---

```@autodocs
Modules = [moment_kinetics.parameter_scans]
```
