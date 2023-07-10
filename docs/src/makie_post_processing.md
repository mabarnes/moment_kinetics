Post processing
===============

How to
------

Post processing functionality is provided by the
[`moment_kinetics.makie_post_processing`](@ref) module. To run the post
processing, call
[`moment_kinetics.makie_post_processing.makie_post_process`](@ref) e.g.
```julia
julia> using moment_kinetics.makie_post_process
julia> makie_post_process("runs/example-run/")
```
or
```julia
julia> makie_post_process("runs/example-run1/", "runs/example-run2/", "runs/example-run3/")
```

What this function does is controlled by the settings in an input file, by
default `post_processing_input.toml`.

To run from the command line
```julia
julia --project run_makie_post_processing.jl dir1 [dir2 [dir3 ...]]
```

If multiple directories are passed, comparison plots will be made. This usually
means that for line plots and animations the output for all the runs will be
drawn on the same plot. For heatmap plots, the runs will be plotted side by
side.

If there is output from several restarts of the same run in a directory, by
default they will all be read and plotted. A single restart can be started by
passing the `restart_id` argument to `makie_post_process()`.

Interactive use
---------------

The functions in [`moment_kinetics.makie_post_processing`](@ref) can be used
interactively (or in standalone scripts). To do so, first get the 'info' for
a run (file names, metadata, etc.) using
[`moment_kinetics.makie_post_processing.get_run_info`](@ref)
```julia
julia> using moment_kinetics.makie_post_processing
julia> run_info = get_run_info("runs/example-run/")
```
or to load from the distribution functions output file `.dfns.h5`
```julia
julia> run_info_dfns = get_run_info("runs/example-run/"; dfns=true)
```
You will usually want to set up the options (stored in
[`moment_kinetics.makie_post_processing.input_dict`](@ref) and
[`moment_kinetics.makie_post_processing.input_dict_dfns`](@ref)) with
[`moment_kinetics.makie_post_processing.setup_makie_post_processing_input!()`](@ref)
```
julia> setup_makie_post_processing_input!("my_input.toml"; run_info_moments=run_info, run_info_dfns=run_info_dfns)
```
The `run_info_moments` and `run_info_dfns` arguments are used to set sensible
defaults for various options - they are not required, and usually you will
probably only pass one (`run_info_dfns` if you loaded distribution function
output, and `run_info_moments` otherwise).

API
---

```@autodocs
Modules = [moment_kinetics.makie_post_processing]
```
