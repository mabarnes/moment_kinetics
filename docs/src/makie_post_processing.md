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

To see all the options that can be set,
[`moment_kinetics.makie_post_processing.generate_example_input_file`](@ref) can
be used to create an example file containing all the options with their default
values. The options are all commented out when the file is created.

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

Then you can make 1d or 2d plots, e.g.
```julia
julia> fig1 = plot_vs_z(run_info, "phi")
julia> fig2 = plot_vs_r_t(run_info, "density"; outfile="density_vs_r_t.pdf")
```
using [`moment_kinetics.makie_post_processing.plot_vs_t`](@ref), etc. for 1d
and [`moment_kinetics.makie_post_processing.plot_vs_r_t`](@ref), etc. for 2d
plots.
The `outfile` argument can be used to save the plot.
You can also change the default values used to select from the other dimensions
```julia
julia> plot_vs_z(run_info, "phi"; outfile="phi_vs_z.pdf", it=42, ir=7)
```

API
---

```@autodocs
Modules = [moment_kinetics.makie_post_processing]
```
