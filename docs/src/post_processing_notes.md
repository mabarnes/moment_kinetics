Post processing
===============

How to
------

Post processing functionality is provided by the
[`makie_post_processing.makie_post_processing`](@ref) module. To run the post
processing, call [`makie_post_processing.makie_post_process`](@ref) e.g.
```julia
julia> using makie_post_processing
julia> makie_post_process("runs/example-run/")
```
or
```julia
julia> makie_post_process("runs/example-run1/", "runs/example-run2/", "runs/example-run3/")
```

What this function does is controlled by the settings in an input file, by
default `post_processing_input.toml`.

!!! note "Example input file"
    You can generate an example input file, with all the options shown (with
    their default values) but commented out, by running
    `makie_post_processing.generate_example_input_file()`.

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

It is also possible to pass an output file (`*.moments.h5` or `*.dfns.h5`)
instead of a directory. The file name is just used to find the directory and
`run_name` (which is the prefix of the file name), so for example you can pass
a `*.moments.h5` file and ask for distribution function plots (as long as the
corresponding `*.dfns.h5` file exists). This is useful if some output files
were moved to a different directory, or the run directory was renamed (e.g. for
backup, or to compare some different input options or code versions).

To see all the options that can be set,
[`makie_post_processing.generate_example_input_file`](@ref) can be used to
create an example file containing all the options with their default values.
The options are all commented out when the file is created.

!!! note "Viewing animations"
    Animations are produced in `.gif` format. Most utilities just play gifs,
    but provide no options to pause them, etc. One that provides a few more
    features is [multigifview](https://github.com/johnomotani/multigifview)
    (developed by @johnomotani).

Interactive use
---------------

The functions in [`makie_post_processing.makie_post_processing`](@ref) can be
used interactively (or in standalone scripts). To do so, first get the 'info'
for a run (file names, metadata, etc.) using
[`makie_post_processing.get_run_info`](@ref)
```julia
julia> using makie_post_processing
julia> run_info = get_run_info("runs/example-run/")
```
or to load from the distribution functions output file `.dfns.h5`
```julia
julia> run_info_dfns = get_run_info("runs/example-run/"; dfns=true)
```
Settings for post-processing are read from an input file, by default
`post_processing_input.toml` (you can select a different one using the
`setup_input_file` argument to `get_run_info()`). The relevant settings for
interactive use are the default indices (`iz0`, `ivpa0`, etc.) that are used to
select slices for 1D/2D plots and animations. The settings are read by
`setup_makie_post_processing!()` which is called by default as part of
`get_run_info()`. You might want to call it directly if you load both 'moments'
and 'distribution functions' data, to get sensible settings for both at the
same time.

Then you can make 1d or 2d plots, e.g.
```julia
julia> fig1 = plot_vs_z(run_info, "phi")
julia> fig2 = plot_vs_r_t(run_info, "density"; outfile="density_vs_r_t.pdf")
```
using [`makie_post_processing.plot_vs_t`](@ref), etc. for 1d and
[`makie_post_processing.plot_vs_r_t`](@ref), etc. for 2d
plots.
The `outfile` argument can be used to save the plot.
You can also change the default values used to select from the other dimensions
```julia
julia> plot_vs_z(run_info, "phi"; outfile="phi_vs_z.pdf", it=42, ir=7)
```
You can make animations in a similar way
```julia
julia> fig1 = animate_vs_z(run_info, "phi"; outfile="phi_vs_z.gif", it=8:12, ir=1)
julia> fig2 = animate_vs_z_r(run_info, "density"; outfile="density_vs_z_r.mp4")
```
using [`makie_post_processing.animate_vs_r`](@ref), etc. for 1d and
[`makie_post_processing.animate_vs_z_r`](@ref), etc. for 2d
animations.
Note that `outfile` is required for animations.

API
---

See [makie\_post\_processing](@ref).
