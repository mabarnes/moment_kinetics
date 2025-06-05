using moment_kinetics.input_structs
using moment_kinetics.load_data: close_run_info, get_run_info_no_setup

using TOML

"""
    makie_post_process(run_dir...;
                       input_file::String=default_input_file_name,
                       restart_index::Union{Nothing,mk_int,AbstractVector}=nothing,
                       plot_prefix::Union{Nothing,AbstractString}=nothing)

Run post processing with input read from a TOML file

`run_dir...` is the path to the directory to plot from. If more than one `run_dir` is
given, plots comparing the runs in `run_dir...` are made.
A moment_kinetics binary output file can also be passed as `run_dir`, in which case the
filename is only used to infer the directory and `run_name`, so it is possible for example
to pass a `.moments.h5` output file and still make distribution function plots (as long as
the corresponding `.dfns.h5` file exists).

`restart_index` specifies which restart to read if there are multiple restarts. The
default (`nothing`) reads all restarts and concatenates them. An integer value reads the
restart with that index - `-1` indicates the latest restart (which does not have an
index). A Vector with the same length as `run_dir` can also be passed to give a different
`restart_index` for each run.

`plot_prefix` can be specified to give the prefix (directory and first part of file name)
to use when saving plots/animations. By default the run directory and run name are used if
there is only one run, and "comparison_plots/compare_" is used if there are multiple runs.

If `input_file` does not exist, prints warning and uses default options.
"""
function makie_post_process(run_dir...;
                            input_file::String=default_input_file_name,
                            restart_index::Union{Nothing,mk_int,AbstractVector}=nothing,
                            plot_prefix::Union{Nothing,AbstractString}=nothing)
    if isfile(input_file)
        new_input_dict = TOML.parsefile(input_file)
    else
        println("Warning: $input_file does not exist, using default post-processing "
                * "options")
        new_input_dict = OrderedDict{String,Any}()
    end

    return makie_post_process([run_dir...], new_input_dict; restart_index=restart_index,
                              plot_prefix=plot_prefix)
end

"""
    makie_post_process(run_dir::Union{String,Vector{String}},
                       new_input_dict::Dict{String,Any};
                       restart_index::Union{Nothing,mk_int,AbstractVector}=nothing,
                       plot_prefix::Union{Nothing,AbstractString}=nothing)

Run post prossing, with (non-default) input given in a Dict

`run_dir` is the path to the directory to plot from. If `run_dir` is a `Vector{String}`,
plots comparing the runs in `run_dir` are made.
A moment_kinetics binary output file can also be passed as `run_dir`, in which case the
filename is only used to infer the directory and `run_name`, so it is possible for example
to pass a `.moments.h5` output file and still make distribution function plots (as long as
the corresponding `.dfns.h5` file exists).

`input_dict` is a dictionary containing settings for the post-processing.

`restart_index` specifies which restart to read if there are multiple restarts. The
default (`nothing`) reads all restarts and concatenates them. An integer value reads the
restart with that index - `-1` indicates the latest restart (which does not have an
index). A Vector with the same length as `run_dir` can also be passed to give a different
`restart_index` for each run.

`plot_prefix` can be specified to give the prefix (directory and first part of file name)
to use when saving plots/animations. By default the run directory and run name are used if
there is only one run, and "comparison_plots/compare_" is used if there are multiple runs.
"""
function makie_post_process(run_dir::Union{String,Vector{String}},
                            new_input_dict::AbstractDict{String,Any};
                            restart_index::Union{Nothing,mk_int,AbstractVector}=nothing,
                            plot_prefix::Union{Nothing,AbstractString}=nothing)
    if isa(run_dir, String)
        # Make run_dir a one-element tuple if it is not a tuple
        run_dir = [run_dir]
    end
    # Normalise by removing any trailing slashes - with a slash basename() would return an
    # empty string
    run_dir = [rstrip(ri, '/') for ri ∈ run_dir]

    new_input_dict = convert_to_OrderedDicts!(new_input_dict)

    if !isa(restart_index, AbstractVector)
        # Convert scalar restart_index to Vector so we can treat everything the same below
        restart_index = [restart_index for _ ∈ run_dir]
    end

    # Special handling for itime_* and itime_*_dfns because they are needed in order to
    # set up `time` and `time_dfns` in run_info, but run_info is needed to set several
    # other default values in setup_makie_post_processing_input!().
    itime_min = get(new_input_dict, "itime_min", 1)
    itime_max = get(new_input_dict, "itime_max", 0)
    itime_skip = get(new_input_dict, "itime_skip", 1)
    itime_min_dfns = get(new_input_dict, "itime_min_dfns", 1)
    itime_max_dfns = get(new_input_dict, "itime_max_dfns", 0)
    itime_skip_dfns = get(new_input_dict, "itime_skip_dfns", 1)
    run_info_moments = get_run_info(zip(run_dir, restart_index)..., itime_min=itime_min,
                                    itime_max=itime_max, itime_skip=itime_skip,
                                    do_setup=false)
    if !isa(run_info_moments, AbstractVector)
        run_info_moments = Any[run_info_moments]
    end
    run_info_dfns = get_run_info(zip(run_dir, restart_index)..., itime_min=itime_min_dfns,
                                 itime_max=itime_max_dfns, itime_skip=itime_skip_dfns,
                                 dfns=true, do_setup=false)
    if !isa(run_info_dfns, AbstractVector)
        run_info_dfns = Any[run_info_dfns]
    end

    if all(ri === nothing for ri in (run_info_moments..., run_info_dfns...))
        error("No output files found for either moments or dfns in $run_dir")
    end
    setup_makie_post_processing_input!(new_input_dict, run_info_moments=run_info_moments,
                                       run_info_dfns=run_info_dfns)

    has_rdim = any(ri !== nothing && ri.r.n > 1 for ri ∈ run_info_moments)
    has_zdim = any(ri !== nothing && ri.z.n > 1 for ri ∈ run_info_moments)

    # Only plot electron stuff if some runs have electrons
    if any(ri !== nothing for ri ∈ run_info_moments)
        has_electrons = any(r.composition.electron_physics
                            ∈ (braginskii_fluid, kinetic_electrons,
                               kinetic_electrons_with_temperature_equation)
                            for r in run_info_moments)
    else
        has_electrons = any(r.composition.electron_physics
                            ∈ (braginskii_fluid, kinetic_electrons,
                               kinetic_electrons_with_temperature_equation)
                            for r in run_info_dfns)
    end

    # Only plot neutral stuff if all runs have neutrals
    if any(ri !== nothing for ri ∈ run_info_moments)
        has_neutrals = all(r.n_neutral_species > 0 for r in run_info_moments)
    else
        has_neutrals = all(r.n_neutral_species > 0 for r in run_info_dfns)
    end

    is_1V = all(ri !== nothing && ri.vperp.n == 1 && ri.vzeta.n == 1 && ri.vr.n == 1
                for ri ∈ run_info_dfns)

    # Plots from moment variables
    #############################

    if any(ri !== nothing for ri ∈ run_info_moments)
        has_moments = true

        # Default to plotting moments from 'moments' files
        run_info = run_info_moments
    else
        has_moments = false
        # Fall back to trying to plot from 'dfns' files if those are all we have
        run_info = run_info_dfns
    end

    if any(ri !== nothing for ri ∈ run_info_dfns)
        has_dfns = true
    else
        has_dfns = false
    end

    if plot_prefix === nothing
        if length(run_info) == 1
            plot_prefix = run_info[1].run_prefix * "_"
        else
            comparison_plot_dir = "comparison_plots"
            mkpath(comparison_plot_dir)
            plot_prefix = joinpath(comparison_plot_dir, "compare_")
        end
    else
        if length(run_info) != 1
            comparison_plot_dir = "comparison_plots_$plot_prefix"
            mkpath(comparison_plot_dir)
            plot_prefix = joinpath(comparison_plot_dir, "compare_")
        end
    end

    timestep_diagnostics(run_info, run_info_dfns; plot_prefix=plot_prefix)
    if any((ri.composition.electron_physics ∈ (kinetic_electrons,
                                               kinetic_electrons_with_temperature_equation)
            && ri.t_input["kinetic_electron_solver"] != implicit_steady_state) for ri ∈ run_info)
        timestep_diagnostics(run_info, run_info_dfns; plot_prefix=plot_prefix, electron=true)
    end

    do_steady_state_residuals = any(input_dict[v]["steady_state_residual"]
                                    for v ∈ all_moment_variables)
    if do_steady_state_residuals
        textoutput_files = [ri.run_prefix * "_residuals.txt"
                            for ri in run_info if ri !== nothing]
        for (f, ri) in zip(textoutput_files, run_info)
            # Write the time into the output file. Also overwrites any existing file so we
            # can append for each variable below
            open(f, "w") do io
                # Use lpad to get fixed-width strings to print, so we get nice columns of
                # output. 24 characters should be enough to represent any float with at
                # least a couple of spaces in front to separate columns (e.g.  "
                # -3.141592653589793e100"
                line = string((lpad(string(x), 24) for x ∈ ri.time[2:end])...)
                line *= "  # time"
                println(io, line)
            end
        end
        steady_state_residual_fig_axes =
            _get_steady_state_residual_fig_axes(length(run_info))
    else
        steady_state_residual_fig_axes = nothing
    end

    for variable_name ∈ all_moment_variables
        plots_for_variable(run_info, variable_name; plot_prefix=plot_prefix,
                           has_rdim=has_rdim, has_zdim=has_zdim, is_1V=is_1V,
                           steady_state_residual_fig_axes=steady_state_residual_fig_axes)
    end

    if do_steady_state_residuals
        _save_residual_plots(steady_state_residual_fig_axes, plot_prefix)
    end

    # Plots from distribution function variables
    ############################################
    if any(ri !== nothing for ri in run_info_dfns)
        dfn_variable_list = ion_dfn_variables
        if has_electrons
            dfn_variable_list = tuple(dfn_variable_list..., electron_dfn_variables...)
        end
        if has_neutrals
            dfn_variable_list = tuple(dfn_variable_list..., neutral_dfn_variables...)
        end
        for variable_name ∈ dfn_variable_list
            plots_for_dfn_variable(run_info_dfns, variable_name; plot_prefix=plot_prefix,
                                   has_rdim=has_rdim, has_zdim=has_zdim, is_1V=is_1V)
        end
    end

    compare_runs(run_info, run_info_dfns; has_electrons, has_neutrals,
                 plot_prefix=plot_prefix, has_rdim=has_rdim, has_zdim=has_zdim,
                 is_1V=is_1V, has_dfns=has_dfns)

    plot_charged_pdf_2D_at_wall(run_info_dfns; plot_prefix=plot_prefix)
    if has_electrons
        plot_charged_pdf_2D_at_wall(run_info_dfns; plot_prefix=plot_prefix, electron=true)
    end
    if has_neutrals
        plot_neutral_pdf_2D_at_wall(run_info_dfns; plot_prefix=plot_prefix)
    end

    constraints_plots(run_info; plot_prefix=plot_prefix)

    if has_rdim
        # Plots for 2D instability do not make sense for 1D simulations
        instability_input = input_dict["instability2D"]
        if any((instability_input["plot_1d"], instability_input["plot_2d"],
                instability_input["animate_perturbations"]))
            # Get zind from the first variable in the loop (phi), and use the same one for
            # all subseqeunt variables.
            zind = Union{mk_int,Nothing}[nothing for _ ∈ run_info_moments]
            for variable_name ∈ ("phi", "density", "temperature")
                zind = instability2D_plots(run_info_moments, variable_name,
                                           plot_prefix=plot_prefix, zind=zind)
            end
        end
    end

    Chodura_condition_plots(run_info_dfns, plot_prefix=plot_prefix)

    sound_wave_plots(run_info; plot_prefix=plot_prefix)

    collisionality_plots(run_info, plot_prefix)

    #mk_1D1V_term_size_diagnostics(run_info, run_info_dfns, plot_prefix)

    manufactured_solutions_analysis(run_info; plot_prefix=plot_prefix)
    manufactured_solutions_analysis_dfns(run_info_dfns; plot_prefix=plot_prefix)

    timing_data(run_info; plot_prefix=plot_prefix, this_input_dict=input_dict)
    parallel_scaling(run_info; plot_prefix=plot_prefix, this_input_dict=input_dict)
    parallel_scaling(run_info; plot_prefix=plot_prefix, this_input_dict=input_dict, weak=true)

    for ri ∈ run_info
        close_run_info(ri)
    end
    for ri ∈ run_info_dfns
        close_run_info(ri)
    end

    return nothing
end

"""
    generate_example_input_file(filename::String=$default_input_file_name;
                                overwrite::Bool=false)

Create an example makie-post-processing input file.

Every option is commented out, but filled with the default value.

Pass `filename` to choose the name of the example file (defaults to the default input file
name used by `makie_post_process()`).

Pass `overwrite=true` to overwrite any existing file at `filename`.
"""
function generate_example_input_file(filename::String=default_input_file_name;
                                     overwrite::Bool=false)

    if ispath(filename) && !overwrite
        error("$filename already exists. If you want to overwrite it, pass "
              * "`overwrite=true` to `generate_example_input_file()`.")
    end

    # Get example input, then convert to a String formatted as the contents of a TOML
    # file
    input_dict = generate_example_input_Dict()
    buffer = IOBuffer()
    TOML.print(buffer, input_dict)
    file_contents = String(take!(buffer))

    # Separate file_contents into individual lines
    file_contents = split(file_contents, "\n")

    # Add comment character to all values (i.e. skipping section headings)
    for (i, line) ∈ enumerate(file_contents)
        if !startswith(line, "[") && !(line == "")
            # Not a section heading, so add comment character
            file_contents[i] = "#" * line
        end
    end

    # Join back into single string
    file_contents = join(file_contents, "\n")

    # Write to output file
    open(filename, write=true, truncate=overwrite, append=false) do io
        print(io, file_contents)
    end

    return nothing
end

"""
    generate_maximal_input_file(filename::String=default_input_file_name;
                                overwrite::Bool=false)

Generate an input file with all `Bool` options set to `true`, and no other options
present. Intended mostly for setting up tests to check that `makie_post_process` runs
without errors.

`handle_errors` is set to `false` (although it is a `Bool`) so that any errors are not
caught, and so will register as a test failure.
"""
function generate_maximal_input_file(filename::String=default_input_file_name;
                                     overwrite::Bool=false)

    if ispath(filename) && !overwrite
        error("$filename already exists. If you want to overwrite it, pass "
              * "`overwrite=true` to `generate_example_input_file()`.")
    end

    # Get example input, then convert to a String formatted as the contents of a TOML
    # file
    input_dict = generate_example_input_Dict()

    # Filter out any non-Bool options, set "handle_errors" to false, and all other Bool
    # options to true.
    function filter_input!(d)
        for (k,v) ∈ d
            if isa(v, AbstractDict)
                filter_input!(v)
            elseif k == "handle_errors"
                d[k] = false
            elseif isa(v, Bool)
                d[k] = true
            else
                pop!(d, k)
            end
        end
        return nothing
    end
    filter_input!(input_dict)

    # Write to output file
    open(filename, write=true, truncate=overwrite, append=false) do io
        TOML.print(io, input_dict)
    end

    return nothing
end

"""
    generate_example_input_Dict()

Create a Dict containing all the makie-post-processing options with default values
"""
function generate_example_input_Dict()
    original_input = deepcopy(input_dict)
    original_input_dfns = deepcopy(input_dict_dfns)

    # Set up input_dict and input_dict_dfns with all-default parameters
    setup_makie_post_processing_input!(OrderedDict{String,Any}())

    # Merge input_dict and input_dict_dfns, then convert to a String formatted as the
    # contents of a TOML file
    combined_input_dict = merge(input_dict_dfns, input_dict)

    # Restore original state of input_dict and input_dict_dfns
    clear_Dict!(input_dict)
    clear_Dict!(input_dict_dfns)
    merge!(input_dict, original_input)
    merge!(input_dict_dfns, original_input_dfns)

    return combined_input_dict
end

"""
    get_run_info(run_dir...; itime_min=1, itime_max=0,
                 itime_skip=1, dfns=false, initial_electron=false, electron_debug=false,
                 do_setup=true, setup_input_file=nothing)
    get_run_info((run_dir, restart_index)...; itime_min=1, itime_max=0,
                 itime_skip=1, dfns=false, initial_electron=false, electron_debug=false,
                 do_setup=true, setup_input_file=nothing)

Get file handles and other info for a single run

`run_dir` is either the directory to read output from (whose name should be the
`run_name`), or a moment_kinetics binary output file. If a file is passed, it is only used
to infer the directory and `run_name`, so it is possible for example to pass a
`.moments.h5` output file and also `dfns=true` and the `.dfns.h5` file will be the one
actually opened (as long as it exists).

`restart_index` can be given by passing a Tuple, e.g. `("runs/example", 42)` as the
positional argument. It specifies which restart to read if there are multiple restarts. If
no `restart_index` is given or if `nothing` is passed, read all restarts and concatenate
them. An integer value reads the restart with that index - `-1` indicates the latest
restart (which does not have an index).

Several runs can be loaded at the same time by passing multiple positional arguments. Each
argument can be a String `run_dir` giving a directory to read output from or a Tuple
`(run_dir, restart_index)` giving both a directory and a restart index (it is allowed to
mix Strings and Tuples in a call).

By default load data from moments files, pass `dfns=true` to load from distribution
functions files, or `initial_electron=true` and `dfns=true` to load from initial electron
state files, or `electron_debug=true` and `dfns=true` to load from electron debug files.

The `itime_min`, `itime_max` and `itime_skip` options can be used to select only a slice
of time points when loading data. In `makie_post_process` these options are read from the
input (if they are set) before `get_run_info()` is called, so that the `run_info` returned
can be passed to [`setup_makie_post_processing_input!`](@ref), to be used for defaults for
the remaining options. If either `itime_min` or `itime_max` are ≤0, their values are used
as offsets from the final time index of the run.

`setup_makie_post_processing_input!()` is called at the end of `get_run_info()`, for
convenience when working interactively. Use
[`moment_kinetics.load_data.get_run_info_no_setup`](@ref) if you do not want this. A
post-processing input file can be passed to `setup_input_file` that will be passed to
`setup_makie_post_processing_input!()` if you do not want to use the default input file.
"""
function get_run_info(args...; do_setup=true, setup_input_file=nothing, dfns=false,
                      kwargs...)

    run_info = get_run_info_no_setup(args...; dfns=dfns, kwargs...)

    if do_setup
        if dfns
            setup_makie_post_processing_input!(
                setup_input_file; run_info_dfns=run_info,
                allow_missing_input_file=(setup_input_file === nothing))
        else
            setup_makie_post_processing_input!(
                setup_input_file; run_info_moments=run_info,
                allow_missing_input_file=(setup_input_file === nothing))
        end
    end

    return run_info
end
