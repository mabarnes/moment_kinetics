"""
Post processing functions using Makie.jl

Options are read by default from a file `post_processing_input.toml`, if it exists.

The plots can be generated from the command line by running
```
julia --project run_makie_post_processing.jl dir1 [dir2 [dir3 ...]]
```
"""
module makie_post_processing

export makie_post_process, generate_example_input_file, get_variable,
       setup_makie_post_processing_input!, get_run_info, close_run_info
export animate_f_unnorm_vs_vpa, animate_f_unnorm_vs_vpa_z, get_1d_ax, get_2d_ax,
       irregular_heatmap, irregular_heatmap!, plot_f_unnorm_vs_vpa,
       plot_f_unnorm_vs_vpa_z, positive_or_nan, get_variable, positive_or_nan,
       put_legend_above, put_legend_below, put_legend_left, put_legend_right
export timing_data

include("shared_utils.jl")

# Need this import just to allow links in the docstrings to be understood by Documenter.jl
import moment_kinetics
using moment_kinetics: check_so_newer_than_code
using moment_kinetics.analysis: analyze_fields_data, check_Chodura_condition,
                                get_r_perturbation, get_Fourier_modes_2D,
                                get_Fourier_modes_1D, steady_state_residuals,
                                get_unnormalised_f_dzdt_1d, get_unnormalised_f_coords_2d,
                                get_unnormalised_f_1d, vpagrid_to_dzdt_2d,
                                get_unnormalised_f_2d
using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.input_structs
using moment_kinetics.looping: all_dimensions, ion_dimensions, neutral_dimensions
using moment_kinetics.manufactured_solns: manufactured_solutions,
                                          manufactured_electric_fields
using moment_kinetics.load_data: close_run_info, get_run_info_no_setup, get_variable,
                                 timestep_diagnostic_variables, em_variables,
                                 ion_moment_variables, electron_moment_variables,
                                 neutral_moment_variables, all_moment_variables,
                                 ion_dfn_variables, electron_dfn_variables,
                                 neutral_dfn_variables, all_dfn_variables, ion_variables,
                                 neutral_variables, all_variables, ion_source_variables,
                                 neutral_source_variables, electron_source_variables
using moment_kinetics.initial_conditions: vpagrid_to_dzdt
using .shared_utils: calculate_and_write_frequencies
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.velocity_moments: integrate_over_vspace,
                                        integrate_over_neutral_vspace

using Combinatorics
using LaTeXStrings
using LsqFit
using MPI
using NaNMath
using OrderedCollections
using StatsBase
using TOML

using CairoMakie
using Makie

const default_input_file_name = "post_processing_input.toml"

"""
Global dict containing settings for makie_post_processing. Can be re-loaded at any time
to change settings.

Is an OrderedDict so the order of sections is nicer if `input_dict` is written out as a
TOML file.
"""
const input_dict = OrderedDict{String,Any}()

"""
Global dict containing settings for makie_post_processing for files with distribution
function output. Can be re-loaded at any time to change settings.

Is an OrderedDict so the order of sections is nicer if `input_dict_dfns` is written out as
a TOML file.
"""
const input_dict_dfns = OrderedDict{String,Any}()

const one_dimension_combinations_no_t = setdiff(all_dimensions, (:s, :sn))
const one_dimension_combinations = (:t, one_dimension_combinations_no_t...)
const two_dimension_combinations_no_t = Tuple(
          Tuple(c) for c in unique((combinations(setdiff(ion_dimensions, (:s,)), 2)...,
                                    combinations(setdiff(neutral_dimensions, (:sn,)), 2)...)))
const two_dimension_combinations = Tuple(
         Tuple(c) for c in
         unique((combinations((:t, setdiff(ion_dimensions, (:s,))...), 2)...,
                 combinations((:t, setdiff(neutral_dimensions, (:sn,))...), 2)...)))

"""
    makie_post_process(run_dir...;
                       input_file::String=default_input_file_name,
                       restart_index::Union{Nothing,mk_int,Tuple}=nothing,
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
index). A tuple with the same length as `run_dir` can also be passed to give a different
`restart_index` for each run.

`plot_prefix` can be specified to give the prefix (directory and first part of file name)
to use when saving plots/animations. By default the run directory and run name are used if
there is only one run, and "comparison_plots/compare_" is used if there are multiple runs.

If `input_file` does not exist, prints warning and uses default options.
"""
function makie_post_process(run_dir...;
                            input_file::String=default_input_file_name,
                            restart_index::Union{Nothing,mk_int,Tuple}=nothing,
                            plot_prefix::Union{Nothing,AbstractString}=nothing)
    if isfile(input_file)
        new_input_dict = TOML.parsefile(input_file)
    else
        println("Warning: $input_file does not exist, using default post-processing "
                * "options")
        new_input_dict = OrderedDict{String,Any}()
    end

    return makie_post_process(run_dir, new_input_dict; restart_index=restart_index,
                              plot_prefix=plot_prefix)
end

"""
    makie_post_process(run_dir::Union{String,Tuple},
                       new_input_dict::Dict{String,Any};
                       restart_index::Union{Nothing,mk_int,Tuple}=nothing,
                       plot_prefix::Union{Nothing,AbstractString}=nothing)

Run post prossing, with (non-default) input given in a Dict

`run_dir...` is the path to the directory to plot from. If more than one `run_dir` is
given, plots comparing the runs in `run_dir...` are made.
A moment_kinetics binary output file can also be passed as `run_dir`, in which case the
filename is only used to infer the directory and `run_name`, so it is possible for example
to pass a `.moments.h5` output file and still make distribution function plots (as long as
the corresponding `.dfns.h5` file exists).

`input_dict` is a dictionary containing settings for the post-processing.

`restart_index` specifies which restart to read if there are multiple restarts. The
default (`nothing`) reads all restarts and concatenates them. An integer value reads the
restart with that index - `-1` indicates the latest restart (which does not have an
index). A tuple with the same length as `run_dir` can also be passed to give a different
`restart_index` for each run.

`plot_prefix` can be specified to give the prefix (directory and first part of file name)
to use when saving plots/animations. By default the run directory and run name are used if
there is only one run, and "comparison_plots/compare_" is used if there are multiple runs.
"""
function makie_post_process(run_dir::Union{String,Tuple},
                            new_input_dict::AbstractDict{String,Any};
                            restart_index::Union{Nothing,mk_int,Tuple}=nothing,
                            plot_prefix::Union{Nothing,AbstractString}=nothing)
    if isa(run_dir, String)
        # Make run_dir a one-element tuple if it is not a tuple
        run_dir = (run_dir,)
    end
    # Normalise by removing any trailing slashes - with a slash basename() would return an
    # empty string
    run_dir = Tuple(rstrip(ri, '/') for ri ∈ run_dir)

    new_input_dict = convert_to_OrderedDicts!(new_input_dict)

    if !isa(restart_index, Tuple)
        # Convert scalar restart_index to Tuple so we can treat everything the same below
        restart_index = Tuple(restart_index for _ ∈ run_dir)
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
    if !isa(run_info_moments, Tuple)
        run_info_moments = (run_info_moments,)
    end
    run_info_dfns = get_run_info(zip(run_dir, restart_index)..., itime_min=itime_min_dfns,
                                 itime_max=itime_max_dfns, itime_skip=itime_skip_dfns,
                                 dfns=true, do_setup=false)
    if !isa(run_info_dfns, Tuple)
        run_info_dfns = (run_info_dfns,)
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
    end

    timestep_diagnostics(run_info, run_info_dfns; plot_prefix=plot_prefix)
    if any((ri.composition.electron_physics ∈ (kinetic_electrons,
                                               kinetic_electrons_with_temperature_equation)
            && !ri.t_input["implicit_electron_advance"]) for ri ∈ run_info)
        timestep_diagnostics(run_info, run_info_dfns; plot_prefix=plot_prefix, electron=true)
    end

    do_steady_state_residuals = any(input_dict[v]["steady_state_residual"]
                                    for v ∈ all_moment_variables)
    if do_steady_state_residuals
        textoutput_files = Tuple(ri.run_prefix * "_residuals.txt"
                                 for ri in run_info if ri !== nothing)
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

    if all(ri === nothing for ri ∈ run_info_dfns)
        nvperp = nothing
    else
        nvperp = maximum(ri.vperp.n_global for ri ∈ run_info_dfns if ri !== nothing)
    end
    manufactured_solutions_analysis(run_info; plot_prefix=plot_prefix, nvperp=nvperp)
    manufactured_solutions_analysis_dfns(run_info_dfns; plot_prefix=plot_prefix)

    timing_data(run_info; plot_prefix=plot_prefix, this_input_dict=input_dict)

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
    setup_makie_post_processing_input!(input_file::Union{AbstractString,Nothing}=nothing;
                                       run_info_moments=nothing, run_info_dfns=nothing,
                                       allow_missing_input_file=false)
    setup_makie_post_processing_input!(new_input_dict::AbstractDict{String,Any};
                                       run_info_moments=nothing,
                                       run_info_dfns=nothing)

Pass `input_file` to read the input from an input file other than
`$default_input_file_name`. You can also pass a `Dict{String,Any}` of options.

Set up input, storing in the global [`input_dict`](@ref) and [`input_dict_dfns`](@ref) to
be used in the various plotting and analysis functions.

The `run_info` that you are using (as returned by
[`get_run_info`](@ref)) should be passed to `run_info_moments` (if it contains only the
moments), or `run_info_dfns` (if it also contains the distributions functions), or both
(if you have loaded both sets of output).  This allows default values to be set based on
the grid sizes and number of time points read from the output files. Note that
`setup_makie_post_processing_input!()` is called by default at the end of
`get_run_info()`, for conveinence in interactive use.

By default an error is raised if `input_file` does not exist. To continue anyway, using
default options, pass `allow_missing_input_file=true`.
"""
function setup_makie_post_processing_input! end

function setup_makie_post_processing_input!(
        input_file::Union{AbstractString,Nothing}=nothing; run_info_moments=nothing,
        run_info_dfns=nothing, allow_missing_input_file=false)

    if input_file === nothing
        input_file = default_input_file_name
    end

    if isfile(input_file)
        new_input_dict = TOML.parsefile(input_file)
    elseif allow_missing_input_file
        println("Warning: $input_file does not exist, using default post-processing "
                * "options")
        new_input_dict = OrderedDict{String,Any}()
    else
        error("$input_file does not exist")
    end
    setup_makie_post_processing_input!(new_input_dict, run_info_moments=run_info_moments,
                                       run_info_dfns=run_info_dfns)

    return nothing
end

function setup_makie_post_processing_input!(new_input_dict::AbstractDict{String,Any};
                                            run_info_moments=nothing,
                                            run_info_dfns=nothing)

    # Check that, if we are using a custom compiled system image that includes
    # moment_kinetics, the system image is newer than the source code files (if there are
    # changes made to the source code since the system image was compiled, they will not
    # affect the current run). Prints a warning if any code files are newer than the
    # system image.
    check_so_newer_than_code()

    convert_to_OrderedDicts!(new_input_dict)

    if isa(run_info_moments, Tuple)
        has_moments = any(ri !== nothing for ri ∈ run_info_moments)
    else
        has_moments = run_info_moments !== nothing
    end
    if isa(run_info_dfns, Tuple)
        has_dfns = any(ri !== nothing for ri ∈ run_info_dfns)
    else
        has_dfns = run_info_dfns !== nothing
    end

    if !has_moments && !has_dfns
        println("Neither `run_info_moments` nor `run_info_dfns` passed. Setting "
                * "defaults without using grid sizes")
    elseif !has_moments
        println("No run_info_moments, using run_info_dfns to set defaults")
        run_info_moments = run_info_dfns
        has_moments = true
    elseif !has_dfns
        println("No run_info_dfns, defaults for distribution function coordinate sizes "
                * "will be set to 1.")
    end

    _setup_single_input!(input_dict, new_input_dict, run_info_moments, false)
    _setup_single_input!(input_dict_dfns, new_input_dict, run_info_dfns, true)

    return nothing
end

# Utility function to reduce code duplication in setup_makie_post_processing_input!()
function _setup_single_input!(this_input_dict::OrderedDict{String,Any},
                              new_input_dict::AbstractDict{String,Any}, run_info,
                              dfns::Bool)
    # Remove all existing entries from this_input_dict
    clear_Dict!(this_input_dict)

    # Put entries from new_input_dict into this_input_dict
    merge!(this_input_dict, deepcopy(new_input_dict))

    if !isa(run_info, Tuple)
        # Make sure run_info is a Tuple
        run_info= (run_info,)
    end
    has_run_info = any(ri !== nothing for ri ∈ run_info)

    if has_run_info
        nt_unskipped_min = minimum(ri.nt_unskipped for ri in run_info
                                                   if ri !== nothing)
        nt_min = minimum(ri.nt for ri in run_info if ri !== nothing)
        nr_min = minimum(ri.r.n for ri in run_info if ri !== nothing)
        nz_min = minimum(ri.z.n for ri in run_info if ri !== nothing)
    else
        nt_unskipped_min = 1
        nt_min = 1
        nr_min = 1
        nz_min = 1
    end
    if dfns && has_run_info
        if any(ri.vperp !== nothing for ri ∈ run_info)
            nvperp_min = minimum(ri.vperp.n for ri in run_info
                                 if ri !== nothing && ri.vperp !== nothing)
        else
            nvperp_min = 1
        end
        if any(ri.vpa !== nothing for ri ∈ run_info)
            nvpa_min = minimum(ri.vpa.n for ri in run_info
                               if ri !== nothing && ri.vpa !== nothing)
        else
            nvpa_min = 1
        end
        if any(ri.vzeta !== nothing for ri ∈ run_info)
            nvzeta_min = minimum(ri.vzeta.n for ri in run_info
                                 if ri !== nothing && ri.vzeta !== nothing)
        else
            nvzeta_min = 1
        end
        if any(ri.vr !== nothing for ri ∈ run_info)
            nvr_min = minimum(ri.vr.n for ri in run_info
                              if ri !== nothing && ri.vr !== nothing)
        else
            nvr_min = 1
        end
        if any(ri.vz !== nothing for ri ∈ run_info)
            nvz_min = minimum(ri.vz.n for ri in run_info
                              if ri !== nothing && ri.vz !== nothing)
        else
            nvz_min = 1
        end
    else
        nvperp_min = 1
        nvpa_min = 1
        nvzeta_min = 1
        nvr_min = 1
        nvz_min = 1
    end

    # Whitelist of options that only apply at the global level, and should not be used
    # as defaults for per-variable options.
    # Notes:
    # - Don't allow setting "itime_*" and "itime_*_dfns" per-variable because we
    #   load time and time_dfns in run_info and these must use the same
    #   "itime_*"/"itime_*_dfns" setting as each variable.
    only_global_options = ("itime_min", "itime_max", "itime_skip", "itime_min_dfns",
                           "itime_max_dfns", "itime_skip_dfns", "handle_errors")

    set_defaults_and_check_top_level!(this_input_dict;
       # Options that only apply at the global level (not per-variable)
       ################################################################
       # Options that provide the defaults for per-variable settings
       #############################################################
       colormap="reverse_deep",
       animation_ext="gif",
       # Slice t to this value when making time-independent plots
       it0=nt_min,
       it0_dfns=nt_min,
       # Choose this species index when not otherwise specified
       is0=1,
       # Slice r to this value when making reduced dimensionality plots
       ir0=max(cld(nr_min, 3), 1),
       # Slice z to this value when making reduced dimensionality plots
       iz0=max(cld(nz_min, 3), 1),
       # Slice vperp to this value when making reduced dimensionality plots
       ivperp0=max(cld(nvperp_min, 3), 1),
       # Slice vpa to this value when making reduced dimensionality plots
       ivpa0=max(cld(nvpa_min, 3), 1),
       # Slice vzeta to this value when making reduced dimensionality plots
       ivzeta0=max(cld(nvzeta_min, 3), 1),
       # Slice vr to this value when making reduced dimensionality plots
       ivr0=max(cld(nvr_min, 3), 1),
       # Slice vz to this value when making reduced dimensionality plots
       ivz0=max(cld(nvz_min, 3), 1),
       # Time index to start from
       itime_min=1,
       # Time index to end at
       itime_max=nt_unskipped_min,
       # Load every `time_skip` time points for EM and moment variables, to save memory
       itime_skip=1,
       # Time index to start from for distribution functions
       itime_min_dfns=1,
       # Time index to end at for distribution functions
       itime_max_dfns=nt_unskipped_min,
       # Load every `time_skip` time points for distribution function variables, to save
       # memory
       itime_skip_dfns=1,
       plot_vs_r=true,
       plot_vs_z=true,
       plot_vs_r_t=true,
       plot_vs_z_t=true,
       plot_vs_z_r=true,
       animate_vs_z=false,
       animate_vs_r=false,
       animate_vs_z_r=false,
       show_element_boundaries=false,
       steady_state_residual=false,
       # By default, errors are caught so that later plots can still be made. For
       # debugging it can be useful to turn this off.
       handle_errors=true,
      )

    section_defaults = OrderedDict(k=>v for (k,v) ∈ this_input_dict
                                   if !isa(v, AbstractDict) &&
                                      !(k ∈ only_global_options))
    for variable_name ∈ tuple(all_moment_variables..., timestep_diagnostic_variables...)
        set_defaults_and_check_section!(
            this_input_dict, variable_name;
            OrderedDict(Symbol(k)=>v for (k,v) ∈ section_defaults)...)
    end

    plot_options_1d = Tuple(Symbol(:plot_vs_, d) for d ∈ one_dimension_combinations)
    plot_log_options_1d = Tuple(Symbol(:plot_log_vs_, d) for d ∈ one_dimension_combinations)
    plot_options_2d = Tuple(Symbol(:plot_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations)
    plot_log_options_2d = Tuple(Symbol(:plot_log_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations)
    animate_options_1d = Tuple(Symbol(:animate_vs_, d) for d ∈ one_dimension_combinations_no_t)
    animate_log_options_1d = Tuple(Symbol(:animate_log_vs_, d) for d ∈ one_dimension_combinations_no_t)
    animate_options_2d = Tuple(Symbol(:animate_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations_no_t)
    animate_log_options_2d = Tuple(Symbol(:animate_log_vs_, d2, :_, d1) for (d1, d2) ∈ two_dimension_combinations_no_t)
    for variable_name ∈ all_dfn_variables
        set_defaults_and_check_section!(
            this_input_dict, variable_name;
            check_moments=false,
            (o=>false for o ∈ plot_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ plot_log_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ plot_options_2d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ plot_log_options_2d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_log_options_1d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_options_2d if String(o) ∉ keys(section_defaults))...,
            (o=>false for o ∈ animate_log_options_2d if String(o) ∉ keys(section_defaults))...,
            plot_unnorm_vs_vpa=false,
            plot_unnorm_vs_vz=false,
            plot_unnorm_vs_vpa_z=false,
            plot_unnorm_vs_vz_z=false,
            plot_log_unnorm_vs_vpa=false,
            plot_log_unnorm_vs_vz=false,
            plot_log_unnorm_vs_vpa_z=false,
            plot_log_unnorm_vs_vz_z=false,
            animate_unnorm_vs_vpa=false,
            animate_unnorm_vs_vz=false,
            animate_unnorm_vs_vpa_z=false,
            animate_unnorm_vs_vz_z=false,
            animate_log_unnorm_vs_vpa=false,
            animate_log_unnorm_vs_vz=false,
            animate_log_unnorm_vs_vpa_z=false,
            animate_log_unnorm_vs_vz_z=false,
            OrderedDict(Symbol(k)=>v for (k,v) ∈ section_defaults)...)
        # Sort keys to make dict easier to read
        sort!(this_input_dict[variable_name])
    end

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf";
        plot=false,
        animate=false,
        advection_velocity=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf_electron";
        plot=false,
        animate=false,
        advection_velocity=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf_neutral";
        plot=false,
        animate=false,
        advection_velocity=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "constraints";
        plot=false,
        animate=false,
        it0=this_input_dict["it0"],
        ir0=this_input_dict["ir0"],
        iz0=this_input_dict["iz0"],
        ivperp0=this_input_dict["ivperp0"],
        ivpa0=this_input_dict["ivpa0"],
        ivzeta0=this_input_dict["ivzeta0"],
        ivr0=this_input_dict["ivr0"],
        ivz0=this_input_dict["ivz0"],
        animation_ext=this_input_dict["animation_ext"],
        show_element_boundaries=this_input_dict["show_element_boundaries"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "Chodura_condition";
        plot_vs_t=false,
        plot_vs_r=false,
        plot_vs_r_t=false,
        plot_f_over_vpa2=false,
        animate_f_over_vpa2=false,
        it0=this_input_dict["it0"],
        ir0=this_input_dict["ir0"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "instability2D";
        plot_1d=false,
        plot_2d=false,
        animate_perturbations=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "sound_wave_fit";
        calculate_frequency=false,
        plot=false,
        ir0=this_input_dict["ir0"],
        iz0=this_input_dict["iz0"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "manufactured_solns";
        calculate_error_norms=true,
        wall_plots=false,
        (o=>false for o ∈ plot_options_1d)...,
        (o=>false for o ∈ plot_log_options_1d)...,
        (o=>false for o ∈ plot_options_2d)...,
        (o=>false for o ∈ plot_log_options_2d)...,
        (o=>false for o ∈ animate_options_1d)...,
        (o=>false for o ∈ animate_log_options_1d)...,
        (o=>false for o ∈ animate_options_2d)...,
        (o=>false for o ∈ animate_log_options_2d if String(o) ∉ keys(section_defaults))...,
        (o=>section_defaults[String(o)] for o ∈ (:it0, :ir0, :iz0, :ivperp0, :ivpa0, :ivzeta0, :ivr0, :ivz0))...,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
        show_element_boundaries=this_input_dict["show_element_boundaries"],
       )
    sort!(this_input_dict["manufactured_solns"])

    set_defaults_and_check_section!(
        this_input_dict, "timestep_diagnostics";
        plot=true,
        animate_CFL=false,
        plot_timestep_residual=false,
        animate_timestep_residual=false,
        plot_timestep_error=false,
        animate_timestep_error=false,
        plot_steady_state_residual=false,
        animate_steady_state_residual=false,
       )

    set_defaults_and_check_section!(
        this_input_dict, "timing_data";
        plot=false,
        threshold=1.0e-2,
        include_patterns=String[],
        exclude_patterns=String[],
        ranks=mk_int[],
        figsize=[600,800]
       )

    # We allow top-level options in the post-processing input file
    check_sections!(this_input_dict; check_no_top_level_options=false)

    return nothing
end

function makie_post_processing_error_handler(e::Exception, message::String)
    handle_errors = get(input_dict, "handle_errors", true)
    if isa(e, InterruptException) || !handle_errors
        rethrow(e)
    else
        println(message * "\nError was $e.")
        return nothing
    end
end

"""
    get_run_info(run_dir...; itime_min=1, itime_max=0,
                 itime_skip=1, dfns=false, initial_electron=false, do_setup=true,
                 setup_input_file=nothing)
    get_run_info((run_dir, restart_index)...; itime_min=1, itime_max=0,
                 itime_skip=1, dfns=false, initial_electron=false, do_setup=true,
                 setup_input_file=nothing)

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
state files.

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

const chunk_size_1d = 10000
const chunk_size_2d = 100
struct VariableCache{T1,T2,T3}
    run_info::T1
    variable_name::String
    t_chunk_size::mk_int
    n_tinds::mk_int
    tinds_range_global::Union{UnitRange{mk_int},StepRange{mk_int}}
    tinds_chunk::Union{Base.RefValue{UnitRange{mk_int}},Base.RefValue{StepRange{mk_int}}}
    data_chunk::T2
    dim_slices::T3
end

function VariableCache(run_info, variable_name::String, t_chunk_size::mk_int;
                       it::Union{Nothing,AbstractRange}, is, iz, ir, ivperp, ivpa, ivzeta,
                       ivr, ivz)
    if it === nothing
        tinds_range_global = run_info.itime_min:run_info.itime_skip:run_info.itime_max
    else
        tinds_range_global = it
    end
    n_tinds = length(tinds_range_global)

    t_chunk_size = min(t_chunk_size, n_tinds)
    tinds_chunk = 1:t_chunk_size
    dim_slices = (is=is, iz=iz, ir=ir, ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                  ivz=ivz)
    data_chunk = get_variable(run_info, variable_name; it=tinds_range_global[tinds_chunk],
                              dim_slices...)

    return VariableCache(run_info, variable_name, t_chunk_size,
                         n_tinds, tinds_range_global, Ref(tinds_chunk),
                         data_chunk, dim_slices)
end

function get_cache_slice(variable_cache::VariableCache, tind)
    tinds_chunk = variable_cache.tinds_chunk[]
    local_tind = findfirst(i->i==tind, tinds_chunk)

    if local_tind === nothing
        if tind > variable_cache.n_tinds
            error("tind=$tind is bigger than the number of time indices "
                  * "($(variable_cache.n_tinds))")
        end
        # tind is not in the cache, so get a new chunk
        chunk_size = variable_cache.t_chunk_size
        new_chunk_start = ((tind-1) ÷ chunk_size) * chunk_size + 1
        new_chunk = new_chunk_start:min(new_chunk_start + chunk_size - 1, variable_cache.n_tinds)
        variable_cache.tinds_chunk[] = new_chunk
        selectdim(variable_cache.data_chunk,
                  ndims(variable_cache.data_chunk), 1:length(new_chunk)) .=
            get_variable(variable_cache.run_info, variable_cache.variable_name;
                         it=variable_cache.tinds_range_global[new_chunk],
                         variable_cache.dim_slices...)
        local_tind = findfirst(i->i==tind, new_chunk)
    end

    return selectdim(variable_cache.data_chunk, ndims(variable_cache.data_chunk),
                     local_tind)
end

function variable_cache_extrema(variable_cache::VariableCache; transform=identity)
    # Bit of a hack to iterate through all chunks that can be in the cache
    chunk_size = variable_cache.t_chunk_size
    data_min = data_max = NaN
    for it ∈ ((i - 1) * chunk_size + 1 for i ∈ 1:(variable_cache.n_tinds ÷ chunk_size))
        get_cache_slice(variable_cache, it)
        this_min, this_max = NaNMath.extrema(transform.(variable_cache.data_chunk))
        data_min = NaNMath.min(data_min, this_min)
        data_max = NaNMath.max(data_max, this_max)
    end

    return data_min, data_max
end

"""
    plots_for_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                       has_zdim=true, is_1V=false,
                       steady_state_residual_fig_axes=nothing)

Make plots for the EM field or moment variable `variable_name`.

Which plots to make are determined by the settings in the section of the input whose
heading is the variable name.

`run_info` is the information returned by [`get_run_info`](@ref).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`has_rdim`, `has_zdim` and/or `is_1V` can be passed to allow the function to skip some
plots that do not make sense for 0D/1D or 1V simulations (regardless of the settings).

`steady_state_residual_fig_axes` contains the figure, axes and legend places for steady
state residual plots.
"""
function plots_for_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                            has_zdim=true, is_1V=false,
                            steady_state_residual_fig_axes=nothing)
    input = Dict_to_NamedTuple(input_dict[variable_name])

    # test if any plot is needed
    if !(any(v for (k,v) in pairs(input) if
           startswith(String(k), "plot") || startswith(String(k), "animate") ||
           k == :steady_state_residual))
        return nothing
    end

    if !has_rdim && variable_name == "Er"
        return nothing
    elseif !has_zdim && variable_name == "Ez"
        return nothing
    elseif variable_name == "collision_frequency" &&
            all(ri.collisions.krook_collisions_option == "none" for ri ∈ run_info)
        # No Krook collisions active, so do not make plots.
        return nothing
    elseif variable_name ∈ union(electron_moment_variables, electron_source_variables, electron_dfn_variables) &&
            all(ri.composition.electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
                for ri ∈ run_info)
        return nothing
    end

    println("Making plots for $variable_name")
    flush(stdout)

    variable = nothing
    try
        variable = get_variable(run_info, variable_name)
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "plots_for_variable() failed for $variable_name - could not load data.")
    end

    if variable_name ∈ em_variables
        species_indices = (nothing,)
    elseif variable_name ∈ neutral_moment_variables ||
           variable_name ∈ neutral_dfn_variables
        species_indices = 1:maximum(ri.n_neutral_species for ri ∈ run_info)
    elseif variable_name ∈ ion_moment_variables ||
           variable_name ∈ ion_dfn_variables
        species_indices = 1:maximum(ri.n_ion_species for ri ∈ run_info)
    elseif variable_name in ion_source_variables
        species_indices = 1:maximum(length(ri.external_source_settings.ion) for ri ∈ run_info)
    elseif variable_name in electron_source_variables
        species_indices = 1:maximum(length(ri.external_source_settings.electron) for ri ∈ run_info)
    elseif variable_name in neutral_source_variables
        species_indices = 1:maximum(length(ri.external_source_settings.neutral) for ri ∈ run_info)
    else
        species_indices = 1:1
        #error("variable_name=$variable_name not found in any defined group")
    end
    for is ∈ species_indices
        if is !== nothing
            variable_prefix = plot_prefix * variable_name * "_spec$(is)_"
            log_variable_prefix = plot_prefix * "log" * variable_name * "_spec$(is)_"
        else
            variable_prefix = plot_prefix * variable_name * "_"
            log_variable_prefix = plot_prefix * "log" * variable_name * "_"
        end
        if variable_name == "Er" && !has_rdim
            # Skip if there is no r-dimension
            continue
        end
        if variable_name == "Ez" && !has_zdim
            # Skip if there is no r-dimension
            continue
        end
        if has_rdim && input.plot_vs_r_t
            plot_vs_r_t(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_r_t.pdf")
        end
        if has_zdim && input.plot_vs_z_t
            plot_vs_z_t(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_t.pdf")
        end
        if has_rdim && input.plot_vs_r
            plot_vs_r(run_info, variable_name, is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_r.pdf")
        end
        if has_zdim && input.plot_vs_z
            plot_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_z.pdf")
        end
        if has_rdim && has_zdim && input.plot_vs_z_r
            plot_vs_z_r(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_r.pdf")
        end
        if has_zdim && input.animate_vs_z
            animate_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_z." * input.animation_ext)
        end
        if has_rdim && input.animate_vs_r
            animate_vs_r(run_info, variable_name, is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_r." * input.animation_ext)
        end
        if has_rdim && has_zdim && input.animate_vs_z_r
            animate_vs_z_r(run_info, variable_name, is=is, data=variable, input=input,
                           outfile=variable_prefix * "vs_r." * input.animation_ext)
        end
        if input.steady_state_residual
            calculate_steady_state_residual(run_info, variable_name; is=is, data=variable,
                                            fig_axes=steady_state_residual_fig_axes)
        end
    end

    return nothing
end

"""
    plots_for_dfn_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                           has_zdim=true, is_1V=false)

Make plots for the distribution function variable `variable_name`.

Which plots to make are determined by the settings in the section of the input whose
heading is the variable name.

`run_info` is the information returned by [`get_run_info()`](@ref). The `dfns=true` keyword
argument must have been passed to [`get_run_info()`](@ref) so that output files containing
the distribution functions are being read.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`has_rdim`, `has_zdim` and/or `is_1V` can be passed to allow the function to skip some
plots that do not make sense for 0D/1D or 1V simulations (regardless of the settings).
"""
function plots_for_dfn_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                                has_zdim=true, is_1V=false)
    input = Dict_to_NamedTuple(input_dict_dfns[variable_name])

    is_neutral = variable_name ∈ neutral_dfn_variables
    is_electron = variable_name ∈ electron_dfn_variables

    if is_neutral
        animate_dims = setdiff(neutral_dimensions, (:sn,))
        if is_1V
            animate_dims = setdiff(animate_dims, (:vzeta, :vr))
        end
    else
        animate_dims = setdiff(ion_dimensions, (:s,))
        if is_1V
            animate_dims = setdiff(animate_dims, (:vperp,))
        end
    end
    if !has_rdim
        animate_dims = setdiff(animate_dims, (:r,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    plot_dims = tuple(:t, animate_dims...)

    moment_kinetic = any(ri !== nothing
                         && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                         for ri ∈ run_info)

    # test if any plot is needed
    if !any(v for (k,v) in pairs(input) if
            startswith(String(k), "plot") || startswith(String(k), "animate"))
        return nothing
    end

    println("Making plots for $variable_name")
    flush(stdout)

    if is_neutral
        species_indices = 1:maximum(ri.n_neutral_species for ri ∈ run_info)
    else
        species_indices = 1:maximum(ri.n_ion_species for ri ∈ run_info)
    end
    for is ∈ species_indices
        variable_prefix = plot_prefix * variable_name * "_"
        log_variable_prefix = plot_prefix * "log" * variable_name * "_"

        # Note that we use `yscale=log10` and `transform=positive_or_nan` rather than
        # defining a custom scaling function (which would return NaN for negative
        # values) because it messes up the automatic minimum value for the colorscale:
        # The transform removes any zero or negative values from the data, so the
        # minimum value for the colorscale is set by the smallest positive value; with
        # only the custom colorscale, the minimum would be negative and the
        # corresponding color would be the color for NaN, which does not go on the
        # Colorbar and so causes an error.
        for (log, yscale, transform, var_prefix) ∈
                ((:"", nothing, identity, variable_prefix),
                 (:_log, log10, x->positive_or_nan(x; epsilon=1.e-20), log_variable_prefix))
            for dim ∈ plot_dims
                if input[Symbol(:plot, log, :_vs_, dim)]
                    func = getfield(makie_post_processing, Symbol(:plot_vs_, dim))
                    outfile = var_prefix * "vs_$dim.pdf"
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         yscale=yscale, transform=transform)
                end
            end
            for (dim1, dim2) ∈ combinations(plot_dims, 2)
                if input[Symbol(:plot, log, :_vs_, dim2, :_, dim1)]
                    func = getfield(makie_post_processing,
                                    Symbol(:plot_vs_, dim2, :_, dim1))
                    outfile = var_prefix * "vs_$(dim2)_$(dim1).pdf"
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         colorscale=yscale, transform=transform)
                end
            end
            for dim ∈ animate_dims
                if input[Symbol(:animate, log, :_vs_, dim)]
                    func = getfield(makie_post_processing, Symbol(:animate_vs_, dim))
                    outfile = var_prefix * "vs_$dim." * input.animation_ext
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         yscale=yscale, transform=transform)
                end
            end
            for (dim1, dim2) ∈ combinations(animate_dims, 2)
                if input[Symbol(:animate, log, :_vs_, dim2, :_, dim1)]
                    func = getfield(makie_post_processing,
                                    Symbol(:animate_vs_, dim2, :_, dim1))
                    outfile = var_prefix * "vs_$(dim2)_$(dim1)." * input.animation_ext
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         colorscale=yscale, transform=transform)
                end
            end

            if moment_kinetic
                if is_neutral
                    if input[Symbol(:plot, log, :_unnorm_vs_vz)]
                        outfile = var_prefix * "unnorm_vs_vz.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, neutral=true, is=is,
                                             outfile=outfile, yscale=yscale, transform=transform)
                    end
                    if has_zdim && input[Symbol(:plot, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true, is=is,
                                               outfile=outfile, colorscale=yscale,
                                               transform=transform)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vz)]
                        outfile = var_prefix * "unnorm_vs_vz." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input, neutral=true, is=is,
                                                outfile=outfile, yscale=yscale,
                                                transform=transform)
                    end
                    if has_zdim && input[Symbol(:animate, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true, is=is,
                                                  outfile=outfile, colorscale=yscale,
                                                  transform=transform)
                    end
                else
                    if input[Symbol(:plot, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, electron=is_electron,
                                             is=is, outfile=outfile, yscale=yscale,
                                             transform=transform)
                    end
                    if has_zdim && input[Symbol(:plot, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input,
                                               electron=is_electron, is=is,
                                               outfile=outfile, colorscale=yscale,
                                               transform=transform)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input,
                                                electron=is_electron, is=is,
                                                outfile=outfile, yscale=yscale,
                                                transform=transform)
                    end
                    if has_zdim && input[Symbol(:animate, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input,
                                                  electron=is_electron, is=is,
                                                  outfile=outfile, colorscale=yscale,
                                                  transform=transform)
                    end
                end
                check_moment_constraints(run_info, is_neutral; input=input, plot_prefix)
            end
        end
    end

    return nothing
end

function check_moment_constraints(run_info::Tuple, is_neutral; input, plot_prefix)
    if !input.check_moments
        return nothing
    end

    # For now, don't support comparison plots
    if length(run_info) > 1
        error("Comparison plots not supported by check_moment_constraints()")
    end
    return check_moment_constraints(run_info[1], is_neutral; input=input,
                                    plot_prefix=plot_prefix)
end

function check_moment_constraints(run_info, is_neutral; input, plot_prefix)
    if !input.check_moments
        return nothing
    end

    # For now assume there is only one ion or neutral species
    is = 1

    if is_neutral
        fn = get_variable(run_info, "f_neutral")
        if run_info.evolve_density
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integrate_over_neutral_vspace(
                    @view(fn[:,:,:,iz,ir,is,it]), run_info.vz.grid, 0, run_info.vz.wgts,
                    run_info.vr.grid, 0, run_info.vr.wgts, run_info.vzeta.grid, 0,
                    run_info.vzeta.wgts)
            end
            error = moment .- 1.0
            animate_vs_z(run_info, "density moment neutral"; data=error, input=input,
                         outfile=plot_prefix * "density_moment_neutral_check.gif")
        end

        if run_info.evolve_upar
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integrate_over_neutral_vspace(
                    @view(fn[:,:,:,iz,ir,is,it]), run_info.vz.grid, 1, run_info.vz.wgts,
                    run_info.vr.grid, 0, run_info.vr.wgts, run_info.vzeta.grid, 0,
                    run_info.vzeta.wgts)
            end
            error = moment
            animate_vs_z(run_info, "parallel flow neutral"; data=error, input=input,
                         outfile=plot_prefix * "parallel_flow_moment_neutral_check.gif")
        end

        if run_info.evolve_ppar
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integrate_over_neutral_vspace(
                    @view(fn[:,:,:,iz,ir,is,it]), run_info.vz.grid, 2, run_info.vz.wgts,
                    run_info.vr.grid, 0, run_info.vr.wgts, run_info.vzeta.grid, 0,
                    run_info.vzeta.wgts)
            end
            error = moment .- 0.5
            animate_vs_z(run_info, "parallel pressure neutral"; data=error, input=input,
                         outfile=plot_prefix * "parallel_pressure_moment_neutral_check.gif")
        end
    else
        f = get_variable(run_info, "f")
        if run_info.evolve_density
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integrate_over_vspace(
                    @view(f[:,:,iz,ir,is,it]), run_info.vpa.grid, 0, run_info.vpa.wgts,
                    run_info.vperp.grid, 0, run_info.vperp.wgts)
            end
            error = moment .- 1.0
            animate_vs_z(run_info, "density moment"; data=error, input=input,
                         outfile=plot_prefix * "density_moment_check.gif")
        end

        if run_info.evolve_upar
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integrate_over_vspace(
                    @view(f[:,:,iz,ir,is,it]), run_info.vpa.grid, 1, run_info.vpa.wgts,
                    run_info.vperp.grid, 0, run_info.vperp.wgts)
            end
            error = moment
            animate_vs_z(run_info, "parallel flow moment"; data=error, input=input,
                         outfile=plot_prefix * "parallel_flow_moment_check.gif")
        end

        if run_info.evolve_ppar
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integrate_over_vspace(
                    @view(f[:,:,iz,ir,is,it]), run_info.vpa.grid, 2, run_info.vpa.wgts,
                    run_info.vperp.grid, 0, run_info.vperp.wgts)
            end
            error = moment .- 0.5
            animate_vs_z(run_info, "parallel pressure moment"; data=error, input=input,
                         outfile=plot_prefix * "parallel_pressure_moment_check.gif")
        end
    end

    return nothing
end

# Generate 1d plot functions for each dimension
for dim ∈ one_dimension_combinations
    function_name_str = "plot_vs_$dim"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim_str = String(dim)
    if dim == :t
        dim_grid = :( run_info.time )
    else
        dim_grid = :( run_info.$dim.grid )
    end
    idim = Symbol(:i, dim)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, yscale=nothing,
                 transform=identity, axis_args=Dict{Symbol,Any}(), it=nothing,
                 $($spaces)ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                 $($spaces)ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, ax=nothing, label=nothing,
                 $($spaces)outfile=nothing, yscale=nothing, transform=identity,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                 $($spaces)iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
                 $($spaces)ivr=nothing, ivz=nothing, kwargs...)

             Plot `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref)) vs $($dim_str).

             If a Tuple of `run_info` is passed, the plots from each run are overlayed on
             the same axis, and a legend is added.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             If `outfile` is given, the plot will be saved to a file with that name. The
             suffix determines the file type.

             `yscale` can be used to set the scaling function for the y-axis. Options are
             `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `lines!() function`.

             When a single `run_info` is passed, `label` can be used to set the label for
             the line created by this plot, which would be used if it is added to a
             `Legend`.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be added to `ax`.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case the object
             returned by Makie's `lines!()` function is returned.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, yscale=nothing,
                                     transform=identity, axis_args=Dict{Symbol,Any}(),
                                     $idim=nothing, kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end

                     if input === nothing
                         if run_info[1].dfns
                             if var_name ∈ keys(input_dict_dfns)
                                 input = input_dict_dfns[var_name]
                             else
                                 input = input_dict_dfns
                             end
                         else
                             if var_name ∈ keys(input_dict)
                                 input = input_dict[var_name]
                             else
                                 input = input_dict
                             end
                         end
                     end
                     if input isa AbstractDict
                         input = Dict_to_NamedTuple(input)
                     end

                     n_runs = length(run_info)

                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale, axis_args...)
                     for (d, ri) ∈ zip(data, run_info)
                         $function_name(ri, var_name, is=is, data=d, input=input, ax=ax,
                                        transform=transform, label=ri.run_name,
                                        $idim=$idim, kwargs...)
                     end

                     if input.show_element_boundaries && Symbol($dim_str) != :t
                         # Just plot element boundaries from first run, assuming that all
                         # runs being compared use the same grid.
                         ri = run_info[1]
                         element_boundary_inds =
                             [i for i ∈ 1:ri.$dim.ngrid-1:ri.$dim.n_global
                                if $idim === nothing || i ∈ $idim]
                         element_boundary_positions = ri.$dim.grid[element_boundary_inds]
                         vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                     end

                     if n_runs > 1
                         put_legend_above(fig, ax)
                     end

                     if outfile !== nothing
                         save(outfile, fig)
                     end
                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str) failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, fig=nothing, ax=nothing,
                                     label=nothing, outfile=nothing,
                                     axis_args=Dict{Symbol,Any}(), it=nothing,
                                     ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices($(QuoteNode(dim));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = get_variable(run_info, var_name; dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim)); input=input, it=it,
                                         is=is, ir=ir, iz=iz, ivperp=ivperp, ivpa=ivpa,
                                         ivzeta=ivzeta, ivr=ivr, ivz=ivz)
                 end

                 if ax === nothing
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         axis_args...)
                     ax_was_nothing = true
                 else
                     ax_was_nothing = false
                 end

                 x = $dim_grid
                 if $idim !== nothing
                     x = x[$idim]
                 end
                 plot_1d(x, data; label=label, ax=ax, kwargs...)

                 if input.show_element_boundaries && Symbol($dim_str) != :t && ax_was_nothing
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim.ngrid-1:run_info.$dim.n_global
                            if $idim === nothing || i ∈ $idim]
                     element_boundary_positions = run_info.$dim.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                 end

                 if outfile !== nothing
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save(outfile, fig)
                 end

                 return fig
             end
         end)
end

# Generate 2d plot functions for all combinations of dimensions
for (dim1, dim2) ∈ two_dimension_combinations
    function_name_str = "plot_vs_$(dim2)_$(dim1)"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim1_str = String(dim1)
    dim2_str = String(dim2)
    if dim1 == :t
        dim1_grid = :( run_info.time )
    else
        dim1_grid = :( run_info.$dim1.grid )
    end
    dim2_grid = :( run_info.$dim2.grid )
    idim1 = Symbol(:i, dim1)
    idim2 = Symbol(:i, dim2)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, colorscale=identity,
                 $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                 $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                 $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                 $($spaces)kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, ax=nothing,
                 $($spaces)colorbar_place=nothing, title=nothing,
                 $($spaces)outfile=nothing, colorscale=identity, transform=identity,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                 $($spaces)iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
                 $($spaces)ivr=nothing, ivz=nothing, kwargs...)

             Plot `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref))vs $($dim1_str) and $($dim2_str).

             If a Tuple of `run_info` is passed, the plots from each run are displayed in
             a horizontal row, and the subtitle for each subplot is the 'run name'.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             If `outfile` is given, the plot will be saved to a file with that name. The
             suffix determines the file type.

             `colorscale` can be used to set the scaling function for the colors. Options
             are `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `heatmap!() function`.

             When a single `run_info` is passed, `title` can be used to set the title for
             the (sub-)plot.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be added to `ax`. A colorbar will be created in
             `colorbar_place` if it is given a `GridPosition`.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case the object
             returned by Makie's `heatmap!()` function is returned.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, transform=identity,
                                     axis_args=Dict{Symbol,Any}(), kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end
                     fig, ax, colorbar_places = get_2d_ax(length(run_info);
                                                          title=get_variable_symbol(var_name),
                                                          axis_args...)
                     for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
                         $function_name(ri, var_name; is=is, data=d, input=input, ax=a,
                                        transform=transform, colorbar_place=cp,
                                        title=ri.run_name, kwargs...)
                     end

                     if outfile !== nothing
                         save(outfile, fig)
                     end
                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str) failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, ax=nothing,
                                     colorbar_place=nothing, title=nothing,
                                     outfile=nothing, axis_args=Dict{Symbol,Any}(),
                                     it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices($(QuoteNode(dim1)),
                                                              $(QuoteNode(dim2));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = get_variable(run_info, var_name; dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim2)), $(QuoteNode(dim1));
                                         input=input, it=it, is=is, ir=ir, iz=iz,
                                         ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                                         ivz=ivz)
                 end
                 if input === nothing
                     colormap = "reverse_deep"
                 else
                     colormap = input.colormap
                 end
                 if title === nothing
                     title = get_variable_symbol(var_name)
                 end

                 if ax === nothing
                     fig, ax, colorbar_place = get_2d_ax(; title=title, axis_args...)
                     ax_was_nothing = true
                 else
                     fig = nothing
                     ax_was_nothing = false
                 end

                 x = $dim2_grid
                 if $idim2 !== nothing
                     x = x[$idim2]
                 end
                 y = $dim1_grid
                 if $idim1 !== nothing
                     y = y[$idim1]
                 end
                 plot_2d(x, y, data; ax=ax, xlabel="$($dim2_str)",
                         ylabel="$($dim1_str)", colorbar_place=colorbar_place,
                         colormap=colormap, kwargs...)

                 if input.show_element_boundaries && Symbol($dim2_str) != :t
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim2.ngrid-1:run_info.$dim2.n_global
                            if $idim2 === nothing || i ∈ $idim2]
                     element_boundary_positions = run_info.$dim2.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end
                 if input.show_element_boundaries && Symbol($dim1_str) != :t
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim1.ngrid-1:run_info.$dim1.n_global
                            if $idim1 === nothing || i ∈ $idim1]
                     element_boundary_positions = run_info.$dim1.grid[element_boundary_inds]
                     hlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end

                 if outfile !== nothing
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save(outfile, fig)
                 end

                 return fig
             end
         end)
end

# Generate 1d animation functions for each dimension
for dim ∈ one_dimension_combinations_no_t
    function_name_str = "animate_vs_$dim"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim_str = String(dim)
    dim_grid = :( run_info.$dim.grid )
    idim = Symbol(:i, dim)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, yscale=nothing,
                 $($spaces)transform=identity, ylims=nothing,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing, iz=nothing,
                 $($spaces)ivperp=nothing, ivpa=nothing, ivzeta=nothing, ivr=nothing,
                 $($spaces)ivz=nothing, kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, frame_index=nothing, ax=nothing,
                 $($spaces)fig=nothing, outfile=nothing, yscale=nothing,
                 $($spaces)transform=identity, ylims=nothing, label=nothing,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing, iz=nothing,
                 $($spaces)ivperp=nothing, ivpa=nothing, ivzeta=nothing, ivr=nothing,
                 $($spaces)ivz=nothing, kwargs...)

             Animate `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref))vs $($dim_str).

             If a Tuple of `run_info` is passed, the animations from each run are
             overlayed on the same axis, and a legend is added.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             `ylims` can be passed a Tuple (ymin, ymax) to set the y-axis limits. By
             default the minimum and maximum of the data (over all time points) will be
             used.

             `yscale` can be used to set the scaling function for the y-axis. Options are
             `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `lines!() function`.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be added to `ax`.

             When a single `run_info` is passed, `label` can be passed to set a custom
             label for the line. By default the `run_info.run_name` is used.

             `outfile` is required for animations unless `ax` is passed. The animation
             will be saved to a file named `outfile`.  The suffix determines the file
             type. If both `outfile` and `ax` are passed, then the `Figure` containing
             `ax` must be passed to `fig` to allow the animation to be saved.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case returns `nothing`.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, yscale=nothing,
                                     ylims=nothing, axis_args=Dict{Symbol,Any}(),
                                     it=nothing, $idim=nothing, kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end

                     if input === nothing
                         if run_info[1].dfns
                             if var_name ∈ keys(input_dict_dfns)
                                 input = input_dict_dfns[var_name]
                             else
                                 input = input_dict_dfns
                             end
                         else
                             if var_name ∈ keys(input_dict)
                                 input = input_dict[var_name]
                             else
                                 input = input_dict
                             end
                         end
                     end
                     if input isa AbstractDict
                         input = Dict_to_NamedTuple(input)
                     end

                     n_runs = length(run_info)

                     frame_index = Observable(1)
                     if length(run_info) == 1 ||
                         all(ri.nt == run_info[1].nt &&
                             all(isapprox.(ri.time, run_info[1].time))
                             for ri ∈ run_info[2:end])
                         # All times are the same
                         time = select_slice(run_info[1].time, :t; input=input, it=it)
                         title = lift(i->string("t = ", time[i]), frame_index)
                     else
                         title = lift(i->join((string("t", irun, " = ",
                                                      select_slice(ri.time, :t; input=input, it=it)[i])
                                               for (irun,ri) ∈ enumerate(run_info)), "; "),
                                      frame_index)
                     end
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         title=title, yscale=yscale, axis_args...)

                     for (d, ri) ∈ zip(data, run_info)
                         $function_name(ri, var_name; is=is, data=d, input=input,
                                        ylims=ylims, frame_index=frame_index, ax=ax,
                                        it=it, $idim=$idim, kwargs...)
                     end

                     if input.show_element_boundaries
                         # Just plot element boundaries from first run, assuming that all
                         # runs being compared use the same grid.
                         ri = run_info[1]
                         element_boundary_inds =
                             [i for i ∈ 1:ri.$dim.ngrid-1:ri.$dim.n_global
                                if $idim === nothing || i ∈ $idim]
                         element_boundary_positions = ri.$dim.grid[element_boundary_inds]
                         vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                     end

                     if n_runs > 1
                         put_legend_above(fig, ax)
                     end

                     if it === nothing
                         nt = minimum(ri.nt for ri ∈ run_info)
                     else
                         nt = length(it)
                     end
                     save_animation(fig, frame_index, nt, outfile)

                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str)() failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     fig=nothing, outfile=nothing, yscale=nothing,
                                     ylims=nothing, label=nothing,
                                     axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                                     iz=nothing, ivperp=nothing, ivpa=nothing,
                                     ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices(:t, $(QuoteNode(dim));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = VariableCache(run_info, var_name, chunk_size_1d;
                                          dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim)), :t; input=input, it=it,
                                         is=is, ir=ir, iz=iz, ivperp=ivperp, ivpa=ivpa,
                                         ivzeta=ivzeta, ivr=ivr, ivz=ivz)
                 end
                 if frame_index === nothing
                     ind = Observable(1)
                 else
                     ind = frame_index
                 end
                 if ax === nothing
                     time = select_slice(run_info.time, :t; input=input, it=it)
                     title = lift(i->string("t = ", time[i]), ind)
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale, title=title, axis_args...)
                 else
                     fig = nothing
                 end
                 if label === nothing
                     label = run_info.run_name
                 end

                 x = $dim_grid
                 if $idim !== nothing
                     x = x[$idim]
                 end
                 animate_1d(x, data; ax=ax, ylims=ylims, frame_index=ind,
                            label=label, kwargs...)

                 if input.show_element_boundaries && fig !== nothing
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim.ngrid-1:run_info.$dim.n_global
                            if $idim === nothing || i ∈ $idim]
                     element_boundary_positions = run_info.$dim.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                 end

                 if frame_index === nothing
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end

                     if isa(data, VariableCache)
                         nt = data.n_tinds
                     else
                         nt = size(data, 2)
                     end

                     save_animation(fig, ind, nt, outfile)
                 end

                 return fig
             end
         end)
end

# Generate 2d animation functions for all combinations of dimensions
for (dim1, dim2) ∈ two_dimension_combinations_no_t
    function_name_str = "animate_vs_$(dim2)_$(dim1)"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim1_str = String(dim1)
    dim2_str = String(dim2)
    dim1_grid = :( run_info.$dim1.grid )
    dim2_grid = :( run_info.$dim2.grid )
    idim1 = Symbol(:i, dim1)
    idim2 = Symbol(:i, dim2)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, colorscale=identity,
                 $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                 $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                 $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                 $($spaces)kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, frame_index=nothing, ax=nothing,
                 $($spaces)fig=nothing, colorbar_place=colorbar_place,
                 $($spaces)title=nothing, outfile=nothing, colorscale=identity,
                 $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                 $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                 $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                 $($spaces)kwargs...)

             Animate `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref))vs $($dim1_str) and $($dim2_str).

             If a Tuple of `run_info` is passed, the animations from each run are
             created in a horizontal row, with each sub-animation having the 'run name' as
             its subtitle.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             `colorscale` can be used to set the scaling function for the colors. Options
             are `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `heatmap!() function`.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be created in `ax`. When `ax` is passed, a colorbar will be
             created at `colorbar_place` if a `GridPosition` is passed to
             `colorbar_place`.

             `outfile` is required for animations unless `ax` is passed. The animation
             will be saved to a file named `outfile`.  The suffix determines the file
             type. If both `outfile` and `ax` are passed, then the `Figure` containing
             `ax` must be passed to `fig` to allow the animation to be saved.

             When a single `run_info` is passed, the (sub-)title can be set with the
             `title` argument.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case returns `nothing`.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, transform=identity,
                                     axis_args=Dict{Symbol,Any}(), it=nothing, kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end

                     frame_index = Observable(1)

                     if length(run_info) > 1
                         title = get_variable_symbol(var_name)
                         subtitles = (lift(i->string(ri.run_name, "\nt = ",
                                                     select_slice(ri.time, :t; input=input, it=it)[i]),
                                           frame_index)
                                      for ri ∈ run_info)
                     else
                         time = select_slice(run_info[1].time, :t; input=input, it=it)
                         title = lift(i->string(get_variable_symbol(var_name), "\nt = ",
                                                run_info[1].time[i]),
                                      frame_index)
                         subtitles = nothing
                     end
                     fig, ax, colorbar_places = get_2d_ax(length(run_info);
                                                          title=title,
                                                          subtitles=subtitles,
                                                          axis_args...)

                     for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
                         $function_name(ri, var_name; is=is, data=d, input=input,
                                        transform=transform, frame_index=frame_index,
                                        ax=a, colorbar_place=cp, it=it, kwargs...)
                     end

                     if it === nothing
                         nt = minimum(ri.nt for ri ∈ run_info)
                     else
                         nt = length(it)
                     end
                     save_animation(fig, frame_index, nt, outfile)

                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str) failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     fig=nothing, colorbar_place=nothing,
                                     title=nothing, outfile=nothing,
                                     axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                                     iz=nothing, ivperp=nothing, ivpa=nothing,
                                     ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if frame_index === nothing
                     ind = Observable(1)
                 else
                     ind = frame_index
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices(:t, $(QuoteNode(dim1)),
                                                              $(QuoteNode(dim2));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = VariableCache(run_info, var_name, chunk_size_2d;
                                          dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim2)), $(QuoteNode(dim1)), :t;
                                         input=input, it=it, is=is, ir=ir, iz=iz,
                                         ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                                         ivz=ivz)
                 end
                 if input === nothing
                     colormap = "reverse_deep"
                 else
                     colormap = input.colormap
                 end
                 if title === nothing && ax == nothing
                     time = select_slice(run_info.time, :t; input=input, it=it)
                     title = lift(i->string(get_variable_symbol(var_name), "\nt = ",
                                            run_info.time[i]),
                                  ind)
                 end

                 if ax === nothing
                     fig, ax, colorbar_place = get_2d_ax(; title=title, axis_args...)
                     ax_was_nothing = true
                 else
                     ax_was_nothing = false
                 end

                 x = $dim2_grid
                 if $idim2 !== nothing
                     x = x[$idim2]
                 end
                 y = $dim1_grid
                 if $idim1 !== nothing
                     y = y[$idim1]
                 end
                 anim = animate_2d(x, y, data; xlabel="$($dim2_str)",
                                   ylabel="$($dim1_str)", frame_index=ind, ax=ax,
                                   colorbar_place=colorbar_place, colormap=colormap,
                                   kwargs...)

                 if input.show_element_boundaries
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim2.ngrid-1:run_info.$dim2.n_global
                            if $idim2 === nothing || i ∈ $idim2]
                     element_boundary_positions = run_info.$dim2.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end
                 if input.show_element_boundaries
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim1.ngrid-1:run_info.$dim1.n_global
                            if $idim1 === nothing || i ∈ $idim1]
                     element_boundary_positions = run_info.$dim1.grid[element_boundary_inds]
                     hlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end

                 if frame_index === nothing
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end
                     if ax_was_nothing && fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     if isa(data, VariableCache)
                         nt = data.n_tinds
                     else
                         nt = size(data, 3)
                     end
                     save_animation(fig, ind, nt, outfile)
                 end

                 return fig
             end
         end)
end

"""
    get_1d_ax(n=nothing; title=nothing, subtitles=nothing, yscale=nothing,
              get_legend_place=nothing, size=nothing, kwargs...)

Create a new `Figure` `fig` and `Axis` `ax` intended for 1d plots.

`title` gives an overall title to the `Figure`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.

By default creates a single `Axis`, and returns `(fig, ax)`.
If a number of axes `n` is passed, then `ax` is a `Vector{Axis}` of length `n` (even if
`n` is 1). The axes are created in a horizontal row, and the width of the figure is
increased in proportion to `n`.

`get_legend_place` can be set to one of (:left, :right, :above, :below) to create a
`GridPosition` for a legend in the corresponding place relative to each `Axis`. If
`get_legend_place` is set, `(fig, ax, legend_place)` is returned where `legend_place` is a
`GridPosition` (if `n=nothing`) or a Tuple of `n` `GridPosition`s.

When `n` is passed, `subtitles` can be passed a Tuple of length `n` which will be used to
set a subtitle for each `Axis` in `ax`.

`size` is passed through to the `Figure` constructor. Its default value is `(600, 400)` if
`n` is not passed, or `(600*n, 400)` if `n` is passed.

Extra `kwargs` are passed to the `Axis()` constructor.
"""
function get_1d_ax(n=nothing; title=nothing, subtitles=nothing, yscale=nothing,
                   get_legend_place=nothing, size=nothing, kwargs...)
    valid_legend_places = (nothing, :left, :right, :above, :below)
    if get_legend_place ∉ valid_legend_places
        error("get_legend_place=$get_legend_place is not one of $valid_legend_places")
    end
    if yscale !== nothing
        kwargs = tuple(kwargs..., :yscale=>yscale)
    end
    if n == nothing
        if size == nothing
            size = (600, 400)
        end
        fig = Figure(size=size)
        ax = Axis(fig[1,1]; kwargs...)
        if get_legend_place === :left
            legend_place = fig[1,0]
        elseif get_legend_place === :right
            legend_place = fig[1,2]
        elseif get_legend_place === :above
            legend_place = fig[0,1]
        elseif get_legend_place === :below
            legend_place = fig[2,1]
        end
        if title !== nothing
            title_layout = fig[0,1] = GridLayout()
            Label(title_layout[1,1:2], title)
        end
    else
        if size == nothing
            size = (600*n, 400)
        end
        fig = Figure(size=size)
        plot_layout = fig[1,1] = GridLayout()

        if title !== nothing
            title_layout = fig[0,1] = GridLayout()
            Label(title_layout[1,1:2], title)
        end

        if get_legend_place === :left
            if subtitles === nothing
                ax = [Axis(plot_layout[1,2*i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,2*i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[1,2*i-1] for i in 1:n]
        elseif get_legend_place === :right
            if subtitles === nothing
                ax = [Axis(plot_layout[1,2*i-1]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,2*i-1]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[1,2*i] for i in 1:n]
        elseif get_legend_place === :above
            if subtitles === nothing
                ax = [Axis(plot_layout[2,i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[2,i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[1,i] for i in 1:n]
        elseif get_legend_place === :below
            if subtitles === nothing
                ax = [Axis(plot_layout[1,i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[2,i] for i in 1:n]
        else
            if subtitles === nothing
                ax = [Axis(plot_layout[1,i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
        end
    end

    if get_legend_place === nothing
        return fig, ax
    else
        return fig, ax, legend_place
    end
end

"""
    get_2d_ax(n=nothing; title=nothing, subtitles=nothing, size=nothing, kwargs...)

Create a new `Figure` `fig` and `Axis` `ax` intended for 2d plots.

`title` gives an overall title to the `Figure`.

By default creates a single `Axis`, and returns `(fig, ax, colorbar_place)`, where
`colorbar_place` is a location in the grid layout that can be passed to `Colorbar()`
located immediately to the right of `ax`.
If a number of axes `n` is passed, then `ax` is a `Vector{Axis}` and `colorbar_place` is a
`Vector{GridPosition}` of length `n` (even if `n` is 1). The axes are created in a
horizontal row, and the width of the figure is increased in proportion to `n`.

When `n` is passed, `subtitles` can be passed a Tuple of length `n` which will be used to
set a subtitle for each `Axis` in `ax`.

`size` is passed through to the `Figure` constructor. Its default value is `(600, 400)` if
`n` is not passed, or `(600*n, 400)` if `n` is passed.

Extra `kwargs` are passed to the `Axis()` constructor.
"""
function get_2d_ax(n=nothing; title=nothing, subtitles=nothing, size=nothing, kwargs...)
    if n == nothing
        if size == nothing
            size = (600, 400)
        end
        fig = Figure(size=size)
        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)
            irow = 2
        else
            irow = 1
        end
        ax = Axis(fig[irow,1]; kwargs...)
        colorbar_place = fig[irow,2]
    else
        if size == nothing
            size = (600*n, 400)
        end
        fig = Figure(size=size)

        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)

            plot_layout = fig[2,1] = GridLayout()
        else
            plot_layout = fig[1,1] = GridLayout()
        end
        if subtitles === nothing
            ax = [Axis(plot_layout[1,2*i-1]; kwargs...) for i in 1:n]
        else
            ax = [Axis(plot_layout[1,2*i-1]; title=st, kwargs...)
                  for (i,st) in zip(1:n, subtitles)]
        end
        colorbar_place = [plot_layout[1,2*i] for i in 1:n]
    end

    return fig, ax, colorbar_place
end

"""
    plot_1d(xcoord, data; ax=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
            yscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
            kwargs...)

Make a 1d plot of `data` vs `xcoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the plot will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created.

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `lines!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`lines!()`.
"""
function plot_1d(xcoord, data; ax=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
                 yscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
                 kwargs...)
    if ax === nothing
        fig, ax = get_1d_ax(; axis_args...)
    else
        fig = nothing
    end

    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if title !== nothing
        ax.title = title
    end

    if transform !== identity
        # Use transform to allow user to do something like data = abs.(data)
        # Don't actually apply identity transform in case this function is called with
        # `data` being a Makie Observable (in which case transform.(data) would be an
        # error).
        data = transform.(data)
    end

    l = lines!(ax, xcoord, data; kwargs...)

    if yscale !== nothing
        ax.yscale = yscale
    end

    if fig === nothing
        return l
    else
        return fig
    end
end

"""
    plot_2d(xcoord, ycoord, data; ax=nothing, colorbar_place=nothing, xlabel=nothing,
            ylabel=nothing, title=nothing, colormap="reverse_deep",
            colorscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
            kwargs...)

Make a 2d plot of `data` vs `xcoord` and `ycoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`colorscale` can be used to set the scaling function for the colors. Options are
`identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and
`Makie.Symlog10`. `transform` is a function that is applied element-by-element to the data
before it is plotted. For example when using a log scale on data that may contain some
negative values it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the plot will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created.

`colormap` is included explicitly because we do some special handling so that extra Makie
functionality can be specified by a prefix to the `colormap` string, rather than the
standard Makie mechanism of creating a struct that modifies the colormap. For example
`Reverse("deep")` can be passed as `"reverse_deep"`. This is useful so that these extra
colormaps can be specified in an input file, but is not needed for interactive use.

When `xcoord` and `ycoord` are both one-dimensional, uses Makie's `heatmap!()` function
for the plot. If either or both of `xcoord` and `ycoord` are two-dimensional, instead uses
[`irregular_heatmap!`](@ref).

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `heatmap!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`heatmap!()`.
"""
function plot_2d(xcoord, ycoord, data; ax=nothing, colorbar_place=nothing, xlabel=nothing,
                 ylabel=nothing, title=nothing, colormap="reverse_deep",
                 colorscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
                 kwargs...)
    if ax === nothing
        fig, ax, colorbar_place = get_2d_ax(; axis_args...)
    else
        fig = nothing
    end

    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if title !== nothing
        ax.title = title
    end
    colormap = parse_colormap(colormap)
    if colorscale !== nothing
        kwargs = tuple(kwargs..., :colorscale=>colorscale)
    end

    if transform !== identity
        # Use transform to allow user to do something like data = abs.(data)
        # Don't actually apply identity transform in case this function is called with
        # `data` being a Makie Observable (in which case transform.(data) would be an
        # error).
        data = transform.(data)
    end

    # Convert grid point values to 'cell face' values for heatmap
    if xcoord isa Observable
        xcoord = lift(grid_points_to_faces, xcoord)
    else
        xcoord = grid_points_to_faces(xcoord)
    end
    if ycoord isa Observable
        ycoord = lift(grid_points_to_faces, ycoord)
    else
        ycoord = grid_points_to_faces(ycoord)
    end

    if xcoord isa Observable
        ndims_x = ndims(xcoord.val)
    else
        ndims_x = ndims(xcoord)
    end
    if ycoord isa Observable
        ndims_y = ndims(ycoord.val)
    else
        ndims_y = ndims(ycoord)
    end
    if ndims_x == 1 && ndims_y == 1
        hm = heatmap!(ax, xcoord, ycoord, data; colormap=colormap, kwargs...)
    else
        hm = irregular_heatmap!(ax, xcoord, ycoord, data; colormap=colormap, kwargs...)
    end

    if colorbar_place === nothing
        println("Warning: colorbar_place argument is required to make a color bar")
    else
        Colorbar(colorbar_place, hm)
    end

    if fig === nothing
        return hm
    else
        return fig
    end
end

"""
    animate_1d(xcoord, data; frame_index=nothing, ax=nothing, fig=nothing,
               xlabel=nothing, ylabel=nothing, title=nothing, yscale=nothing,
               transform=identity, outfile=nothing, ylims=nothing,
               axis_args=Dict{Symbol,Any}(), kwargs...)

Make a 1d animation of `data` vs `xcoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`ylims` can be passed a Tuple (ymin, ymax) to set the y-axis limits. By default the
minimum and maximum of the data (over all time points) will be used.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the animation will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created. If `ax` is passed, you should also pass an
`Observable{mk_int}` to `frame_index` so that the data for this animation can be updated
when `frame_index` is changed.

If `outfile` is passed the animation will be saved to a file with that name. The suffix
determines the file type. If `ax` is passed at the same time as `outfile` then the
`Figure` containing `ax` must also be passed (to the `fig` argument) so that the animation
can be saved.

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `lines!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`lines!()`.
"""
function animate_1d(xcoord, data; frame_index=nothing, ax=nothing, fig=nothing,
                    xlabel=nothing, ylabel=nothing, title=nothing, yscale=nothing,
                    transform=identity, ylims=nothing, outfile=nothing,
                    axis_args=Dict{Symbol,Any}(), kwargs...)

    if frame_index === nothing
        ind = Observable(1)
    else
        ind = frame_index
    end

    if ax === nothing
        fig, ax = get_1d_ax(; title=title, xlabel=xlabel, ylabel=ylabel, yscale=yscale,
                            axis_args...)
    end

    if !isa(data, VariableCache)
        # Apply transform before calculating extrema
        data = transform.(data)
    end

    if ylims === nothing
        if isa(data, VariableCache)
            datamin, datamax = variable_cache_extrema(data; transform=transform)
        else
            datamin, datamax = NaNMath.extrema(data)
        end
        if ax.limits.val[2] === nothing
            # No limits set yet, need to use minimum and maximum of data over all time,
            # otherwise the automatic axis scaling would use the minimum and maximum of
            # the data at the initial time point.
            ylims!(ax, datamin, datamax)
        else
            # Expand currently set limits to ensure they include the minimum and maxiumum
            # of the data.
            current_ymin, current_ymax = ax.limits.val[2]
            ylims!(ax, min(datamin, current_ymin), max(datamax, current_ymax))
        end
    else
        # User passed ylims explicitly, so set those.
        ylims!(ax, ylims)
    end

    # Use transform to allow user to do something like data = abs.(data)
    if isa(data, VariableCache)
        line_data = @lift(transform.(get_cache_slice(data, $ind)))
    else
        line_data = @lift(@view data[:,$ind])
    end
    lines!(ax, xcoord, line_data; kwargs...)

    if outfile !== nothing
        if fig === nothing
            error("When `outfile` is passed to save the animation, must either pass both "
                  * "`fig` and `ax` or neither. Only `ax` was passed.")
        end
        nt = size(data, 2)
        save_animation(fig, ind, nt, outfile)
    end
end

"""
    animate_2d(xcoord, ycoord, data; frame_index=nothing, ax=nothing, fig=nothing,
               colorbar_place=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
               outfile=nothing, colormap="reverse_deep", colorscale=nothing,
               transform=identity, axis_args=Dict{Symbol,Any}(), kwargs...)

Make a 2d animation of `data` vs `xcoord` and `ycoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`colorscale` can be used to set the scaling function for the colors. Options are
`identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and
`Makie.Symlog10`. `transform` is a function that is applied element-by-element to the data
before it is plotted. For example when using a log scale on data that may contain some
negative values it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the animation will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created. If `ax` is passed, you should also pass an
`Observable{mk_int}` to `frame_index` so that the data for this animation can be updated
when `frame_index` is changed.

If `outfile` is passed the animation will be saved to a file with that name. The suffix
determines the file type. If `ax` is passed at the same time as `outfile` then the
`Figure` containing `ax` must also be passed (to the `fig` argument) so that the animation
can be saved.

`colormap` is included explicitly because we do some special handling so that extra Makie
functionality can be specified by a prefix to the `colormap` string, rather than the
standard Makie mechanism of creating a struct that modifies the colormap. For example
`Reverse("deep")` can be passed as `"reverse_deep"`. This is useful so that these extra
colormaps can be specified in an input file, but is not needed for interactive use.

When `xcoord` and `ycoord` are both one-dimensional, uses Makie's `heatmap!()` function
for the plot. If either or both of `xcoord` and `ycoord` are two-dimensional, instead uses
[`irregular_heatmap!`](@ref).

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `heatmap!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`heatmap!()`.
"""
function animate_2d(xcoord, ycoord, data; frame_index=nothing, ax=nothing, fig=nothing,
                    colorbar_place=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
                    outfile=nothing, colormap="reverse_deep", colorscale=nothing,
                    transform=identity, axis_args=Dict{Symbol,Any}(), kwargs...)
    colormap = parse_colormap(colormap)

    if ax === nothing
        fig, ax, colorbar_place = get_2d_ax(; title=title, axis_args...)
    end
    if frame_index === nothing
        ind = Observable(1)
    else
        ind = frame_index
    end
    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if colorscale !== nothing
        kwargs = tuple(kwargs..., :colorscale=>colorscale)
    end

    xcoord = grid_points_to_faces(xcoord)
    ycoord = grid_points_to_faces(ycoord)

    # Use transform to allow user to do something like data = abs.(data)
    if isa(data, VariableCache)
        heatmap_data = @lift(transform.(get_cache_slice(data, $ind)))
    else
        data = transform.(data)
        heatmap_data = @lift(@view data[:,:,$ind])
    end
    if ndims(xcoord) == 1 && ndims(ycoord) == 1
        hm = heatmap!(ax, xcoord, ycoord, heatmap_data; colormap=colormap, kwargs...)
    else
        hm = irregular_heatmap!(ax, xcoord, ycoord, heatmap_data; colormap=colormap, kwargs...)
    end
    Colorbar(colorbar_place, hm)

    if outfile !== nothing
        if fig === nothing
            error("When `outfile` is passed to save the animation, must either pass both "
                  * "`fig` and `ax` or neither. Only `ax` was passed.")
        end
        nt = size(data, 3)
        save_animation(fig, ind, nt, outfile)
    end

    return fig
end

"""
    save_animation(fig, frame_index, nt, outfile)

Animate `fig` and save the result in `outfile`.

`frame_index` is the `Observable{mk_int}` that updates the data used to make `fig` to a
new time point. `nt` is the total number of time points to create.

The suffix of `outfile` determines the file type.
"""
function save_animation(fig, frame_index, nt, outfile)
    record(fig, outfile, 1:nt, framerate=5) do it
        frame_index[] = it
    end
    return nothing
end

"""
   put_legend_above(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the left of a new row at the
top of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_above(fig, ax; kwargs...)
    return Legend(fig[0,1], ax; tellheight=true, tellwidth=false, kwargs...)
end

"""
   put_legend_below(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the left of a new row at the
bottom of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_below(fig, ax; kwargs...)
    return Legend(fig[end+1,1], ax; tellheight=true, tellwidth=false, kwargs...)
end

"""
   put_legend_left(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the bottom of a new column at
the left of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_left(fig, ax; kwargs...)
    return Legend(fig[end,0], ax; kwargs...)
end

"""
   put_legend_right(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the bottom of a new column at
the right of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_right(fig, ax; kwargs...)
    return Legend(fig[end,end+1], ax; kwargs...)
end

"""
    curvilinear_grid_mesh(xs, ys, zs, colors)

Tesselates the grid defined by `xs` and `ys` in order to form a mesh with per-face coloring
given by `colors`.

The grid defined by `xs` and `ys` must have dimensions `(nx, ny) == size(colors) .+ 1`, as
is the case for heatmap/image.

Code from: https://github.com/MakieOrg/Makie.jl/issues/742#issuecomment-1415809653
"""
function curvilinear_grid_mesh(xs, ys, zs, colors)
    if zs isa Observable
        nx, ny = size(zs.val)
    else
        nx, ny = size(zs)
    end
    if colors isa Observable
        ni, nj = size(colors.val)
        eltype_colors = eltype(colors.val)
    else
        ni, nj = size(colors)
        eltype_colors = eltype(colors)
    end
    @assert (nx == ni+1) & (ny == nj+1) "Expected nx, ny = ni+1, nj+1; got nx=$nx, ny=$ny, ni=$ni, nj=$nj.  nx/y are size(zs), ni/j are size(colors)."
    if xs isa Observable && ys isa Observable && zs isa Observable
        input_points_vec = lift((x, y, z)->Makie.matrix_grid(identity, x, y, z), xs, ys, zs)
    elseif xs isa Observable && ys isa Observable
        input_points_vec = lift((x, y)->Makie.matrix_grid(identity, x, y, zs), xs, ys)
    elseif ys isa Observable && zs isa Observable
        input_points_vec = lift((y, z)->Makie.matrix_grid(identity, xs, y, z), ys, zs)
    elseif xs isa Observable && zs isa Observable
        input_points_vec = lift((x, z)->Makie.matrix_grid(identity, x, ys, z), xs, zs)
    elseif xs isa Observable
        input_points_vec = lift(x->Makie.matrix_grid(identity, x, ys, zs), xs)
    elseif ys isa Observable
        input_points_vec = lift(y->Makie.matrix_grid(identity, xs, y, zs), ys)
    elseif zs isa Observable
        input_points_vec = lift(z->Makie.matrix_grid(identity, xs, ys, z), zs)
    else
        input_points_vec = Makie.matrix_grid(identity, xs, ys, zs)
    end
    if input_points_vec isa Observable
        input_points = lift(x->reshape(x, (ni, nj) .+ 1), input_points_vec)
    else
        input_points = reshape(input_points_vec, (ni, nj) .+ 1)
    end

    n_input_points = (ni + 1) * (nj + 1)

    function get_triangle_points(input_points)
        triangle_points = Vector{Point3f}()
        sizehint!(triangle_points, n_input_points * 2 * 3)
        @inbounds for j in 1:nj
            for i in 1:ni
                # push two triangles to make a square
                # first triangle
                push!(triangle_points, input_points[i, j])
                push!(triangle_points, input_points[i+1, j])
                push!(triangle_points, input_points[i+1, j+1])
                # second triangle
                push!(triangle_points, input_points[i+1, j+1])
                push!(triangle_points, input_points[i, j+1])
                push!(triangle_points, input_points[i, j])
            end
        end
        return triangle_points
    end
    if input_points isa Observable
        triangle_points = lift(get_triangle_points, input_points)
    else
        triangle_points = get_triangle_points(input_points)
    end

    function get_triangle_colors(colors)
        triangle_colors = Vector{eltype_colors}()
        sizehint!(triangle_colors, n_input_points * 2 * 3)
        @inbounds for j in 1:nj
            for i in 1:ni
                # push two triangles to make a square
                # first triangle
                push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j])
                # second triangle
                push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j])
            end
        end
        return triangle_colors
    end
    if colors isa Observable
        triangle_colors = lift(get_triangle_colors, colors)
    else
        triangle_colors = get_triangle_colors(colors)
    end

    # Triangle faces is a constant vector of indices. Note this depends on the loop
    # structure here being the same as that in get_triangle_points() and
    # get_triangle_colors()
    triangle_faces = Vector{CairoMakie.Makie.GeometryBasics.TriangleFace{UInt32}}()
    sizehint!(triangle_faces, n_input_points * 2)
    point_ind = 1
    @inbounds for j in 1:nj
        for i in 1:ni
            # push two triangles to make a square
            # first triangle
            push!(triangle_faces, CairoMakie.Makie.GeometryBasics.TriangleFace{UInt32}((point_ind, point_ind+1, point_ind+2)))
            point_ind += 3
            # second triangle
            push!(triangle_faces, CairoMakie.Makie.GeometryBasics.TriangleFace{UInt32}((point_ind, point_ind+1, point_ind+2)))
            point_ind += 3
        end
    end

    return triangle_points, triangle_faces, triangle_colors
end

"""
    irregular_heatmap(xs, ys, zs; kwargs...)

Plot a heatmap where `xs` and `ys` are allowed to define irregularly spaced, 2d grids.
`zs` gives the value in each cell of the grid.

The grid defined by `xs` and `ys` must have dimensions `(nx, ny) == size(zs) .+ 1`, as
is the case for heatmap/image.

`xs` be an array of size (nx,ny) or a vector of size (nx).

`ys` be an array of size (nx,ny) or a vector of size (ny).

`kwargs` are passed to Makie's `mesh()` function.

Code adapted from: https://github.com/MakieOrg/Makie.jl/issues/742#issuecomment-1415809653
"""
function irregular_heatmap(xs, ys, zs; kwargs...)
    fig = Figure()
    ax = Axis(fig[1,1])
    hm = irregular_heatmap!(ax, xs, ys, zs; kwargs...)

    return fig, ax, hm
end

"""
    irregular_heatmap!(ax, xs, ys, zs; kwargs...)

Plot a heatmap onto the Axis `ax` where `xs` and `ys` are allowed to define irregularly
spaced, 2d grids.  `zs` gives the value in each cell of the grid.

The grid defined by `xs` and `ys` must have dimensions `(nx, ny) == size(zs) .+ 1`, as
is the case for heatmap/image.

`xs` be an array of size (nx,ny) or a vector of size (nx).

`ys` be an array of size (nx,ny) or a vector of size (ny).

`kwargs` are passed to Makie's `mesh()` function.

Code adapted from: https://github.com/MakieOrg/Makie.jl/issues/742#issuecomment-1415809653
"""
function irregular_heatmap!(ax, xs, ys, zs; kwargs...)
    if xs isa Observable
        ndims_x = ndims(xs.val)
        if ndims_x == 1
            nx = length(xs.val)
        else
            nx = size(xs.val, 1)
        end
    else
        ndims_x = ndims(xs)
        if ndims(xs) == 1
            nx = length(xs)
        else
            nx = size(xs, 1)
        end
    end
    if ys isa Observable
        ndims_y = ndims(ys.val)
        if ndims_y == 1
            ny = length(ys.val)
        else
            ny = size(ys.val, 2)
        end
    else
        ndims_y = ndims(ys)
        if ndims_y == 1
            ny = length(ys)
        else
            ny = size(ys, 2)
        end
    end

    if zs isa Observable
        ni, nj = size(zs.val)
    else
        ni, nj = size(zs)
    end
    @assert (nx == ni+1) & (ny == nj+1) "Expected nx, ny = ni+1, nj+1; got nx=$nx, ny=$ny, ni=$ni, nj=$nj.  nx/y are size(xs)/size(ys), ni/j are size(zs)."

    if ndims_x == 1
        # Copy to an array of size (nx,ny)
        if xs isa Observable
            xs = lift(x->repeat(x, 1, ny), x)
        else
            xs = repeat(xs, 1, ny)
        end
    end
    if ndims_y == 1
        # Copy to an array of size (nx,ny)
        if ys isa Observable
            ys = lift(x->repeat(x', nx, 1), ys)
        else
            ys = repeat(ys', nx, 1)
        end
    end

    vertices, faces, colors = curvilinear_grid_mesh(xs, ys, zeros(nx, ny), zs)

    return mesh!(ax, vertices, faces; color = colors, shading = NoShading, kwargs...)
end

"""
    select_slice(variable::AbstractArray, dims::Symbol...; input=nothing, it=nothing,
                 is=1, ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                 ivzeta=nothing, ivr=nothing, ivz=nothing)

Returns a slice of `variable` that includes only the dimensions given in `dims...`, e.g.
```
select_slice(variable, :t, :r)
```
to get a two dimensional slice with t- and r-dimensions.

Any other dimensions present in `variable` have a single point selected. By default this
point is set by the options in `input` (which must be a NamedTuple) (or the final point
for time or the size of the dimension divided by 3 if `input` is not given). These
defaults can be overridden using the keyword arguments `it`, `is`, `ir`, `iz`, `ivperp`,
`ivpa`, `ivzeta`, `ivr`, `ivz`. Ranges can also be passed to these keyword arguments for
the 'kept dimensions' in `dims` to select a subset of those dimensions.

This function only recognises what the dimensions of `variable` are by the number of
dimensions in the array. It assumes that either the variable has already been sliced to
the correct dimensions (if `ndims(variable) == length(dims)` it just returns `variable`)
or that `variable` has the full number of dimensions it could have (i.e. 'field' variables
have 3 dimensions, 'moment' variables 4, 'ion distribution function' variables 6 and
'neutral distribution function' variables 7).
"""
function select_slice end

function select_slice(variable::AbstractArray{T,1}, dims::Symbol...; input=nothing,
                      is=nothing, kwargs...) where T
    if length(dims) > 1
        error("Tried to get a slice of 1d variable with dimensions $dims")
    elseif length(dims) < 1
        error("1d variable must have already been sliced, so don't know what the dimensions are")
    else
        # Array is not a standard shape, so assume it is already sliced to the right 2
        # dimensions
        return variable
    end
end

function select_slice(variable::AbstractArray{T,2}, dims::Symbol...; input=nothing,
                      is=nothing, kwargs...) where T
    if length(dims) > 2
        error("Tried to get a slice of 2d variable with dimensions $dims")
    elseif length(dims) < 2
        error("2d variable must have already been sliced, so don't know what the dimensions are")
    else
        # Array is not a standard shape, so assume it is already sliced to the right 2
        # dimensions
        return variable
    end
end

function select_slice(variable::AbstractArray{T,3}, dims::Symbol...; input=nothing,
                      it=nothing, is=nothing, ir=nothing, iz=nothing, kwargs...) where T
    # Array is (z,r,t)

    if length(dims) > 3
        error("Tried to get a slice of 3d variable with dimensions $dims")
    end

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 3)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 2) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 3, it0)
    end
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 2, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 1, iz0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,4}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, kwargs...) where T
    # Array is (z,r,species,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 4)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 2) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 4, it0)
    end
    slice = selectdim(slice, 3, is)
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 2, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 1, iz0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,5}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivperp=nothing,
                      ivpa=nothing, kwargs...) where T
    # Array is (vpa,vperp,z,r,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 5)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 4) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 3) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if ivperp !== nothing
        ivperp0 = ivperp
    elseif input === nothing || :ivperp0 ∉ input
        ivperp0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivperp0 = input.ivperp0
    end
    if ivpa !== nothing
        ivpa0 = ivpa
    elseif input === nothing || :ivpa0 ∉ input
        ivpa0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivpa0 = input.ivpa0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 5, it0)
    end
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 4, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 3, iz0)
    end
    if :vperp ∉ dims || ivperp !== nothing
        slice = selectdim(slice, 2, ivperp0)
    end
    if :vpa ∉ dims || ivpa !== nothing
        slice = selectdim(slice, 1, ivpa0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,6}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivperp=nothing,
                      ivpa=nothing, kwargs...) where T
    # Array is (vpa,vperp,z,r,species,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 6)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 4) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 3) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if ivperp !== nothing
        ivperp0 = ivperp
    elseif input === nothing || :ivperp0 ∉ input
        ivperp0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivperp0 = input.ivperp0
    end
    if ivpa !== nothing
        ivpa0 = ivpa
    elseif input === nothing || :ivpa0 ∉ input
        ivpa0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivpa0 = input.ivpa0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 6, it0)
    end
    slice = selectdim(slice, 5, is)
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 4, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 3, iz0)
    end
    if :vperp ∉ dims || ivperp !== nothing
        slice = selectdim(slice, 2, ivperp0)
    end
    if :vpa ∉ dims || ivpa !== nothing
        slice = selectdim(slice, 1, ivpa0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,7}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivzeta=nothing,
                      ivr=nothing, ivz=nothing, kwargs...) where T
    # Array is (vz,vr,vzeta,z,r,species,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 7)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 5) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 4) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if ivzeta !== nothing
        ivzeta0 = ivzeta
    elseif input === nothing || :ivzeta0 ∉ input
        ivzeta0 = max(size(variable, 3) ÷ 3, 1)
    else
        ivzeta0 = input.ivzeta0
    end
    if ivr !== nothing
        ivr0 = ivr
    elseif input === nothing || :ivr0 ∉ input
        ivr0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivr0 = input.ivr0
    end
    if ivz !== nothing
        ivz0 = ivz
    elseif input === nothing || :ivz0 ∉ input
        ivz0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivz0 = input.ivz0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 7, it0)
    end
    slice = selectdim(slice, 6, is)
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 5, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 4, iz0)
    end
    if :vzeta ∉ dims || ivzeta !== nothing
        slice = selectdim(slice, 3, ivzeta0)
    end
    if :vr ∉ dims || ivr !== nothing
        slice = selectdim(slice, 2, ivr0)
    end
    if :vz ∉ dims || ivz !== nothing
        slice = selectdim(slice, 1, ivz0)
    end

    return slice
end

"""
get_dimension_slice_indices(keep_dims...; input, it=nothing, is=nothing,
                            ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                            ivzeta=nothing, ivr=nothing, ivz=nothing)

Get indices for dimensions to slice

The indices are taken from `input`, unless they are passed as keyword arguments

The dimensions in `keep_dims` are not given a slice (those are the dimensions we want in
the variable after slicing).
"""
function get_dimension_slice_indices(keep_dims...; run_info, input, it=nothing,
                                     is=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing)
    if isa(input, AbstractDict)
        input = Dict_to_NamedTuple(input)
    end
    return (:it=>(it === nothing ? (:t ∈ keep_dims ? nothing : input.it0) : it),
            :is=>(is === nothing ? (:s ∈ keep_dims ? nothing : input.is0) : is),
            :ir=>(ir === nothing ? (:r ∈ keep_dims ? nothing : input.ir0) : ir),
            :iz=>(iz === nothing ? (:z ∈ keep_dims ? nothing : input.iz0) : iz),
            :ivperp=>(ivperp === nothing ? (:vperp ∈ keep_dims ? nothing : input.ivperp0) : ivperp),
            :ivpa=>(ivpa === nothing ? (:vpa ∈ keep_dims ? nothing : input.ivpa0) : ivpa),
            :ivzeta=>(ivzeta === nothing ? (:vzeta ∈ keep_dims ? nothing : input.ivzeta0) : ivzeta),
            :ivr=>(ivr === nothing ? (:vr ∈ keep_dims ? nothing : input.ivr0) : ivr),
            :ivz=>(ivz === nothing ? (:vz ∈ keep_dims ? nothing : input.ivz0) : ivz))
end

"""
    grid_points_to_faces(coord::AbstractVector)
    grid_points_to_faces(coord::Observable{T} where T <: AbstractVector)
    grid_points_to_faces(coord::AbstractMatrix)
    grid_points_to_faces(coord::Observable{T} where T <: AbstractMatrix)

Turn grid points in `coord` into 'cell faces'.

Returns `faces`, which has a length one greater than `coord`. The first and last values of
`faces` are the first and last values of `coord`. The intermediate values are the mid
points between grid points.
"""
function grid_points_to_faces end

function grid_points_to_faces(coord::AbstractVector)
    n = length(coord)
    faces = allocate_float(n+1)
    faces[1] = coord[1]
    for i ∈ 2:n
        faces[i] = 0.5*(coord[i-1] + coord[i])
    end
    faces[n+1] = coord[n]

    return faces
end

function grid_points_to_faces(coord::Observable{T} where T <: AbstractVector)
    n = length(coord.val)
    faces = allocate_float(n+1)
    faces[1] = coord.val[1]
    for i ∈ 2:n
        faces[i] = 0.5*(coord.val[i-1] + coord.val[i])
    end
    faces[n+1] = coord.val[n]

    return faces
end

function grid_points_to_faces(coord::AbstractMatrix)
    ni, nj = size(coord)
    faces = allocate_float(ni+1, nj+1)
    faces[1,1] = coord[1,1]
    for j ∈ 2:nj
        faces[1,j] = 0.5*(coord[1,j-1] + coord[1,j])
    end
    faces[1,nj+1] = coord[1,nj]
    for i ∈ 2:ni
        faces[i,1] = 0.5*(coord[i-1,1] + coord[i,1])
        for j ∈ 2:nj
            faces[i,j] = 0.25*(coord[i-1,j-1] + coord[i-1,j] + coord[i,j-1] + coord[i,j])
        end
        faces[i,nj+1] = 0.5*(coord[i-1,nj] + coord[i,nj])
    end
    faces[ni+1,1] = coord[ni,1]
    for j ∈ 2:nj
        faces[ni+1,j] = 0.5*(coord[ni,j-1] + coord[ni,j])
    end
    faces[ni+1,nj+1] = coord[ni,nj]

    return faces
end

function grid_points_to_faces(coord::Observable{T} where T <: AbstractMatrix)
    ni, nj = size(coord.val)
    faces = allocate_float(ni+1, nj+1)
    faces[1,1] = coord.val[1,1]
    for j ∈ 2:nj
        faces[1,j] = 0.5*(coord.val[1,j-1] + coord.val[1,j])
    end
    faces[1,nj+1] = coord.val[1,nj]
    for i ∈ 2:ni
        faces[i,1] = 0.5*(coord.val[i-1,1] + coord.val[i,1])
        for j ∈ 2:nj
            faces[i,j] = 0.25*(coord.val[i-1,j-1] + coord.val[i-1,j] + coord.val[i,j-1] + coord.val[i,j])
        end
        faces[i,nj+1] = 0.5*(coord.val[i-1,nj] + coord.val[i,nj])
    end
    faces[ni+1,1] = coord.val[ni,1]
    for j ∈ 2:nj
        faces[ni+1,j] = 0.5*(coord.val[ni,j-1] + coord.val[ni,j])
    end
    faces[ni+1,nj+1] = coord.val[ni,nj]

    return faces
end

"""
    get_variable_symbol(variable_name)

Get a symbol corresponding to a `variable_name`

For example `get_variable_symbol("phi")` returns `"ϕ"`.

If the symbol has not been defined, just return `variable_name`.
"""
function get_variable_symbol(variable_name)
    symbols_for_variables = Dict("phi"=>"ϕ", "Er"=>"Er", "Ez"=>"Ez", "density"=>"n",
                                 "parallel_flow"=>"u∥", "parallel_pressure"=>"p∥",
                                 "parallel_heat_flux"=>"q∥", "thermal_speed"=>"vth",
                                 "temperature"=>"T", "density_neutral"=>"nn",
                                 "uzeta_neutral"=>"unζ", "ur_neutral"=>"unr",
                                 "uz_neutral"=>"unz", "pzeta_neutral"=>"pnζ",
                                 "pr_neutral"=>"pnr", "pz_neutral"=>"pnz",
                                 "qzeta_neutral"=>"qnζ", "qr_neutral"=>"qnr",
                                 "qz_neutral"=>"qnz", "thermal_speed_neutral"=>"vnth",
                                 "temperature_neutral"=>"Tn")

    return get(symbols_for_variables, variable_name, variable_name)
end

"""
    parse_colormap(colormap)

Parse a `colormap` option

Allows us to have a string option which can be set in the input file and still use
Reverse, etc. conveniently.
"""
function parse_colormap(colormap)
    if colormap === nothing
        return colormap
    elseif startswith(colormap, "reverse_")
        # Use split to remove the "reverse_" prefix
        return Reverse(String(split(colormap, "reverse_", keepempty=false)[1]))
    else
        return colormap
    end
end

"""
     _get_steady_state_residual_fig_axes(n_runs)

Utility method to avoid code duplication when creating the fig_axes OrderedDict for
calculate_steady_state_residual.

`n_runs` sets the number of axes to create in each entry.
"""
function _get_steady_state_residual_fig_axes(n_runs)
    return OrderedDict(
                "RMS absolute residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="RMS absolute residual",
                                                   yscale=log10, get_legend_place=:right),
                "max absolute residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="max absolute residual",
                                                   yscale=log10, get_legend_place=:right),
                "RMS relative residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="RMS relative residual",
                                                   yscale=log10, get_legend_place=:right),
                "max relative residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="max relative residual",
                                                   yscale=log10, get_legend_place=:right))
end

# Utility method to avoid code duplication when saving the calculate_steady_state_residual
# plots
function _save_residual_plots(fig_axes, plot_prefix)
    try
        for (key, fa) ∈ fig_axes
            for (ax, lp) ∈ zip(fa[2], fa[3])
                Legend(lp, ax)
            end
            save(plot_prefix * replace(key, " "=>"_") * ".pdf", fa[1])
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in _save_residual_plots().")
    end
end

"""
calculate_steady_state_residual(run_info, variable_name; is=1, data=nothing,
                                plot_prefix=nothing, fig_axes=nothing, i_run=1)

Calculate and plot the 'residuals' for `variable_name`.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

If the variable has a species dimension, `is` selects which species to analyse.

By default the variable will be loaded from file. If the data has already been loaded, it
can be passed to `data` instead. `data` should be a Tuple of the same length as `run_info`
if `run_info` is a Tuple.

If `plot_prefix` is passed, it gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`.

`fig_axes` can be passed an OrderedDict of Tuples as returned by
[`_get_steady_state_residual_fig_axes`](@ref) - each tuple contains the Figure `fig` and
Axis or Tuple{Axis} `ax` to which to add the plot corresponding to its key. If `run_info`
is a Tuple, `ax` for each entry must be a Tuple of the same length.
"""
function calculate_steady_state_residual end

function calculate_steady_state_residual(run_info::Tuple, variable_name; is=1,
                                         data=nothing, plot_prefix=nothing,
                                         fig_axes=nothing)
    try
        n_runs = length(run_info)
        if data === nothing
            data = Tuple(nothing for _ ∈ 1:n_runs)
        end
        if fig_axes === nothing
            fig_axes = _get_steady_state_residual_fig_axes(length(run_info))
        end

        for (i, (ri, d)) ∈ enumerate(zip(run_info, data))
            calculate_steady_state_residual(ri, variable_name; is=is, data=d,
                                            fig_axes=fig_axes, i_run=i)
        end

        if plot_prefix !== nothing
            _save_residual_plots(fig_axes, plot_prefix)
        end

        return fig_axes
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in calculate_steady_state_residual().")
    end
end

function calculate_steady_state_residual(run_info, variable_name; is=1, data=nothing,
                                         plot_prefix=nothing, fig_axes=nothing,
                                         i_run=1)

    if data === nothing
        data = get_variable(run_info, variable_name; is=is)
    end

    t_dim = ndims(data)
    nt = size(data, t_dim)
    variable = selectdim(data, t_dim, 2:nt)
    variable_at_previous_time = selectdim(data, t_dim, 1:nt-1)
    dt = @views @. run_info.time[2:nt] - run_info.time[1:nt-1]
    residual_norms = steady_state_residuals(variable, variable_at_previous_time, dt)

    textoutput_file = run_info.run_prefix * "_residuals.txt"
    open(textoutput_file, "a") do io
        for (key, residual) ∈ residual_norms
            # Use lpad to get fixed-width strings to print, so we get nice columns of
            # output. 24 characters should be enough to represent any float with at
            # least a couple of spaces in front to separate columns (e.g.  "
            # -3.141592653589793e100"
            line = string((lpad(string(x), 24) for x ∈ residual)...)

            # Print to stdout as well for convenience
            println(key, ": ", line)

            line *= "  # " * variable_name
            if is !== nothing
                line *= string(is)
            end
            line *= " " * key
            println(io, line)
        end
    end

    if fig_axes === nothing
        fig_axes = _get_steady_state_residual_fig_axes(1)
    end

    t = @view run_info.time[2:end]
    with_theme(Theme(Lines=(cycle=[:color, :linestyle],))) do
        for (key, norm) ∈ residual_norms
            @views plot_1d(t, norm; label="$variable_name", ax=fig_axes[key][2][i_run])
        end
    end

    if plot_prefix !== nothing
        _save_residual_plots(fig_axes, plot_prefix)
    end

    return fig_axes
end

"""
    plot_f_unnorm_vs_vpa(run_info; input=nothing, electron=false, neutral=false,
                         it=nothing, is=1, iz=nothing, fig=nothing, ax=nothing,
                         outfile=nothing, yscale=identity, transform=identity,
                         axis_args=Dict{Symbol,Any}(), kwargs...)

Plot an unnormalized distribution function against \$v_\\parallel\$ at a fixed z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are overlayed on the same axis.

By default plots the ion distribution function. If `electron=true` is passed, plots the
electron distribution function instead. If `neutral=true` is passed, plots the neutral
distribution function instead.

`is` selects which species to analyse.

`it` and `iz` specify the indices of the time- and z-points to choose. By default they are
taken from `input`.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

If `outfile` is given, the plot will be saved to a file with that name. The suffix
determines the file type.

When `run_info` is not a Tuple, an Axis can be passed to `ax` to have the plot added to
`ax`. When `ax` is passed, if `outfile` is passed to save the plot, then the Figure
containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to [`plot_1d`](@ref).
"""
function plot_f_unnorm_vs_vpa end

function plot_f_unnorm_vs_vpa(run_info::Tuple; f_over_vpa2=false, electron=false,
                              neutral=false, outfile=nothing,
                              axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        species_label = neutral ? "n" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, axis_args...)

        for ri ∈ run_info
            plot_f_unnorm_vs_vpa(ri; f_over_vpa2=f_over_vpa2, electron=electron,
                                 neutral=neutral, ax=ax, kwargs...)
        end

        if n_runs > 1
            put_legend_above(fig, ax)
        end

        if outfile !== nothing
            save(outfile, fig)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_f_unnorm_vs_vpa().")
    end
end

function plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=false, input=nothing, electron=false,
                              neutral=false, it=nothing, is=1, iz=nothing, fig=nothing,
                              ax=nothing, outfile=nothing, transform=identity,
                              axis_args=Dict{Symbol,Any}(), kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if it == nothing
        it = input.it0
    end
    if iz == nothing
        iz = input.iz0
    end

    if ax === nothing
        species_label = neutral ? "n" : electron ? "e" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, axis_args...)
    end

    if neutral
        f = get_variable(run_info, "f_neutral"; it=it, is=is, ir=input.ir0, iz=iz,
                         ivzeta=input.ivzeta0, ivr=input.ivr0)
        density = get_variable(run_info, "density_neutral"; it=it, is=is, ir=input.ir0,
                               iz=iz)
        upar = get_variable(run_info, "uz_neutral"; it=it, is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "thermal_speed_neutral"; it=it, is=is, ir=input.ir0,
                           iz=iz)
        vcoord = run_info.vz
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = get_variable(run_info, "f$suffix"; it=it, is=is, ir=input.ir0, iz=iz,
                         ivperp=input.ivperp0)
        density = get_variable(run_info, "$(prefix)density"; it=it, is=is, ir=input.ir0, iz=iz)
        upar = get_variable(run_info, "$(prefix)parallel_flow"; it=it, is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "$(prefix)thermal_speed"; it=it, is=is, ir=input.ir0, iz=iz)
        vcoord = run_info.vpa
    end

    f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(f, vcoord.grid, density, upar, vth,
                                                run_info.evolve_density,
                                                run_info.evolve_upar,
                                                run_info.evolve_ppar)

    if f_over_vpa2
        dzdt2 = dzdt.^2
        for i ∈ eachindex(dzdt2)
            if dzdt2[i] == 0.0
                dzdt2[i] = 1.0
            end
        end
        f_unnorm ./= dzdt2
    end

    f_unnorm = transform.(f_unnorm)

    l = plot_1d(dzdt, f_unnorm; ax=ax, label=run_info.run_name, kwargs...)

    if input.show_element_boundaries && fig !== nothing
        element_boundary_inds =
        [i for i ∈ 1:run_info.vpa.ngrid-1:run_info.vpa.n_global]
        element_boundary_positions = dzdt[element_boundary_inds]
        vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
    end

    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save(outfile, fig)
    end

    if fig !== nothing
        return fig
    else
        return l
    end
end

"""
    plot_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                           it=nothing, is=1, fig=nothing, ax=nothing, outfile=nothing,
                           yscale=identity, transform=identity, rasterize=true,
                           subtitles=nothing, axis_args=Dict{Symbol,Any}(), kwargs...)

Plot unnormalized distribution function against \$v_\\parallel\$ and z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

By default plots the ion distribution function. If `electron=true` is passed, plots the
electron distribution function instead. If `neutral=true` is passed, plots the neutral
distribution function instead.

`is` selects which species to analyse.

`it` specifies the time-index to choose. By default it is taken from `input`.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

If `outfile` is given, the plot will be saved to a file with that name. The suffix
determines the file type.

When `run_info` is not a Tuple, an Axis can be passed to `ax` to have the plot created in
`ax`. When `ax` is passed, if `outfile` is passed to save the plot, then the Figure
containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`rasterize` is passed through to Makie's `mesh!()` function. The default is to rasterize
plots as vectorized plots from `mesh!()` have a very large file size. Pass `false` to keep
plots vectorized. Pass a number to increase the resolution of the rasterized plot by that
factor.

When `run_info` is a Tuple, `subtitles` can be passed a Tuple (with the same length as
`run_info`) to set the subtitle for each subplot.

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to [`plot_2d`](@ref).
"""
function plot_f_unnorm_vs_vpa_z end

function plot_f_unnorm_vs_vpa_z(run_info::Tuple; electron=false, neutral=false,
                                outfile=nothing, axis_args=Dict{Symbol,Any}(),
                                title=nothing, subtitles=nothing, kwargs...)
    try
        n_runs = length(run_info)
        if subtitles === nothing
            subtitles = Tuple(nothing for _ ∈ 1:n_runs)
        end
        if title !== nothing
            title = neutral ? L"f_{n,\mathrm{unnormalized}}" : electron ? L"f_{e,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        end
        fig, axes, colorbar_places =
            get_2d_ax(n_runs; title=title, xlabel=L"v_\parallel", ylabel=L"z",
                      axis_args...)

        for (ri, ax, colorbar_place, st) ∈ zip(run_info, axes, colorbar_places, subtitles)
            plot_f_unnorm_vs_vpa_z(ri; electron=electron, neutral=neutral, ax=ax,
                                   colorbar_place=colorbar_place, title=st, kwargs...)
        end

        if outfile !== nothing
            save(outfile, fig)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_f_unnorm_vs_vpa_z().")
    end
end

function plot_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                                it=nothing, is=1, fig=nothing, ax=nothing,
                                colorbar_place=nothing, title=nothing, outfile=nothing,
                                transform=identity, rasterize=true,
                                axis_args=Dict{Symbol,Any}(), kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if it == nothing
        it = input.it0
    end

    if ax === nothing
        if title === nothing
            title = neutral ? L"f_{n,\mathrm{unnormalized}}" : electron ? L"f_{e,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        end
        fig, ax, colorbar_place = get_2d_ax(; title=title, xlabel=L"v_\parallel",
                                            ylabel=L"z", axis_args...)
    else
        if title === nothing
            ax.title = run_info.run_name
        else
            ax.title = title
        end
    end

    if neutral
        f = get_variable(run_info, "f_neutral"; it=it, is=is, ir=input.ir0,
                         ivzeta=input.ivzeta0, ivr=input.ivr0)
        density = get_variable(run_info, "density_neutral"; it=it, is=is, ir=input.ir0)
        upar = get_variable(run_info, "uz_neutral"; it=it, is=is, ir=input.ir0)
        vth = get_variable(run_info, "thermal_speed_neutral"; it=it, is=is, ir=input.ir0)
        vpa_grid = run_info.vz.grid
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = get_variable(run_info, "f$suffix"; it=it, is=is, ir=input.ir0, ivperp=input.ivperp0)
        density = get_variable(run_info, "$(prefix)density"; it=it, is=is, ir=input.ir0)
        upar = get_variable(run_info, "$(prefix)parallel_flow"; it=it, is=is, ir=input.ir0)
        vth = get_variable(run_info, "$(prefix)thermal_speed"; it=it, is=is, ir=input.ir0)
        vpa_grid = run_info.vpa.grid
    end

    f_unnorm, z, dzdt = get_unnormalised_f_coords_2d(f, run_info.z.grid,
                                                     vpa_grid, density, upar,
                                                     vth, run_info.evolve_density,
                                                     run_info.evolve_upar,
                                                     run_info.evolve_ppar)

    f_unnorm = transform.(f_unnorm)

    # Rasterize the plot, otherwise the output files are very large
    hm = plot_2d(dzdt, z, f_unnorm; ax=ax, colorbar_place=colorbar_place,
                 rasterize=rasterize, kwargs...)

    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save(outfile, fig)
    end

    if fig !== nothing
        return fig
    else
        return hm
    end
end

"""
    animate_f_unnorm_vs_vpa(run_info; input=nothing, electron=false, neutral=false, is=1,
                            iz=nothing, fig=nothing, ax=nothing, frame_index=nothing,
                            outfile=nothing, yscale=identity, transform=identity,
                            axis_args=Dict{Symbol,Any}(), kwargs...)

Plot an unnormalized distribution function against \$v_\\parallel\$ at a fixed z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to animate is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are overlayed on the same axis.

By default animates the ion distribution function. If `electron=true` is passed, animates
the electron distribution function instead. If `neutral=true` is passed, animates the
neutral distribution function instead.

`is` selects which species to analyse.

`it` and `iz` specify the indices of the time- and z-points to choose. By default they are
taken from `input`.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

`outfile` is required for animations unless `ax` is passed. The animation will be saved to
a file named `outfile`.  The suffix determines the file type. If both `outfile` and `ax`
are passed, then the `Figure` containing `ax` must be passed to `fig` to allow the
animation to be saved.

When `run_info` is not a Tuple, an Axis can be passed to `ax` to have the plot added to
`ax`. When `ax` is passed, if `outfile` is passed to save the plot, then the Figure
containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to `lines!()` (which is used to create the plot, as we have
to handle time-varying coordinates so cannot use [`animate_1d`](@ref)).
"""
function animate_f_unnorm_vs_vpa end

function animate_f_unnorm_vs_vpa(run_info::Tuple; f_over_vpa2=false, electron=false,
                                 neutral=false, outfile=nothing,
                                 axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        frame_index = Observable(1)

        species_label = neutral ? "n" : electron ? "e" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        if length(run_info) == 1 || all(all(isapprox.(ri.time, run_info[1].time)) for ri ∈ run_info[2:end])
            # All times are the same
            title = lift(i->LaTeXString(string("t = ", run_info[1].time[i])), frame_index)
        else
            title = lift(i->LaTeXString(join((string("t", irun, " = ", ri.time[i])
                                              for (irun,ri) ∈ enumerate(run_info)), "; ")),
                         frame_index)
        end
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, title=title,
                            axis_args...)

        for ri ∈ run_info
            animate_f_unnorm_vs_vpa(ri; f_over_vpa2=f_over_vpa2, electron=electron,
                                    neutral=neutral, ax=ax, frame_index=frame_index,
                                    kwargs...)
        end

        if n_runs > 1
            put_legend_above(fig, ax)
        end

        if outfile !== nothing
            nt = minimum(ri.nt for ri ∈ run_info)
            save_animation(fig, frame_index, nt, outfile)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in animate_f_unnorm_vs_vpa().")
    end
end

function animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=false, input=nothing,
                                 electron=false, neutral=false, is=1, iz=nothing,
                                 fig=nothing, ax=nothing, frame_index=nothing,
                                 outfile=nothing, yscale=nothing, transform=identity,
                                 axis_args=Dict{Symbol,Any}(), kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if iz == nothing
        iz = input.iz0
    end

    if ax === nothing
        frame_index = Observable(1)
        title = lift(i->LaTeXString(string("t = ", run_info.time[i])), frame_index)
        species_label = neutral ? "n" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, title=title,
                            axis_args...)
    end
    if frame_index === nothing
        error("Must pass an Observable to `frame_index` when passing `ax`.")
    end

    if neutral
        f = VariableCache(run_info, "f_neutral", chunk_size_1d; it=nothing, is=is,
                          ir=input.ir0, iz=iz, ivperp=nothing, ivpa=nothing,
                          ivzeta=input.ivzeta0, ivr=input.ivr0, ivz=nothing)
        density = get_variable(run_info, "density_neutral"; is=is, ir=input.ir0, iz=iz)
        upar = get_variable(run_info, "uz_neutral"; is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "thermal_speed_neutral"; is=is, ir=input.ir0, iz=iz)
        vcoord = run_info.vz
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = VariableCache(run_info, "f$suffix", chunk_size_2d; it=nothing, is=is,
                          ir=input.ir0, iz=iz, ivperp=input.ivperp0, ivpa=nothing,
                          ivzeta=nothing, ivr=nothing, ivz=nothing)
        density = get_variable(run_info, "$(prefix)density"; is=is, ir=input.ir0, iz=iz)
        upar = get_variable(run_info, "$(prefix)parallel_flow"; is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "$(prefix)thermal_speed"; is=is, ir=input.ir0, iz=iz)
        vcoord = run_info.vpa
    end

    function get_this_f_unnorm(it)
        f_unnorm = get_unnormalised_f_1d(get_cache_slice(f, it), density[it], vth[it],
                                         run_info.evolve_density, run_info.evolve_ppar)

        if f_over_vpa2
            this_dzdt = vpagrid_to_dzdt(vcoord.grid, vth[it], upar[it],
                                        run_info.evolve_ppar, run_info.evolve_upar)
            this_dzdt2 = this_dzdt.^2
            for i ∈ eachindex(this_dzdt2)
                if this_dzdt2[i] == 0.0
                    this_dzdt2[i] = 1.0
                end
            end

            f_unnorm = @. copy(f_unnorm) / this_dzdt2
        end

        return f_unnorm
    end

    # Get extrema of dzdt
    dzdtmin = Inf
    dzdtmax = -Inf
    fmin = Inf
    fmax = -Inf
    for it ∈ 1:run_info.nt
        this_dzdt = vpagrid_to_dzdt(vcoord.grid, vth[it], upar[it],
                                    run_info.evolve_ppar, run_info.evolve_upar)
        this_dzdtmin, this_dzdtmax = extrema(this_dzdt)
        dzdtmin = min(dzdtmin, this_dzdtmin)
        dzdtmax = max(dzdtmax, this_dzdtmax)

        this_f_unnorm = get_this_f_unnorm(it)

        this_fmin, this_fmax = NaNMath.extrema(transform.(this_f_unnorm))
        fmin = min(fmin, this_fmin)
        fmax = max(fmax, this_fmax)
    end
    yheight = fmax - fmin
    xwidth = dzdtmax - dzdtmin
    if yscale ∈ (log, log10)
        # Need to calclutate y offsets differently to non-logarithmic y-axis case, to
        # ensure ymin is not negative.
        limits!(ax, dzdtmin - 0.01*xwidth, dzdtmax + 0.01*xwidth,
                fmin * (fmin/fmax)^0.01, fmax * (fmax/fmin)^0.01)
    else
        limits!(ax, dzdtmin - 0.01*xwidth, dzdtmax + 0.01*xwidth,
                fmin - 0.01*yheight, fmax + 0.01*yheight)
    end

    dzdt = @lift vpagrid_to_dzdt(vcoord.grid, vth[$frame_index], upar[$frame_index],
                                 run_info.evolve_ppar, run_info.evolve_upar)
    f_unnorm = @lift transform.(get_this_f_unnorm($frame_index))

    l = plot_1d(dzdt, f_unnorm; ax=ax, label=run_info.run_name, yscale=yscale, kwargs...)

    if input.show_element_boundaries && fig !== nothing
        element_boundary_inds =
        [i for i ∈ 1:run_info.vpa.ngrid-1:run_info.vpa.n_global]
        element_boundary_positions = @lift $dzdt[element_boundary_inds]
        vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
    end


    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save_animation(fig, frame_index, run_info.nt, outfile)
    end

    if fig !== nothing
        return fig
    else
        return l
    end
end

"""
    animate_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                              is=1, fig=nothing, ax=nothing, frame_index=nothing,
                              outfile=nothing, yscale=identity, transform=identity,
                              axis_args=Dict{Symbol,Any}(), kwargs...)

Animate an unnormalized distribution function against \$v_\\parallel\$ and z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

By default animates the ion distribution function. If `electron=true` is passed, animates
the electron distribution function instead. If `neutral=true` is passed, animates the
neutral distribution function instead.

`is` selects which species to analyse.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

`outfile` is required for animations unless `ax` is passed. The animation will be saved to
a file named `outfile`.  The suffix determines the file type. If both `outfile` and `ax`
are passed, then the `Figure` containing `ax` must be passed to `fig` to allow the
animation to be saved.

When `run_info` is not a Tuple, an Axis can be passed to `ax` to have the animation
created in `ax`. When `ax` is passed, if `outfile` is passed to save the animation, then
the Figure containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to [`plot_2d`](@ref) (which is used to create the plot, as
we have to handle time-varying coordinates so cannot use [`animate_2d`](@ref)).
"""
function animate_f_unnorm_vs_vpa_z end

function animate_f_unnorm_vs_vpa_z(run_info::Tuple; electron=false, neutral=false,
                                   outfile=nothing, axis_args=Dict{Symbol,Any}(),
                                   kwargs...)
    try
        n_runs = length(run_info)

        frame_index = Observable(1)

        var_name = neutral ? L"f_{n,\mathrm{unnormalized}}" : electron ? L"f_{e,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        if length(run_info) > 1
            title = var_name
            subtitles = (lift(i->LaTeXString(string(ri.run_name, "\nt = ", ri.time[i])),
                              frame_index)
                         for ri ∈ run_info)
        else
            title = lift(i->LaTeXString(string(var_name, L",\;t = ",
                                               run_info[1].time[i])),
                         frame_index)
            subtitles = nothing
        end
        fig, axes, colorbar_places = get_2d_ax(n_runs; title=title, subtitles=subtitles,
                                               xlabel=L"v_\parallel", ylabel=L"z",
                                               axis_args...)

        for (ri, ax, colorbar_place) ∈ zip(run_info, axes, colorbar_places)
            animate_f_unnorm_vs_vpa_z(ri; electron=electron, neutral=neutral, ax=ax,
                                      colorbar_place=colorbar_place, frame_index=frame_index,
                                      kwargs...)
        end

        if outfile !== nothing
            nt = minimum(ri.nt for ri ∈ run_info)
            save_animation(fig, frame_index, nt, outfile)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in animate_f_unnorm_vs_vpa_z().")
    end
end

function animate_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                                   is=1, fig=nothing, ax=nothing, colorbar_place=nothing,
                                   frame_index=nothing, outfile=nothing,
                                   transform=identity, axis_args=Dict{Symbol,Any}(),
                                   kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if ax === nothing
        frame_index = Observable(1)
        var_name = neutral ? L"f_{n,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        title = lift(i->LaTeXString(string(var_name, "\nt = ", run_info.time[i])),
                     frame_index)
        fig, ax, colorbar_place = get_2d_ax(; title=title, xlabel=L"v_\parallel",
                                            ylabel=L"z", axis_args...)
    end
    if frame_index === nothing
        error("Must pass an Observable to `frame_index` when passing `ax`.")
    end

    if neutral
        f = VariableCache(run_info, "f_neutral", chunk_size_2d; it=nothing, is=is,
                          ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                          ivzeta=input.ivzeta0, ivr=input.ivr0, ivz=nothing)
        density = VariableCache(run_info, "density_neutral", chunk_size_1d; it=nothing,
                                is=is, ir=input.ir0, iz=nothing, ivperp=nothing,
                                ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing)
        upar = VariableCache(run_info, "uz_neutral", chunk_size_1d; it=nothing, is=is,
                             ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                             ivzeta=nothing, ivr=nothing, ivz=nothing)
        vth = VariableCache(run_info, "thermal_speed_neutral", chunk_size_1d; it=nothing,
                            is=is, ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                            ivzeta=nothing, ivr=nothing, ivz=nothing)
        vpa_grid = run_info.vz.grid
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = VariableCache(run_info, "f$suffix", chunk_size_2d; it=nothing, is=is,
                          ir=input.ir0, iz=nothing, ivperp=input.ivperp0, ivpa=nothing,
                          ivzeta=nothing, ivr=nothing, ivz=nothing)
        density = VariableCache(run_info, "$(prefix)density", chunk_size_1d; it=nothing,
                                is=is, ir=input.ir0, iz=nothing, ivperp=nothing,
                                ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing)
        upar = VariableCache(run_info, "$(prefix)parallel_flow", chunk_size_1d;
                             it=nothing, is=is, ir=input.ir0, iz=nothing, ivperp=nothing,
                             ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing)
        vth = VariableCache(run_info, "$(prefix)thermal_speed", chunk_size_1d; it=nothing,
                            is=is, ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                            ivzeta=nothing, ivr=nothing, ivz=nothing)
        vpa_grid = run_info.vpa.grid
    end

    # Get extrema of dzdt
    dzdtmin = Inf
    dzdtmax = -Inf
    for it ∈ 1:run_info.nt
        this_dzdt = vpagrid_to_dzdt_2d(vpa_grid, get_cache_slice(vth, it),
                                       get_cache_slice(upar, it), run_info.evolve_ppar,
                                       run_info.evolve_upar)
        this_dzdtmin, this_dzdtmax = extrema(this_dzdt)
        dzdtmin = min(dzdtmin, this_dzdtmin)
        dzdtmax = max(dzdtmax, this_dzdtmax)
    end
    # Set x-limits of ax so that plot always fits within axis
    xlims!(ax, dzdtmin, dzdtmax)

    dzdt = @lift vpagrid_to_dzdt_2d(vpa_grid, get_cache_slice(vth, $frame_index),
                                    get_cache_slice(upar, $frame_index),
                                    run_info.evolve_ppar, run_info.evolve_upar)
    f_unnorm = @lift transform.(get_unnormalised_f_2d(
                                    get_cache_slice(f, $frame_index),
                                    get_cache_slice(density, $frame_index),
                                    get_cache_slice(vth, $frame_index),
                                    run_info.evolve_density, run_info.evolve_ppar))

    hm = plot_2d(dzdt, run_info.z.grid, f_unnorm; ax=ax, colorbar_place=colorbar_place,
                 kwargs...)

    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save_animation(fig, frame_index, run_info.nt, outfile)
    end

    if fig !== nothing
        return fig
    else
        return hm
    end
end

"""
    plot_charged_pdf_2D_at_wall(run_info; plot_prefix, electron=false)

Make plots/animations of the ion distribution function at wall boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where line
plots/animations from the different runs are overlayed on the same axis, and heatmap
plots/animations are displayed in a horizontal row.

Settings are read from the `[wall_pdf]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`. When `run_info`
is not a Tuple, `plot_prefix` is optional - plots/animations will be saved only if it is
passed.

If `electron=true` is passed, plot electron distribution function instead of ion
distribution function.
"""
function plot_charged_pdf_2D_at_wall(run_info; plot_prefix, electron=false)
    try
        if electron
            electron_prefix = "electron_"
            electron_suffix = "_electron"
        else
            electron_prefix = ""
            electron_suffix = ""
        end
        input = Dict_to_NamedTuple(input_dict_dfns["wall_pdf$electron_suffix"])
        if !(input.plot || input.animate || input.advection_velocity)
            # nothing to do
            return nothing
        end
        if !any(ri !== nothing for ri ∈ run_info)
            println("Warning: no distribution function output, skipping wall_pdf plots")
            return nothing
        end

        z_lower = 1
        z_upper = run_info[1].z.n
        if !all(ri.z.n == z_upper for ri ∈ run_info)
            println("Cannot run plot_charged_pdf_2D_at_wall() for runs with different "
                    * "z-grid sizes. Got $(Tuple(ri.z.n for ri ∈ run_info))")
            return nothing
        end

        if electron
            println("Making plots of electron distribution function at walls")
        else
            println("Making plots of ion distribution function at walls")
        end
        flush(stdout)

        has_rdim = any(ri !== nothing && ri.r.n > 1 for ri ∈ run_info)
        has_zdim = any(ri !== nothing && ri.z.n > 1 for ri ∈ run_info)
        is_1V = all(ri !== nothing && ri.vperp.n == 1 for ri ∈ run_info)
        moment_kinetic = !electron &&
                         any(ri !== nothing
                             && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                             for ri ∈ run_info)

        nt = minimum(ri.nt for ri ∈ run_info)

        for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+4, "wall-"),
                                   (z_upper, z_upper-4:z_upper, "wall+"))
            f_input = copy(input_dict_dfns["f"])
            f_input["iz0"] = z

            if input.plot
                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f$electron_suffix")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                    label="$(run_label)iz=$iz", ax=ax)
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa.pdf"
                save(outfile, fig)

                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                    label="$(run_label)iz=$iz", ax=ax, yscale=log10,
                                    transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "logpdf$(electron_suffix)_$(label)_vs_vpa.pdf"
                save(outfile, fig)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vpa_unnorm", ylabel="f$(electron_suffix)_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; input=f_input, is=1, iz=iz,
                                                 label="$(run_label)iz=$iz", ax=ax)
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "pdf_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)

                    fig, ax = get_1d_ax(; xlabel="vpa_unnorm", ylabel="f_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; input=f_input, is=1, iz=iz,
                                                 label="$(run_label)iz=$iz", ax=ax, yscale=log10,
                                                 transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "logpdf_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)
                end

                if !is_1V
                    plot_vs_vpa_vperp(run_info, "f$electron_suffix"; is=1, input=f_input,
                                      outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_vperp.pdf")
                end

                if has_zdim
                    plot_vs_vpa_z(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                  outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_z.pdf")
                end

                if has_rdim && has_zdim
                    plot_vs_z_r(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_z_r.pdf")
                end

                if has_rdim
                    plot_vs_vpa_r(run_info, "f$electron_suffix"; is=1, input=f_input,
                                  outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_r.pdf")
                end
            end

            if input.animate
                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f$electron_suffix")
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                       label="$(run_label)iz=$iz", ax=ax,
                                       frame_index=frame_index)
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f$electron_suffix", yscale=log10)
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                       label="$(run_label)iz=$iz", ax=ax,
                                       frame_index=frame_index,
                                       transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "logpdf$(electron_suffix)_$(label)_vs_vpa." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; is=1, iz=iz, input=f_input,
                                                    label="$(run_label)iz=$iz", ax=ax,
                                                    frame_index=frame_index)
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "pdf_unnorm_$(label)_vs_vpa." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)

                    fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; is=1, iz=iz, input=f_input,
                                                    label="$(run_label)iz=$iz", ax=ax,
                                                    frame_index=frame_index, yscale=log10,
                                                    transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "logpdf_unnorm_$(label)_vs_vpa." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)
                end

                if !is_1V
                    animate_vs_vpa_vperp(run_info, "f$electron_suffix"; is=1, input=f_input,
                                         outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_vperp." * input.animation_ext)
                end

                if has_zdim
                    animate_vs_vpa_z(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                     outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_z." * input.animation_ext)
                end

                if has_rdim && has_zdim
                    animate_vs_z_r(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                   outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_z_r." * input.animation_ext)
                end

                if has_rdim
                    animate_vs_vpa_r(run_info, "f$electron_suffix"; is=1, input=f_input,
                                     outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_r." * input.animation_ext)
                end
            end

            if input.advection_velocity
                animate_vs_vpa(run_info, "$(electron_prefix)vpa_advect_speed"; is=1, input=f_input,
                               outfile=plot_prefix * "$(electron_prefix)vpa_advect_speed_$(label)_vs_vpa." * input.animation_ext)
            end
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_charged_pdf_2D_at_wall().")
    end

    return nothing
end

"""
    plot_neutral_pdf_2D_at_wall(run_info; plot_prefix)

Make plots/animations of the neutral particle distribution function at wall boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where line
plots/animations from the different runs are overlayed on the same axis, and heatmap
plots/animations are displayed in a horizontal row.

Settings are read from the `[wall_pdf_neutral]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`. When `run_info`
is not a Tuple, `plot_prefix` is optional - plots/animations will be saved only if it is
passed.
"""
function plot_neutral_pdf_2D_at_wall(run_info; plot_prefix)
    try
        input = Dict_to_NamedTuple(input_dict_dfns["wall_pdf_neutral"])
        if !(input.plot || input.animate || input.advection_velocity)
            # nothing to do
            return nothing
        end
        if !any(ri !== nothing for ri ∈ run_info)
            println("Warning: no distribution function output, skipping wall_pdf plots")
            return nothing
        end

        z_lower = 1
        z_upper = run_info[1].z.n
        if !all(ri.z.n == z_upper for ri ∈ run_info)
            println("Cannot run plot_neutral_pdf_2D_at_wall() for runs with different "
                    * "z-grid sizes. Got $(Tuple(ri.z.n for ri ∈ run_info))")
            return nothing
        end

        println("Making plots of neutral distribution function at walls")
        flush(stdout)

        has_rdim = any(ri !== nothing && ri.r.n > 1 for ri ∈ run_info)
        has_zdim = any(ri !== nothing && ri.z.n > 1 for ri ∈ run_info)
        is_1V = all(ri !== nothing && ri.vzeta.n == 1 && ri.vr.n == 1 for ri ∈ run_info)
        moment_kinetic = any(ri !== nothing
                             && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                             for ri ∈ run_info)
        nt = minimum(ri.nt for ri ∈ run_info)

        for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+4, "wall-"),
                                   (z_upper, z_upper-4:z_upper, "wall+"))
            f_neutral_input = copy(input_dict_dfns["f_neutral"])
            f_neutral_input["iz0"] = z

            if input.plot
                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                   label="$(run_label)iz=$iz", ax=ax)
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz.pdf"
                save(outfile, fig)

                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                   label="$(run_label)iz=$iz", ax=ax, yscale=log10,
                                   transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "logpdf_neutral_$(label)_vs_vpa.pdf"
                save(outfile, fig)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vz_unnorm", ylabel="f_neutral_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; neutral=true, input=f_neutral_input,
                                                 is=1, iz=iz, label="$(run_label)iz=$iz",
                                                 ax=ax)
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "pdf_neutral_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)

                    fig, ax = get_1d_ax(; xlabel="vz_unnorm", ylabel="f_neutral_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; neutral=true, input=f_neutral_input,
                                                 is=1, iz=iz, label="$(run_label)iz=$iz",
                                                 ax=ax, yscale=log10,
                                                 transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "logpdf_neutral_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)
                end

                if !is_1V
                    plot_vs_vzeta_vr(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_vzeta.pdf")
                    plot_vs_vzeta_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vzeta.pdf")
                    plot_vs_vr_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                  outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vr.pdf")
                end

                if has_zdim
                    plot_vs_vz_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_z.pdf")
                end

                if has_zdim && !is_1V
                    plot_vs_vzeta_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_z.pdf")
                    plot_vs_vr_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_z.pdf")
                end

                if has_rdim && has_zdim
                    plot_vs_z_r(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                outfile=plot_prefix * "pdf_neutral_$(label)_vs_z_r.pdf")
                end

                if has_rdim
                    plot_vs_vz_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_r.pdf")
                    if !is_1V
                        plot_vs_vzeta_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_r.pdf")

                        plot_vs_vr_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_r.pdf")
                    end
                end
            end

            if input.animate
                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                      label="$(run_label)iz=$iz", ax=ax,
                                      frame_index=frame_index)
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral", yscale=log10)
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                      label="$(run_label)iz=$iz", ax=ax,
                                      frame_index=frame_index,
                                      transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_right(fig, ax)
                outfile=plot_prefix * "logpdf_neutral_$(label)_vs_vz." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; neutral=true, is=1, iz=iz,
                                                    input=f_neutral_input,
                                                    label="$(run_label)iz=$iz", ax=ax,
                                                    frame_index=frame_index)
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "pdf_neutral_unnorm_$(label)_vs_vz." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)

                    fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; neutral=true, is=1, iz=iz,
                                                    input=f_neutral_input, label="$(run_label)iz=$iz",
                                                    ax=ax, frame_index=frame_index, yscale=log10,
                                                    transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_right(fig, ax)
                    outfile=plot_prefix * "logpdf_neutral_unnorm_$(label)_vs_vz." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)
                end

                if !is_1V
                    animate_vs_vzeta_vr(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_vzeta." * input.animation_ext)
                    animate_vs_vzeta_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vzeta." * input.animation_ext)
                    animate_vs_vr_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vr." * input.animation_ext)
                end

                if has_zdim
                    animate_vs_vz_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_z." * input.animation_ext)
                end

                if has_zdim && !is_1V
                    animate_vs_vzeta_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                       outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_z." * input.animation_ext)
                    animate_vs_vr_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_z." * input.animation_ext)
                end

                if has_rdim && has_zdim
                    animate_vs_z_r(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                   outfile=plot_prefix * "pdf_neutral_$(label)_vs_z_r." * input.animation_ext)
                end

                if has_rdim
                    animate_vs_vz_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_r." * input.animation_ext)
                    if !is_1V
                        animate_vs_vzeta_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                           outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_r." * input.animation_ext)
                        animate_vs_vr_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_r." * input.animation_ext)
                    end
                end
            end

            if input.advection_velocity
                animate_vs_vz(run_info, "neutral_vz_advect_speed"; is=1, input=f_neutral_input,
                              outfile=plot_prefix * "neutral_vz_advect_speed_$(label)_vs_vz." * input.animation_ext)
            end
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_neutral_pdf_2D_at_wall().")
    end

    return nothing
end

"""
    constraints_plots(run_info; plot_prefix=plot_prefix)

Plot and/or animate the coefficients used to correct the normalised distribution
function(s) (aka shape functions) to obey the moment constraints.

If there were no discretisation errors, we would have \$A=1\$, \$B=0\$, \$C=0\$. The
plots/animations show \$(A-1)\$ so that all three coefficients can be shown nicely on the
same axes.
"""
function constraints_plots(run_info; plot_prefix=plot_prefix)
    input = Dict_to_NamedTuple(input_dict["constraints"])

    if !(input.plot || input.animate)
        return nothing
    end

    try
        println("Making plots of moment constraints coefficients")

        if !isa(run_info, Tuple)
            run_info = (run_info,)
        end

        it0 = input.it0
        ir0 = input.ir0

        if input.plot
            if any(ri.evolve_density || ri.evolve_upar || ri.evolve_ppar
                   for ri ∈ run_info)

                # Ions
                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                        continue
                    end
                    nspecies = ri.n_ion_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "ion_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; it=it0, is=is, ir=ir0)
                        data .-= 1.0
                        plot_vs_z(ri, varname; label=label, data=data, ax=ax, input=input)

                        varname = "ion_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)

                        varname = "ion_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)
                    end
                end
                put_legend_right(fig, ax)
                save(plot_prefix * "ion_constraints.pdf", fig)
            end

            # Neutrals
            if any(ri.n_neutral_species > 1
                   && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                   for ri ∈ run_info)

                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                        continue
                    end
                    nspecies = ri.n_neutral_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "neutral_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; it=it0, is=is, ir=ir0)
                        data .-= 1.0
                        plot_vs_z(ri, varname; label=label, data=data, ax=ax, input=input)

                        varname = "neutral_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)

                        varname = "neutral_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)
                    end
                end
                put_legend_right(fig, ax)
                save(plot_prefix * "neutral_constraints.pdf", fig)
            end

            # Electrons
            #if any(ri.composition.electron_physics ∈ (kinetic_electrons,
            #                                          kinetic_electrons_with_temperature_equation)
            #       for ri ∈ run_info)

            #    fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")
            #    for ri ∈ run_info
            #        if length(run_info) > 1
            #            prefix = ri.run_name * ", "
            #        else
            #            prefix = ""
            #        end

            #        varname = "electron_constraints_A_coefficient"
            #        label = prefix * "(A-1)"
            #        data = get_variable(ri, varname; it=it0, ir=ir0)
            #        data .-= 1.0
            #        plot_vs_z(ri, varname; label=label, data=data, ax=ax, input=input)

            #        varname = "electron_constraints_B_coefficient"
            #        label = prefix * "B"
            #        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, ir=ir0,
            #                  input=input)

            #        varname = "electron_constraints_C_coefficient"
            #        label = prefix * "C"
            #        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, ir=ir0,
            #                  input=input)
            #    end
            #    put_legend_right(fig, ax)
            #    save(plot_prefix * "electron_constraints.pdf", fig)
            #end
        end

        if input.animate
            nt = minimum(ri.nt for ri ∈ run_info)

            if any(ri.evolve_density || ri.evolve_upar || ri.evolve_ppar
                   for ri ∈ run_info)

                # Ions
                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")

                # Calculate plot limits manually so we can exclude the first time point, which
                # often has a large value for (A-1) due to the way initialisation is done,
                # which can make the subsequent values hard to see.
                ymin = Inf
                ymax = -Inf
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                        continue
                    end
                    nspecies = ri.n_ion_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "ion_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        data .-= 1.0
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, input=input)

                        varname = "ion_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)

                        varname = "ion_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)
                    end
                end
                put_legend_right(fig, ax)
                ylims!(ax, ymin, ymax)
                save_animation(fig, frame_index, nt,
                               plot_prefix * "ion_constraints." * input.animation_ext)
            end

            # Neutrals
            if any(ri.n_neutral_species > 1
                   && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                   for ri ∈ run_info)

                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")

                # Calculate plot limits manually so we can exclude the first time point, which
                # often has a large value for (A-1) due to the way initialisation is done,
                # which can make the subsequent values hard to see.
                ymin = Inf
                ymax = -Inf
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                        continue
                    end
                    nspecies = ri.n_neutral_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "neutral_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        data .-= 1.0
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, input=input)

                        varname = "neutral_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)

                        varname = "neutral_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)
                    end
                end
                put_legend_right(fig, ax)
                ylims!(ax, ymin, ymax)
                save_animation(fig, frame_index, nt,
                               plot_prefix * "neutral_constraints." * input.animation_ext)
            end

            # Electrons
            #if any(ri.composition.electron_physics ∈ (kinetic_electrons,
            #                                          kinetic_electrons_with_temperature_equation)
            #       for ri ∈ run_info)

            #    frame_index = Observable(1)
            #    fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")

            #    # Calculate plot limits manually so we can exclude the first time point, which
            #    # often has a large value for (A-1) due to the way initialisation is done,
            #    # which can make the subsequent values hard to see.
            #    ymin = Inf
            #    ymax = -Inf
            #    for ri ∈ run_info
            #        if length(run_info) > 1
            #            prefix = ri.run_name * ", "
            #        else
            #            prefix = ""
            #        end

            #        varname = "electron_constraints_A_coefficient"
            #        label = prefix * "(A-1)"
            #        data = get_variable(ri, varname; ir=ir0)
            #        data .-= 1.0
            #        ymin = min(ymin, minimum(data[:,2:end]))
            #        ymax = max(ymax, maximum(data[:,2:end]))
            #        animate_vs_z(ri, varname; label=label, data=data,
            #                     frame_index=frame_index, ax=ax, input=input)

            #        varname = "electron_constraints_B_coefficient"
            #        label = prefix * "B"
            #        data = get_variable(ri, varname; ir=ir0)
            #        ymin = min(ymin, minimum(data[:,2:end]))
            #        ymax = max(ymax, maximum(data[:,2:end]))
            #        animate_vs_z(ri, varname; label=label, data=data,
            #                     frame_index=frame_index, ax=ax, ir=ir0, input=input)

            #        varname = "electron_constraints_C_coefficient"
            #        label = prefix * "C"
            #        data = get_variable(ri, varname; ir=ir0)
            #        ymin = min(ymin, minimum(data[:,2:end]))
            #        ymax = max(ymax, maximum(data[:,2:end]))
            #        animate_vs_z(ri, varname; label=label, data=data,
            #                     frame_index=frame_index, ax=ax, ir=ir0, input=input)
            #    end
            #    put_legend_right(fig, ax)
            #    ylims!(ax, ymin, ymax)
            #    save_animation(fig, frame_index, nt,
            #                   plot_prefix * "electron_constraints." * input.animation_ext)
            #end
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in constraints_plots().")
    end
end

"""
    Chodura_condition_plots(run_info::Tuple; plot_prefix)
    Chodura_condition_plots(run_info; plot_prefix=nothing, axes=nothing)

Plot the criterion from the Chodura condition at the sheath boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where line
plots from the different runs are overlayed on the same axis, and heatmap plots are
displayed in a horizontal row.

Settings are read from the `[Chodura_condition]` section of the input.

When `run_info` is a Tuple, `plot_prefix` is required and gives the path and prefix for
plots to be saved to. They will be saved with the format
`plot_prefix<some_identifying_string>.pdf`. When `run_info` is not a Tuple, `plot_prefix`
is optional - plots will be saved only if it is passed.

When `run_info` is not a Tuple, a Vector of Axis objects can be passed to `axes`, and each
plot will be added to one of `axes`.
"""
function Chodura_condition_plots end

function Chodura_condition_plots(run_info::Tuple; plot_prefix)
    input = Dict_to_NamedTuple(input_dict_dfns["Chodura_condition"])

    if !any(v for (k,v) ∈ pairs(input) if startswith(String(k), "plot"))
        # No plots to make here
        return nothing
    end
    if !any(ri !== nothing for ri ∈ run_info)
        println("Warning: no distribution function output, skipping Chodura "
                * "condition plots")
        return nothing
    end

    try
        println("Making Chodura condition plots")
        flush(stdout)

        n_runs = length(run_info)

        if n_runs == 1
            Chodura_condition_plots(run_info[1], plot_prefix=plot_prefix)
            return nothing
        end

        figs = []
        axes = Tuple([] for _ ∈ run_info)
        if input.plot_vs_t
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="time",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end

            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="time",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.plot_vs_r
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="r",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end

            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="r",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.plot_vs_r_t
            fig, ax, colorbar_place = get_2d_ax(n_runs; title="Chodura ratio at z=-L/2",
                                                xlabel="r", ylabel="time")
            push!(figs, fig)
            for (a, b, cbp) ∈ zip(axes, ax, colorbar_place)
                push!(a, (b, cbp))
            end

            fig, ax, colorbar_place = get_2d_ax(n_runs; title="Chodura ratio at z=+L/2",
                                                xlabel="r", ylabel="time")
            push!(figs, fig)
            for (a, b, cbp) ∈ zip(axes, ax, colorbar_place)
                push!(a, (b, cbp))
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.plot_f_over_vpa2
            println("going to plot f_over_vpa2")
            fig, ax = get_1d_ax(title="f/vpa^2 lower wall", xlabel="vpa", ylabel="f / vpa^2")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end

            fig, ax = get_1d_ax(title="f/vpa^2 upper wall", xlabel="vpa", ylabel="f / vpa^2")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.animate_f_over_vpa2
            fig, ax = get_1d_ax(title="f/vpa^2 lower wall", xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            push!(figs, fig)
            for a ∈ axes
                push!(a, (ax, frame_index))
            end

            fig, ax = get_1d_ax(title="f/vpa^2 upper wall", xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            push!(figs, fig)
            for a ∈ axes
                push!(a, (ax, frame_index))
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end

        for (ri, ax) ∈ zip(run_info, axes)
            Chodura_condition_plots(ri; axes=ax)
        end

        if input.plot_vs_t
            fig = figs[1]
            ax = axes[1][1]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_t.pdf")
            save(outfile, fig)

            fig = figs[2]
            ax = axes[1][2]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_t.pdf")
            save(outfile, fig)
        end
        if input.plot_vs_r
            fig = figs[3]
            ax = axes[1][3]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r.pdf")
            save(outfile, fig)

            fig = figs[4]
            ax = axes[1][4]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r.pdf")
            save(outfile, fig)
        end
        if input.plot_vs_r_t
            fig = figs[5]
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r_t.pdf")
            save(outfile, fig)

            fig = figs[6]
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r_t.pdf")
            save(outfile, fig)
        end
        if input.plot_f_over_vpa2
            fig = figs[7]
            println("check axes ", axes)
            ax = axes[1][7]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall-_vs_vpa.pdf")
            save(outfile, fig)

            fig = figs[8]
            ax = axes[1][8]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall+_vs_vpa.pdf")
            save(outfile, fig)
        end
        if input.animate_f_over_vpa2
            nt = minimum(ri.nt for ri ∈ run_info)

            fig = figs[9]
            ax = axes[1][9][1]
            frame_index = axes[1][9][2]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall-_vs_vpa." * input.animation_ext)
            save_animation(fig, frame_index, nt, outfile)

            fig = figs[10]
            ax = axes[1][10][1]
            frame_index = axes[1][10][2]
            put_legend_below(fig, ax)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall+_vs_vpa." * input.animation_ext)
            save_animation(fig, frame_index, nt, outfile)
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in Chodura_condition_plots().")
    end

    return nothing
end

function Chodura_condition_plots(run_info; plot_prefix=nothing, axes=nothing)

    if run_info === nothing
        println("In Chodura_condition_plots(), run_info===nothing so skipping")
        return nothing
    end
    if run_info.z.bc != "wall"
        println("In Chodura_condition_plots(), z.bc!=\"wall\" - there is no wall - so "
                * "skipping")
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["Chodura_condition"])

    time = run_info.time
    density = get_variable(run_info, "density")
    upar = get_variable(run_info, "parallel_flow")
    vth = get_variable(run_info, "thermal_speed")
    temp_e = get_variable(run_info, "electron_temperature")
    Er = get_variable(run_info, "Er")
    f_lower = get_variable(run_info, "f", iz=1)
    f_upper = get_variable(run_info, "f", iz=run_info.z.n_global)

    Chodura_ratio_lower, Chodura_ratio_upper, cutoff_lower, cutoff_upper =
        check_Chodura_condition(run_info.r_local, run_info.z_local, run_info.vperp,
                                run_info.vpa, density, upar, vth, temp_e,
                                run_info.composition, Er, run_info.geometry,
                                run_info.z.bc, nothing;
                                evolve_density=run_info.evolve_density,
                                evolve_upar=run_info.evolve_upar,
                                evolve_ppar=run_info.evolve_ppar,
                                f_lower=f_lower, f_upper=f_upper, find_extra_offset=true)

    if input.plot_vs_t
        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="time",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[1]
        end
        plot_1d(time, Chodura_ratio_lower[input.ir0,:], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_t.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="time",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[2]
        end
        plot_1d(time, Chodura_ratio_upper[input.ir0,:], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_t.pdf")
            save(outfile, fig)
        end
    end

    if input.plot_vs_r
        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="r",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[3]
        end
        plot_1d(run_info.r.grid, Chodura_ratio_lower[:,input.it0], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="r",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[4]
        end
        plot_1d(run_info.r.grid, Chodura_ratio_upper[:,input.it0], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r.pdf")
            save(outfile, fig)
        end
    end

    if input.plot_vs_r_t
        if axes === nothing
            fig, ax, colorbar_place = get_2d_ax(title="Chodura ratio at z=-L/2",
                                                xlabel="r", ylabel="time")
            title = nothing
        else
            fig = nothing
            ax, colorbar_place = axes[5]
            title = run_info.run_name
        end
        plot_2d(run_info.r.grid, time, Chodura_ratio_lower, ax=ax,
                colorbar_place=colorbar_place, title=title)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r_t.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax, colorbar_place = get_2d_ax(title="Chodura ratio at z=+L/2",
                                                xlabel="r", ylabel="time")
            title = nothing
        else
            fig = nothing
            ax, colorbar_place = axes[6]
            title = run_info.run_name
        end
        plot_2d(run_info.r.grid, time, Chodura_ratio_upper, ax=ax,
                colorbar_place=colorbar_place, title=title)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r_t.pdf")
            save(outfile, fig)
        end
    end

    if input.plot_f_over_vpa2
        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 lower wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            title = nothing
            label = ""
        else
            fig = nothing
            ax = axes[7]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["it0"] = input.it0
        f_input["ir0"] = input.ir0
        f_input["iz0"] = 1
        plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1, fig=fig,
                             ax=ax, label=label)
        vlines!(ax, cutoff_lower[input.ir0,input.it0]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall-_vs_vpa.pdf"
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 upper wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            title = nothing
            label = ""
        else
            fig = nothing
            ax = axes[8]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["it0"] = input.it0
        f_input["ir0"] = input.ir0
        f_input["iz0"] = run_info.z.n
        plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1, fig=fig,
                             ax=ax, label=label)
        vlines!(ax, cutoff_upper[input.ir0,input.it0]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall+_vs_vpa.pdf"
            save(outfile, fig)
        end
    end

    if input.animate_f_over_vpa2
        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 lower wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            title = nothing
            label = ""
        else
            fig = nothing
            ax, frame_index = axes[9]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["ir0"] = input.ir0
        f_input["iz0"] = 1
        animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1, iz=1,
                                fig=fig, ax=ax, frame_index=frame_index, label=label)
        vlines!(ax, @lift cutoff_lower[input.ir0,$frame_index]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall-_vs_vpa." * input.animation_ext
            save_animation(fig, frame_index, run_info.nt, outfile)
        end

        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 upper wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            title = nothing
            label = ""
        else
            fig = nothing
            ax, frame_index = axes[10]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["ir0"] = input.ir0
        f_input["iz0"] = run_info.z.n
        animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1,
                                iz=run_info.z.n, fig=fig, ax=ax, frame_index=frame_index,
                                label=label)
        vlines!(ax, @lift cutoff_upper[input.ir0,$frame_index]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall+_vs_vpa." * input.animation_ext
            save_animation(fig, frame_index, run_info.nt, outfile)
        end
    end

    return nothing
end

"""
    sound_wave_plots(run_info::Tuple; plot_prefix)
    sound_wave_plots(run_info; outfile=nothing, ax=nothing, phi=nothing)

Calculate decay rate and frequency for the damped 'sound wave' in a 1D1V simulation in a
periodic box. Plot the mode amplitude vs. time along with the fitted decay rate.

The information for the runs to analyse and plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where line
plots from the different runs are overlayed on the same axis.

Settings are read from the `[sound_wave]` section of the input.

When `run_info` is a Tuple, `plot_prefix` is required and gives the path and prefix for
plots to be saved to. They will be saved with the format
`plot_prefix<some_identifying_string>.pdf`.
When `run_info` is not a Tuple, `outfile` can be passed, to save the plot to `outfile`.

When `run_info` is not a Tuple, ax can be passed to add the plot to an existing `Axis`.

When `run_info` is not a Tuple, the array containing data for phi can be passed to `phi` -
by default this data is loaded from the output file.
"""
function sound_wave_plots end

function sound_wave_plots(run_info::Tuple; plot_prefix)
    input = Dict_to_NamedTuple(input_dict["sound_wave_fit"])

    if !input.calculate_frequency && !input.plot
        return nothing
    end

    println("Doing analysis and making plots for sound wave test")
    flush(stdout)

    try
        outfile = plot_prefix * "delta_phi0_vs_t.pdf"

        if length(run_info) == 1
            return sound_wave_plots(run_info[1]; outfile=outfile)
        end

        if input.plot
            fig, ax = get_1d_ax(xlabel="time", ylabel="δϕ", yscale=log10)
        else
            ax = nothing
        end

        for ri ∈ run_info
            sound_wave_plots(ri; ax=ax)
        end

        if input.plot
            put_legend_right(fig, ax)

            save(outfile, fig)

            return fig
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in sound_wave_plots().")
    end

    return nothing
end

function sound_wave_plots(run_info; outfile=nothing, ax=nothing, phi=nothing)
    input = Dict_to_NamedTuple(input_dict["sound_wave_fit"])

    if !input.calculate_frequency && !input.plot
        return nothing
    end

    if ax === nothing && input.plot
        fig, ax = get_1d_ax(xlabel="time", ylabel="δϕ", yscale=log10)
    else
        fig = nothing
    end

    time = run_info.time

    # This analysis is only designed for 1D cases, so only use phi[:,ir0,:]
    if phi === nothing
        phi = get_variable(run_info, "phi"; ir=input.ir0)
    else
        select_slice(phi, :t, :z; input=input)
    end

    phi_fldline_avg, delta_phi = analyze_fields_data(phi, run_info.nt, run_info.z)

    if input.calculate_frequency
        frequency, growth_rate, shifted_time, fitted_delta_phi =
            calculate_and_write_frequencies(run_info.run_prefix, run_info.nt, time,
                                            run_info.z.grid, 1, run_info.nt, input.iz0,
                                            delta_phi, (calculate_frequencies=true,))
    end

    if input.plot
        if outfile === nothing
            # May be plotting multipe runs
            delta_phi_label = run_info.run_name * " δϕ"
            fit_label = run_info.run_name * " fit"
        else
            # Only plotting this run
            delta_phi_label = "δϕ"
            fit_label = "fit"
        end

        @views lines!(ax, time, positive_or_nan.(abs.(delta_phi[input.iz0,:]), epsilon=1.e-20), label=delta_phi_label)

        if input.calculate_frequency
            @views lines!(ax, time, positive_or_nan.(abs.(fitted_delta_phi), epsilon=1.e-20), label=fit_label)
        end

        if outfile !== nothing
            if fig === nothing
                error("Cannot save figure from this function when `ax` was passed. Please "
                      * "save the figure that contains `ax`")
            end
            put_legend_right(fig, ax)
            save(outfile, fig)
        end
    end

    return fig
end

"""
    instability2D_plots(run_info::Tuple, variable_name; plot_prefix, zind=nothing)
    instability2D_plots(run_info, variable_name; plot_prefix, zind=nothing,
                        axes_and_observables=nothing)

Make plots of `variable_name` for analysis of 2D instability.

The information for the runs to analyse and plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, make plots comparing the runs, shown in
a horizontal row..

Settings are read from the `[instability2D]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

When `run_info` is not a Tuple, `axes_and_observables` can be passed to add plots and
animations to existing figures, although this is not very convenient - see the use of this
argument when called from the `run_info::Tuple` method.

If `zind` is not passed, it is calculated as the z-index where the mode seems to have
the maximum growth rate for this variable.
Returns `zind`.
"""
function instability2D_plots end

function instability2D_plots(run_info::Tuple, variable_name; plot_prefix, zind=nothing)
    println("2D instability plots for $variable_name")
    flush(stdout)

    n_runs = length(run_info)
    var_symbol = get_variable_symbol(variable_name)
    instability2D_options = Dict_to_NamedTuple(input_dict["instability2D"])

    if zind === nothing
        zind = Tuple(nothing for _ in 1:n_runs)
    end

    if n_runs == 1
        # Don't need to set up for comparison plots, or include run_name in subplot titles
        zi = instability2D_plots(run_info[1], variable_name, plot_prefix=plot_prefix,
                                 zind=zind[1])
        return Union{mk_int,Nothing}[zi]
    end

    figs = []
    axes_and_observables = Tuple([] for _ ∈ 1:n_runs)
    if instability2D_options.plot_1d
        fig, ax = get_1d_ax(n_runs; title="$var_symbol 1D Fourier components", yscale=log10)
        push!(figs, fig)
        for (i, a) ∈ enumerate(ax)
            push!(axes_and_observables[i], a)
        end
        fig, ax = get_1d_ax(n_runs; title="phase of n_r=1 mode for $var_symbol")
        push!(figs, fig)
        for (i, a) ∈ enumerate(ax)
            push!(axes_and_observables[i], a)
        end
    else
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
    end
    if instability2D_options.plot_2d
        fig, ax = get_1d_ax(n_runs; title="$var_symbol Fourier components", yscale=log10)
        push!(figs, fig)
        for (i, a) ∈ enumerate(ax)
            push!(axes_and_observables[i], a)
        end
        frame_index = Observable(1)
        fig, ax, colorbar_places = get_2d_ax(n_runs; title="$var_symbol Fourier components")
        push!(figs, fig)
        for (i, (a, cb)) ∈ enumerate(zip(ax, colorbar_places))
            push!(axes_and_observables[i], (a, cb, frame_index))
        end

        # Delete any existing mode stats file so we can append to an empty file
        mode_stats_file_name = string(plot_prefix, "mode_$variable_name.txt")
        if isfile(mode_stats_file_name)
            rm(mode_stats_file_name)
        end
    else
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
    end
    if instability2D_options.animate_perturbations
        frame_index = Observable(1)
        fig, ax, colorbar_places = get_2d_ax(n_runs; title="$var_symbol perturbation")
        push!(figs, fig)
        for (i, (a, cb)) ∈ enumerate(zip(ax, colorbar_places))
            push!(axes_and_observables[i], (a, cb, frame_index))
        end
    else
        push!(figs, nothing)
        for i ∈ 1:n_runs
            push!(axes_and_observables[i], nothing)
        end
    end

    for (i, (ri, ax_ob, zi)) ∈ enumerate(zip(run_info, axes_and_observables, zind))
        zi = instability2D_plots(ri, variable_name, plot_prefix=plot_prefix, zind=zi,
                                 axes_and_observables=ax_ob)
        zind[i] = zi
    end

    fig = figs[1]
    if fig !== nothing
        outfile = string(plot_prefix, "$(variable_name)_1D_Fourier_components.pdf")
        save(outfile, fig)
    end

    fig = figs[2]
    if fig !== nothing
        outfile = string(plot_prefix, "$(variable_name)_1D_phase.pdf")
        save(outfile, fig)
    end

    fig = figs[3]
    if fig !== nothing
        outfile = string(plot_prefix, "$(variable_name)_Fourier_components.pdf")
        save(outfile, fig)
    end

    fig = figs[4]
    if fig !== nothing
        frame_index = axes_and_observables[1][4][3]
        nt = minimum(ri.nt for ri ∈ run_info)
        outfile = plot_prefix * variable_name * "_Fourier." *
                  instability2D_options.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    fig = figs[5]
    if fig !== nothing
        frame_index = axes_and_observables[1][5][3]
        nt = minimum(ri.nt for ri ∈ run_info)
        outfile = plot_prefix * variable_name * "_perturbation." *
                  instability2D_options.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    return zind
end

function instability2D_plots(run_info, variable_name; plot_prefix, zind=nothing,
                             axes_and_observables=nothing)
    instability2D_options = Dict_to_NamedTuple(input_dict["instability2D"])

    time = run_info.time

    if variable_name == "temperature"
        variable = get_variable(run_info, "thermal_speed").^2
    else
        variable = get_variable(run_info, variable_name)
    end

    if ndims(variable) == 4
        # Only support single species runs in this routine, so pick is=1
        variable = @view variable[:,:,1,:]
    elseif ndims(variable) > 4
        error("Variables with velocity space dimensions not supported in "
              * "instability2D_plots.")
    end

    if instability2D_options.plot_1d
        function unravel_phase!(phase::AbstractVector)
            # Remove jumps in phase where it crosses from -π to π
            for i ∈ 2:length(phase)
                if phase[i] - phase[i-1] > π
                    @views phase[i:end] .-= 2.0*π
                elseif phase[i] - phase[i-1] < -π
                    @views phase[i:end] .+= 2.0*π
                end
            end
        end
        function get_phase_velocity(phase, time, amplitude)
            # Assume that once the amplitude reaches 2x initial amplitude that the mode is
            # well established, so will be able to measure phase velocity
            startind = findfirst(x -> x>amplitude[1], amplitude)
            if startind === nothing
                startind = 1
            end

            # Linear fit to phase after startind
            linear_model(x, param) = @. param[1]*x+param[2]
            fit = @views curve_fit(linear_model, time[startind:end], phase[startind:end],
                                   [0.0, 0.0])
            phase_velocity = fit.param[1]
            phase_offset = fit.param[2]

            return phase_velocity, phase_offset, startind
        end
        function get_growth_rate(amplitude, time)
            # Assume that once the amplitude reaches 2x initial amplitude that the mode is
            # well established, so will be able to measure phase velocity
            startind = findfirst(x -> x>2*amplitude[1], amplitude)
            if startind === nothing
                startind = 1
            end

            # Linear fit to log(amplitude) after startind
            growth_rate = 0.0
            initial_fit_amplitude = 1.0
            try
                linear_model(x, param) = @. param[1]*x+param[2]
                fit = @views curve_fit(linear_model, time[startind:end],
                                       log.(amplitude[startind:end]), [0.0, 0.0])
                growth_rate = fit.param[1]
                initial_fit_amplitude = exp(fit.param[2])
            catch e
                println("Warning: error $e when fitting growth rate")
            end

            return growth_rate, initial_fit_amplitude, startind
        end

        function plot_Fourier_1D(var, symbol, name)
            # File to save growth rate and frequency to
            if axes_and_observables === nothing
                mode_stats_file = open(string(plot_prefix, "mode_$name.txt"), "w")
            else
                # Processing multiple runs, so any existing mode_stats_file should have
                # already been deleted so that we can append in this function.
                mode_stats_file = open(string(plot_prefix, "mode_$name.txt"), "a")
                println(mode_stats_file, run_info.run_name)
                println(mode_stats_file, "-" ^ length(run_info.run_name))
            end

            amplitude = abs.(var)

            @views growth_rate, initial_fit_amplitude, startind =
                get_growth_rate(amplitude[2,:], time)

            # ikr=2 is the n_r=1 mode, so...
            kr_2 = 2.0*π/run_info.r.L
            println("for $symbol, kr=$kr_2, growth rate is $growth_rate")
            println(mode_stats_file, "kr = $kr_2")
            println(mode_stats_file, "growth_rate = $growth_rate")

            if axes_and_observables === nothing
                fig, ax = get_1d_ax(title="$symbol 1D Fourier components", xlabel="time",
                                    ylabel="amplitude", yscale=log10)
            else
                fig = nothing
                ax = axes_and_observables[1]
                ax.title = run_info.run_name
            end

            n_kr, nt = size(amplitude)

            # Drop constant mode (ikr=1) and aliased (?) modes >n_kr/2
            for ikr ∈ 2:n_kr÷2
                data = amplitude[ikr,:]
                data[data.==0.0] .= NaN
                plot_1d(time, data, ax=ax)
                text!(ax, position=(time[end], data[end]), "ikr=$ikr", fontsize=6,
                      justification=:right)
            end

            plot_1d(time, initial_fit_amplitude.*exp.(growth_rate.*time), ax=ax)
            vlines!(ax, [time[startind]], linestyle=:dot)

            if axes_and_observables === nothing
                outfile = string(plot_prefix, "$(name)_1D_Fourier_components.pdf")
                save(outfile, fig)
            end

            # Plot phase of n_r=1 mode
            phase = angle.(var[2,:])
            unravel_phase!(phase)

            phase_velocity, phase_offset, startind =
                get_phase_velocity(phase, time, @view amplitude[2,:])

            # ikr=2 is the n_r=1 mode, so...
            omega_2 = phase_velocity*kr_2

            println("for $symbol, kr=$kr_2, phase velocity is $phase_velocity, omega=$omega_2")
            println(mode_stats_file, "omega = $omega_2")

            if axes_and_observables === nothing
                fig, ax = get_1d_ax(title="phase of n_r=1 mode", xlabel="time",
                                    ylabel="phase")
            else
                fig = nothing
                ax = axes_and_observables[2]
                ax.title = run_info.run_name
            end

            plot_1d(time, phase, ax=ax, label="phase")
            plot_1d(time, phase_offset.+phase_velocity.*time, ax=ax, label="fit")
            vlines!(ax, [time[startind]], linestyle=:dot)
            axislegend(ax)

            if axes_and_observables === nothing
                outfile = string(plot_prefix, "$(name)_1D_phase.pdf")
                save(outfile, fig)
            end

            if axes_and_observables === nothing
                println(mode_stats_file, "")
            end
            close(mode_stats_file)
        end
        try
            variable_Fourier_1D, zind = get_Fourier_modes_1D(variable, run_info.r,
                                                             run_info.r_spectral, run_info.z,
                                                             zind=zind)
            plot_Fourier_1D(variable_Fourier_1D, get_variable_symbol(variable_name),
                            variable_name)
        catch e
            return makie_post_processing_error_handler(
                       e,
                       "Warning: error in 1D Fourier analysis for $variable_name.")
        end

        # Do this to allow memory to be garbage-collected.
        variable_Fourier_1D = nothing
    end

    if instability2D_options.plot_2d
        function plot_Fourier_2D(var, symbol, name)
            if axes_and_observables === nothing
                fig, ax = get_1d_ax(title="$symbol Fourier components", xlabel="time",
                                    ylabel="amplitude", yscale=log10)
            else
                fig = nothing
                ax = axes_and_observables[3]
                ax.title = run_info.run_name
            end

            n_kz, n_kr, nt = size(var)
            for ikr ∈ 1:n_kr, ikz ∈ 1:n_kz
                ikr!=2 && continue
                data = abs.(var[ikz,ikr,:])
                data[data.==0.0] .= NaN
                plot_1d(time, data, ax=ax)
                text!(ax, position=(time[end], data[end]), "ikr=$ikr, ikz=$ikz", fontsize=6,
                      justification=:right)
            end

            if axes_and_observables === nothing
                outfile = string(plot_prefix, "$(name)_Fourier_components.pdf")
                save(outfile, fig)
            end

            # make a gif animation of Fourier components
            if axes_and_observables === nothing
                ax = nothing
                colorbar_place = nothing
                frame_index = nothing
                outfile = plot_prefix * name * "_Fourier." * instability2D_options.animation_ext
                title = "$symbol Fourier components"
            else
                ax, colorbar_place, frame_index = axes_and_observables[4]
                outfile = nothing
                title = run_info.run_name
            end
            kr = collect(0:n_kr-1) * 2 * π / run_info.r.L
            kz = collect(0:n_kz-1) * 2 * π / run_info.z.L
            animate_2d(kz, kr, abs.(var), xlabel="kz", ylabel="kr",
                       title=title,
                       colormap=instability2D_options.colormap, colorscale=log10, ax=ax,
                       colorbar_place=colorbar_place, frame_index=frame_index,
                       outfile=outfile)
        end
        variable_Fourier = get_Fourier_modes_2D(variable, run_info.r, run_info.r_spectral,
                                                run_info.z, run_info.z_spectral)
        try
            plot_Fourier_2D(variable_Fourier, get_variable_symbol(variable_name),
                            variable_name)
        catch e
            return makie_post_processing_error_handler(
                       e,
                       "Warning: error in 2D Fourier analysis for $variable_name.")
        end

        # Do this to allow memory to be garbage-collected.
        variable_Fourier = nothing
    end

    if instability2D_options.animate_perturbations
        try
            perturbation = get_r_perturbation(variable)
            # make animation of perturbation
            if axes_and_observables === nothing
                ax = nothing
                colorbar_place = nothing
                frame_index = nothing
                outfile = plot_prefix*variable_name*"_perturbation." * instability2D_options.animation_ext
                title = "$(get_variable_symbol(variable_name)) perturbation"
            else
                ax, colorbar_place, frame_index = axes_and_observables[5]
                outfile = nothing
                title = run_info.run_name
            end
            animate_2d(run_info.z.grid, run_info.r.grid, perturbation, xlabel="z", ylabel="r",
                       title=title,
                       colormap=instability2D_options.colormap, ax=ax,
                       colorbar_place=colorbar_place, frame_index=frame_index,
                       outfile=outfile)
        catch e
            return makie_post_processing_error_handler(
                       e,
                       "Warning: error in perturbation animation for $variable_name.")
        end

        # Do this to allow memory to be garbage-collected (although this is redundant
        # here as this is the last thing in the function).
        perturbation = nothing
    end

    return zind
end

# Manufactured solutions analysis
#################################

"""
     manufactured_solutions_get_field_and_field_sym(run_info, variable_name;
         it=nothing, ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
         ivr=nothing, ivz=nothing)

Get the data `variable` for `variable_name` from the output, and calculate the
manufactured solution `variable_sym`.

The information for the runs to analyse and plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`it`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, `ivz` can be used to select a subset
of the grid by passing an integer or range for any dimension.

Returns `variable`, `variable_sym`.
"""
function manufactured_solutions_get_field_and_field_sym(run_info, variable_name;
        it=nothing, ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
        ivr=nothing, ivz=nothing, nvperp)

    variable_name = Symbol(variable_name)

    func_name_lookup = (phi=:phi_func, Er=:Er_func, Ez=:Ez_func, density=:densi_func,
                        parallel_flow=:upari_func, parallel_pressure=:ppari_func,
                        density_neutral=:densn_func, f=:dfni_func, f_neutral=:dfnn_func)

    nt = run_info.nt
    nr = run_info.r.n
    nz = run_info.z.n
    if it === nothing
        it = 1:nt
    end
    if ir === nothing
        ir = 1:nr
    end
    if iz === nothing
        iz = 1:nz
    end
    tinds = run_info.itime_min:run_info.itime_skip:run_info.itime_max
    tinds = tinds[it]

    if nr > 1
        Lr_in = run_info.r.L
    else
        Lr_in = 1.0
    end

    if variable_name ∈ (:phi, :Er, :Ez)
        manufactured_funcs =
            manufactured_electric_fields(Lr_in, run_info.z.L, run_info.r.bc,
                                         run_info.z.bc, run_info.composition,
                                         run_info.r.n, run_info.manufactured_solns_input,
                                         run_info.species)
    elseif variable_name ∈ (:density, :parallel_flow, :parallel_pressure,
                            :density_neutral, :f, :f_neutral)
        manufactured_funcs =
            manufactured_solutions(run_info.manufactured_solns_input, Lr_in, run_info.z.L,
                                   run_info.r.bc, run_info.z.bc, run_info.geometry.input,
                                   run_info.composition, run_info.species, run_info.r.n,
                                   nvperp)
    end

    variable_func = manufactured_funcs[func_name_lookup[variable_name]]

    variable = get_variable(run_info, String(variable_name); it=tinds, is=1, ir=ir, iz=iz,
                            ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr, ivz=ivz)
    variable_sym = similar(variable)

    time = run_info.time
    r_grid = run_info.r.grid
    z_grid = run_info.z.grid

    if variable_name == :f
        vperp_grid = run_info.vperp.grid
        vpa_grid = run_info.vpa.grid
        nvperp = run_info.vperp.n
        nvpa = run_info.vpa.n
        if ivperp === nothing
            ivperp = 1:nvperp
        end
        if ivpa === nothing
            ivpa = 1:nvpa
        end
        counter = 1
        for iit ∈ it, iir ∈ ir, iiz ∈ iz, iivperp ∈ ivperp, iivpa ∈ ivpa
            variable_sym[counter] =
                variable_func(vpa_grid[iivpa], vperp_grid[iivperp], z_grid[iiz],
                              r_grid[iir], time[iit])
            counter += 1
        end
    elseif variable_name == :f_neutral
        vzeta_grid = run_info.vzeta.grid
        vr_grid = run_info.vr.grid
        vz_grid = run_info.vz.grid
        nvzeta = run_info.vzeta.n
        nvr = run_info.vr.n
        nvz = run_info.vz.n
        if ivzeta === nothing
            ivzeta = 1:nvzeta
        end
        if ivr === nothing
            ivr = 1:nvr
        end
        if ivz === nothing
            ivz = 1:nvz
        end
        counter = 1
        for iit ∈ it, iir ∈ ir, iiz ∈ iz, iivzeta ∈ ivzeta, iivr ∈ ivr, iivz ∈ ivz
            variable_sym[counter] =
            variable_func(vz_grid[iivz], vr_grid[iivr], vzeta_grid[iivzeta], z_grid[iiz],
                          r_grid[iir], time[iit])
            counter += 1
        end
    else
        counter = 1
        for iit ∈ it, iir ∈ ir, iiz ∈ iz
            variable_sym[counter] = variable_func(z_grid[iiz], r_grid[iir], time[iit])
            counter += 1
        end
    end

    return variable, variable_sym
end

"""
    compare_moment_symbolic_test(run_info, plot_prefix, field_label, field_sym_label,
                                 norm_label, variable_name; io=nothing)

Compare the computed and manufactured solutions for a field or moment variable
`variable_name`.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`field_label` is the label that will be used for the name of the computed variable in
plots, `field_sym_label` is the label for the manufactured solution, and `norm_label` is
the label for the error (the difference between the computed and manufactured solutions).

If `io` is passed then error norms will be written to that file.
"""
function compare_moment_symbolic_test(run_info, plot_prefix, field_label, field_sym_label,
                                      norm_label, variable_name; io=nothing,
                                      input=nothing, nvperp)

    println("Doing MMS analysis and making plots for $variable_name")
    flush(stdout)

    if input === nothing
        input = Dict_to_NamedTuple(input_dict["manufactured_solns"])
    end

    field, field_sym =
        manufactured_solutions_get_field_and_field_sym(run_info, variable_name; nvperp=nvperp)
    error = field .- field_sym

    nt = run_info.nt
    time = run_info.time
    r = run_info.r
    z = run_info.z

    if !input.calculate_error_norms
        field_norm = nothing
    else
        field_norm = zeros(mk_float,nt)
        for it in 1:nt
            dummy = 0.0
            #dummy_N = 0.0
            for ir in 1:r.n
                for iz in 1:z.n
                    dummy += (field[iz,ir,it] - field_sym[iz,ir,it])^2
                    #dummy_N +=  (field_sym[iz,ir,it])^2
                end
            end
            #field_norm[it] = dummy/dummy_N
            field_norm[it] = sqrt(dummy/(r.n*z.n))
        end
        println_to_stdout_and_file(io, join(field_norm, " "), " # ", variable_name)
        plot_vs_t(run_info, norm_label, input=input, data=field_norm,
                  outfile=plot_prefix*variable_name*"_norm_vs_t.pdf")
    end

    has_rdim = (r.n > 1)
    has_zdim = (z.n > 1)

    if has_rdim && input.wall_plots
        # plot last (by default) timestep field vs r at z_wall

        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(r.grid, select_slice(field, :r; input=input, iz=1), xlabel=L"r",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(r.grid, select_slice(field_sym, :r; input=input, iz=1),
                label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(r.grid, select_slice(error, :r; input=input, iz=1), xlabel=L"r",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "(z_wall-)_vs_r.pdf"
        save(outfile, fig)

        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(r.grid, select_slice(field, :r; input=input, iz=z.n), xlabel=L"r",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(r.grid, select_slice(field_sym, :r; input=input, iz=z.n),
                label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(r.grid, select_slice(error, :r; input=input, iz=z.n), xlabel=L"r",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "(z_wall+)_vs_r.pdf"
        save(outfile, fig)
    end

    if input.plot_vs_t
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(time, select_slice(field, :t; input=input), xlabel=L"t",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(time, select_slice(field_sym, :t; input=input), label=field_sym_label,
                ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(time, select_slice(error, :t; input=input), xlabel=L"t",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_t.pdf"
        save(outfile, fig)
    end
    if has_rdim && input.plot_vs_r
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(r.grid, select_slice(field, :r; input=input), xlabel=L"r",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(r.grid, select_slice(field_sym, :r; input=input), label=field_sym_label,
                ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(r.grid, select_slice(error, :r; input=input), xlabel=L"r",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_r.pdf"
        save(outfile, fig)
    end
    if has_zdim && input.plot_vs_z
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(z.grid, select_slice(field, :z; input=input), xlabel=L"z",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(z.grid, select_slice(field_sym, :z; input=input), label=field_sym_label,
                ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(z.grid, select_slice(error, :z; input=input), xlabel=L"z",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z.pdf"
        save(outfile, fig)
    end
    if has_rdim && input.plot_vs_r_t
        fig, ax, colorbar_place = get_2d_ax(3)
        plot_2d(r.grid, time, select_slice(field, :t, :r; input=input), title=field_label,
                xlabel=L"r", ylabel=L"t", ax=ax[1], colorbar_place=colorbar_place[1])
        plot_2d(r.grid, time, select_slice(field_sym, :t, :r; input=input),
                title=field_sym_label, xlabel=L"r", ylabel=L"t", ax=ax[2],
                colorbar_place=colorbar_place[2])
        plot_2d(r.grid, time, select_slice(error, :t, :r; input=input), title=norm_label,
                xlabel=L"r", ylabel=L"t", ax=ax[3], colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_r_t.pdf"
        save(outfile, fig)
    end
    if has_zdim && input.plot_vs_z_t
        fig, ax, colorbar_place = get_2d_ax(3)
        plot_2d(z.grid, time, select_slice(field, :t, :z; input=input), title=field_label,
                xlabel=L"z", ylabel=L"t", ax=ax[1], colorbar_place=colorbar_place[1])
        plot_2d(z.grid, time, select_slice(field_sym, :t, :z; input=input),
                title=field_sym_label, xlabel=L"z", ylabel=L"t", ax=ax[2],
                colorbar_place=colorbar_place[2])
        plot_2d(z.grid, time, select_slice(error, :t, :z; input=input), title=norm_label,
                xlabel=L"z", ylabel=L"t", ax=ax[3], colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z_t.pdf"
        save(outfile, fig)
    end
    if has_rdim && has_zdim && input.plot_vs_z_r
        fig, ax, colorbar_place = get_2d_ax(3)
        plot_2d(z.grid, r.grid, select_slice(field, :r, :z; input=input),
                title=field_label, xlabel=L"z", ylabel=L"r", ax=ax[1],
                colorbar_place=colorbar_place[1])
        plot_2d(z.grid, r.grid, select_slice(field_sym, :r, :z; input=input),
                title=field_sym_label, xlabel=L"z", ylabel=L"r", ax=ax[2],
                colorbar_place=colorbar_place[2])
        plot_2d(z.grid, r.grid, select_slice(error, :r, :z; input=input),
                title=norm_label, xlabel=L"z", ylabel=L"r", ax=ax[3],
                colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z_r.pdf"
        save(outfile, fig)
    end
    if has_rdim && input.animate_vs_r
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        frame_index = Observable(1)
        animate_1d(r.grid, select_slice(field, :t, :r; input=input),
                   frame_index=frame_index, xlabel="r", ylabel=field_label,
                   label=field_label, ax=ax[1])
        animate_1d(r.grid, select_slice(field_sym, :t, :r; input=input),
                   frame_index=frame_index, label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        animate_1d(r.grid, select_slice(error, :t, :r; input=input),
                   frame_index=frame_index, xlabel="r", ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_r." * input.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end
    if has_zdim && input.animate_vs_z
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        frame_index = Observable(1)
        animate_1d(z.grid, select_slice(field, :t, :z; input=input),
                   frame_index=frame_index, xlabel="z", ylabel=field_label,
                   label=field_label, ax=ax[1])
        animate_1d(z.grid, select_slice(field_sym, :t, :z; input=input),
                   frame_index=frame_index, label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        animate_1d(z.grid, select_slice(error, :t, :z; input=input),
                   frame_index=frame_index, xlabel="z", ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z." * input.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end
    if has_rdim && has_zdim && input.animate_vs_z_r
        fig, ax, colorbar_place = get_2d_ax(3)
        frame_index = Observable(1)
        animate_2d(z.grid, r.grid, select_slice(field, :t, :r, :z; input=input),
                   frame_index=frame_index, title=field_label, xlabel=L"z", ylabel=L"y",
                   ax=ax[1], colorbar_place=colorbar_place[1])
        animate_2d(z.grid, r.grid, select_slice(field_sym, :t, :r, :z; input=input),
                   frame_index=frame_index, title=field_sym_label, xlabel=L"z",
                   ylabel=L"y", ax=ax[2], colorbar_place=colorbar_place[2])
        animate_2d(z.grid, r.grid, select_slice(error, :t, :r, :z; input=input),
                   frame_index=frame_index, title=norm_label, xlabel=L"z", ylabel=L"y",
                   ax=ax[3], colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z_r." * input.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    return field_norm
end

"""
    _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                   field_sym_label, norm_label, plot_dims, animate_dims)

Utility function for making plots to avoid duplicated code in
[`compare_ion_pdf_symbolic_test`](@ref) and
[`compare_neutral_pdf_symbolic_test`](@ref).

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`input` is a NamedTuple of settings to use.

`variable_name` is the name of the variable being plotted.

`plot_prefix` gives the path and prefix for plots to be saved to. They will be saved with
the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`field_label` is the label for the computed variable that will be used in
plots/animations, `field_sym_label` is the label for the manufactured solution, and
`norm_label` is the label for the error.

`plot_dims` are the dimensions of the variable, and `animate_dims` are the same but
omitting `:t`.
"""
function _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                        field_sym_label, norm_label, plot_dims, animate_dims, neutrals)

    nt = run_info.nt
    time = run_info.time

    if neutrals
        all_dims_no_t = (:r, :z, :vzeta, :vr, :vz)
    else
        all_dims_no_t = (:r, :z, :vperp, :vpa)
    end
    all_dims = tuple(:t, all_dims_no_t...)
    all_plot_slices = Tuple(Symbol(:i, d)=>input[Symbol(:i, d, :0)] for d ∈ all_dims)
    all_animate_slices = Tuple(Symbol(:i, d)=>input[Symbol(:i, d, :0)] for d ∈ all_dims_no_t)

    # Options to produce either regular or log-scale plots
    epsilon = 1.0e-30 # minimum data value to include in log plots
    for (log, yscale, transform, error_transform) ∈
            (("", nothing, identity, identity),
             (:_log, log10, x->positive_or_nan(x; epsilon=1.e-30), x->positive_or_nan.(abs.(x); epsilon=1.e-30)))
        for dim ∈ plot_dims
            if input[Symbol(:plot, log, :_vs_, dim)]
                coord = dim === :t ? time : run_info[dim].grid

                slices = (k=>v for (k, v) ∈ all_plot_slices if k != Symbol(:i, dim))
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, legend_place = get_1d_ax(2; yscale=yscale, get_legend_place=:below)
                plot_1d(coord, f, xlabel=L"%$dim", ylabel=field_label, label=field_label,
                        ax=ax[1], transform=transform)
                plot_1d(coord, f_sym, label=field_sym_label, ax=ax[1],
                        transform=transform)
                Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                       orientation=:horizontal)
                plot_1d(coord, error, xlabel=L"%$dim", ylabel=norm_label, ax=ax[2],
                        transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$dim.pdf"
                save(outfile, fig)
            end
        end
        for (dim1, dim2) ∈ combinations(plot_dims, 2)
            if input[Symbol(:plot, log, :_vs_, dim2, :_, dim1)]
                coord1 = dim1 === :t ? time : run_info[dim1].grid
                coord2 = dim2 === :t ? time : run_info[dim2].grid

                slices = (k=>v for (k, v) ∈ all_plot_slices
                          if k ∉ (Symbol(:i, dim1), Symbol(:i, dim2)))
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(coord2, coord1, f, title=field_label, xlabel=L"%$dim2",
                        ylabel=L"%$dim1", ax=ax[1], colorbar_place=colorbar_place[1],
                        colorscale=yscale, transform=transform)
                plot_2d(coord2, coord1, f_sym, title=field_sym_label, xlabel=L"%$dim2",
                        ylabel=L"%$dim1", ax=ax[2], colorbar_place=colorbar_place[2],
                        colorscale=yscale, transform=transform)
                plot_2d(coord2, coord1, error, title=norm_label, xlabel=L"%$dim2",
                        ylabel=L"%$dim1", ax=ax[3], colorbar_place=colorbar_place[3],
                        colorscale=yscale, transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$(dim2)_$(dim1).pdf"
                save(outfile, fig)
            end
        end
        for dim ∈ animate_dims
            if input[Symbol(:animate, log, :_vs_, dim)]
                coord = dim === :t ? time : run_info[dim].grid

                slices = (k=>v for (k, v) ∈ all_animate_slices if k != Symbol(:i, dim))
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, legend_place = get_1d_ax(2; yscale=yscale, get_legend_place=:below)
                frame_index = Observable(1)
                animate_1d(coord, f, frame_index=frame_index, xlabel=L"%$dim",
                           ylabel=field_label, label=field_label, ax=ax[1],
                           transform=transform)
                animate_1d(coord, f_sym, frame_index=frame_index, label=field_sym_label,
                           ax=ax[1], transform=transform)
                Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                       orientation=:horizontal)
                animate_1d(coord, error, frame_index=frame_index, xlabel=L"%$dim",
                           ylabel=norm_label, label=field_label, ax=ax[2],
                           transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$dim." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)
            end
        end
        for (dim1, dim2) ∈ combinations(animate_dims, 2)
            if input[Symbol(:animate, log, :_vs_, dim2, :_, dim1)]
                coord1 = dim1 === :t ? time : run_info[dim1].grid
                coord2 = dim2 === :t ? time : run_info[dim2].grid

                slices = (k=>v for (k, v) ∈ all_animate_slices
                          if k ∉ (Symbol(:i, dim1), Symbol(:i, dim2)))
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                frame_index = Observable(1)
                animate_2d(coord2, coord1, f, frame_index=frame_index, xlabel=L"%$dim2",
                           ylabel=L"%$dim1", title=field_label, ax=ax[1],
                           colorbar_place=colorbar_place[1], colorscale=yscale,
                           transform=transform)
                animate_2d(coord2, coord1, f_sym, frame_index=frame_index,
                           xlabel=L"%$dim2", ylabel=L"%$dim1", title=field_sym_label,
                           ax=ax[2], colorbar_place=colorbar_place[2], colorscale=yscale,
                           transform=transform)
                animate_2d(coord2, coord1, error, frame_index=frame_index,
                           xlabel=L"%$dim2", ylabel=L"%$dim1", title=norm_label,
                           ax=ax[3], colorbar_place=colorbar_place[3], colorscale=yscale,
                           transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$(dim2)_$(dim1)." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)
            end
        end
    end
end

"""
    compare_ion_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                      input=nothing)

Compare the computed and manufactured solutions for the ion distribution function.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

If `io` is passed then error norms will be written to that file.

`input` is a NamedTuple of settings to use. If not given it will be read from the
`[manufactured_solns]` section of [`input_dict_dfns`][@ref].

Note: when calculating error norms, data is loaded only for 1 time point and for an r-z
chunk that is the same size as computed by 1 block of the simulation at run time. This
should prevent excessive memory requirements for this function.
"""
function compare_ion_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                           input=nothing)

    field_label = L"\tilde{f}_i"
    field_sym_label = L"\tilde{f}_i^{sym}"
    norm_label = L"\varepsilon(\tilde{f}_i)"
    variable_name = "f"

    println("Doing MMS analysis and making plots for $variable_name")
    flush(stdout)

    if input === nothing
        input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])
    end

    nt = run_info.nt
    r = run_info.r
    z = run_info.z
    vperp = run_info.vperp
    vpa = run_info.vpa

    if !input.calculate_error_norms
        field_norm = nothing
    else
        # Load data in chunks, with the same size as the chunks that were saved during the
        # run, to avoid running out of memory
        r_chunks = UnitRange{mk_int}[]
        chunk = run_info.r_chunk_size
        nchunks = (r.n ÷ chunk)
        if nchunks == 1
            r_chunks = [1:r.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(r_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(r_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        z_chunks = UnitRange{mk_int}[]
        chunk = run_info.z_chunk_size
        nchunks = (z.n ÷ chunk)
        if nchunks == 1
            z_chunks = [1:z.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(z_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(z_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        field_norm = zeros(mk_float,nt)
        for it in 1:nt
            dummy = 0.0
            #dummy_N = 0.0
            for r_chunk ∈ r_chunks, z_chunk ∈ z_chunks
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, it=it,
                        ir=r_chunk, iz=z_chunk)
                dummy += sum(@. (f - f_sym)^2)
                #dummy_N += sum(f_sym.^2)
            end

            #field_norm[it] = dummy/dummy_N
            field_norm[it] = sqrt(dummy/(r.n*z.n*vperp.n*vpa.n))
        end
        println_to_stdout_and_file(io, join(field_norm, " "), " # ", variable_name)
        plot_vs_t(run_info, norm_label, input=input, data=field_norm,
                  outfile=plot_prefix*"f_norm_vs_t.pdf")
    end

    has_rdim = (r.n > 1)
    has_zdim = (z.n > 1)
    is_1V = (vperp.n == 1)

    if input.wall_plots
        for (iz, z_label) ∈ ((1, "wall-"), (z.n, "wall+"))
            f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0,
                    ir=input.ir0, iz=iz, ivperp=input.ivperp0)
            error = f .- f_sym

            fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
            plot_1d(vpa.grid, f, ax=ax[1], label="num",
                    xlabel=L"v_{\parallel}/L_{v_{\parallel}}", ylabel=field_label)
            plot_1d(vpa.grid, f_sym, ax=ax[1], label="sym")
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                   orientation=:horizontal)

            plot_1d(vpa.grid, error, ax=ax[2], xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                    ylabel=norm_label)

            outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vpa.pdf"
            save(outfile, fig)

            if has_rdim
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ivperp=input.ivperp0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vpa.grid, r.grid, f, ax=ax[1], colorbar_place=colorbar_place[1],
                        title=field_label, xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"r")
                plot_2d(vpa.grid, r.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}", ylabel=L"r")
                plot_2d(vpa.grid, r.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}", ylabel=L"r")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vpa_r.pdf"
                save(outfile, fig)
            end

            if !is_1V
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ir=input.ir0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vpa.grid, vperp.grid, f, ax=ax[1],
                        colorbar_place=colorbar_place[1], title=field_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"v_{\perp}/L_{v_{\perp}}")
                plot_2d(vpa.grid, vperp.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"v_{\perp}/L_{v_{\perp}}")
                plot_2d(vpa.grid, vperp.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"v_{\perp}/L_{v_{\perp}}")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vpa_vperp.pdf"
                save(outfile, fig)
            end
        end
    end

    animate_dims = setdiff(ion_dimensions, (:s,))
    if !has_rdim
        animate_dims = setdiff(animate_dims, (:r,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    if is_1V
        animate_dims = setdiff(animate_dims, (:vperp,))
    end
    plot_dims = tuple(:t, animate_dims...)
    _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                   field_sym_label, norm_label, plot_dims, animate_dims, false)

    return field_norm
end

"""
    compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                      input=nothing)

Compare the computed and manufactured solutions for the neutral distribution function.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

If `io` is passed then error norms will be written to that file.

`input` is a NamedTuple of settings to use. If not given it will be read from the
`[manufactured_solns]` section of [`input_dict_dfns`][@ref].

Note: when calculating error norms, data is loaded only for 1 time point and for an r-z
chunk that is the same size as computed by 1 block of the simulation at run time. This
should prevent excessive memory requirements for this function.
"""
function compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                           input=nothing)

    field_label = L"\tilde{f}_n"
    field_sym_label = L"\tilde{f}_n^{sym}"
    norm_label = L"\varepsilon(\tilde{f}_n)"
    variable_name = "f_neutral"

    println("Doing MMS analysis and making plots for $variable_name")
    flush(stdout)

    if input === nothing
        input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])
    end

    nt = run_info.nt
    r = run_info.r
    z = run_info.z
    vzeta = run_info.vzeta
    vr = run_info.vr
    vz = run_info.vz

    # Load data in chunks, with the same size as the chunks that were saved during the
    # run, to avoid running out of memory
    if !input.calculate_error_norms
        field_norm = nothing
    else
        r_chunks = UnitRange{mk_int}[]
        chunk = run_info.r_chunk_size
        nchunks = (r.n ÷ chunk)
        if nchunks == 1
            r_chunks = [1:r.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(r_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(r_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        z_chunks = UnitRange{mk_int}[]
        chunk = run_info.z_chunk_size
        nchunks = (z.n ÷ chunk)
        if nchunks == 1
            z_chunks = [1:z.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(z_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(z_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        field_norm = zeros(mk_float,nt)
        for it in 1:nt
            dummy = 0.0
            #dummy_N = 0.0
            for r_chunk ∈ r_chunks, z_chunk ∈ z_chunks
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, it=it,
                        ir=r_chunk, iz=z_chunk)
                dummy += sum(@. (f - f_sym)^2)
                #dummy_N += sum(f_sym.^2)
            end

            #field_norm[it] = dummy/dummy_N
            field_norm[it] = sqrt(dummy/(r.n*z.n*vzeta.n*vr.n*vz.n))
        end
        println_to_stdout_and_file(io, join(field_norm, " "), " # ", variable_name)
        plot_vs_t(run_info, norm_label, input=input, data=field_norm,
                  outfile=plot_prefix*variable_name*"_norm_vs_t.pdf")
    end

    has_rdim = (r.n > 1)
    has_zdim = (z.n > 1)
    is_1V = (vzeta.n == 1 && vr.n == 1)

    if input.wall_plots
        for (iz, z_label) ∈ ((1, "wall-"), (z.n, "wall+"))
            f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0,
                    ir=input.ir0, iz=iz, ivzeta=input.ivzeta0, ivr=input.ivr0)
            error = f .- f_sym

            fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
            plot_1d(vz.grid, f, ax=ax[1], label="num",
                    xlabel=L"v_{z}/L_{v_{z}}", ylabel=field_label)
            plot_1d(vz.grid, f_sym, ax=ax[1], label="sym")
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                   orientation=:horizontal)

            plot_1d(vz.grid, error, ax=ax[2], xlabel=L"v_{z}/L_{v_{z}}",
                    ylabel=norm_label)

            outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz.pdf"
            save(outfile, fig)

            if has_rdim
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ivzeta=input.ivzeta0, ivr=input.ivr0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vz.grid, r.grid, f, ax=ax[1], colorbar_place=colorbar_place[1],
                        title=field_label, xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"r")
                plot_2d(vz.grid, r.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{z}/L_{v_{z}}", ylabel=L"r")
                plot_2d(vz.grid, r.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{z}/L_{v_{z}}", ylabel=L"r")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz_r.pdf"
                save(outfile, fig)
            end

            if !is_1V
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ir=input.ir0, ivzeta=input.ivzeta0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vz.grid, vr.grid, f, ax=ax[1],
                        colorbar_place=colorbar_place[1], title=field_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{r}/L_{v_{r}}")
                plot_2d(vz.grid, vr.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{r}/L_{v_{r}}")
                plot_2d(vz.grid, vr.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{r}/L_{v_{r}}")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz_vr.pdf"
                save(outfile, fig)

                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ir=input.ir0, ivr=input.ivr0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vz.grid, vzeta.grid, f, ax=ax[1],
                        colorbar_place=colorbar_place[1], title=field_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{\zeta}/L_{v_{\zeta}}")
                plot_2d(vz.grid, vzeta.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{\zeta}/L_{v_{\zeta}}")
                plot_2d(vz.grid, vzeta.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{\zeta}/L_{v_{\zeta}}")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz_vzeta.pdf"
                save(outfile, fig)
            end
        end
    end

    animate_dims = setdiff(neutral_dimensions, (:sn,))
    if !has_rdim
        animate_dims = setdiff(animate_dims, (:r,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    if is_1V
        animate_dims = setdiff(animate_dims, (:vzeta, :vr))
    end
    plot_dims = tuple(:t, animate_dims...)
    _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                   field_sym_label, norm_label, plot_dims, animate_dims, true)

    return field_norm
end

"""
    manufactured_solutions_analysis(run_info; plot_prefix)
    manufactured_solutions_analysis(run_info::Tuple; plot_prefix)

Compare computed and manufactured solutions for field and moment variables for a 'method
of manufactured solutions' (MMS) test.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

Settings are read from the `[manufactured_solns]` section of the input.

While a Tuple of `run_info` can be passed for compatibility with `makie_post_process()`,
at present comparison of multiple runs is not supported - passing a Tuple of length
greater than one will result in an error.
"""
function manufactured_solutions_analysis end

function manufactured_solutions_analysis(run_info::Tuple; plot_prefix, nvperp)
    if !any(ri !== nothing && ri.manufactured_solns_input.use_for_advance &&
            ri.manufactured_solns_input.use_for_init for ri ∈ run_info)
        # No manufactured solutions tests
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict["manufactured_solns"])
    if !any(v for v ∈ values(input) if isa(v, Bool))
        # Skip as there is nothing to do
        return nothing
    end

    if length(run_info) > 1
        println("Analysing more than one run at once not supported for"
                * "manufactured_solutions_analysis()")
        return nothing
    end
    try
        return manufactured_solutions_analysis(run_info[1]; plot_prefix=plot_prefix,
                                               nvperp=nvperp)
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in manufactured_solutions_analysis().")
    end
end

function manufactured_solutions_analysis(run_info; plot_prefix, nvperp)
    manufactured_solns_input = run_info.manufactured_solns_input
    if !(manufactured_solns_input.use_for_advance && manufactured_solns_input.use_for_init)
        return nothing
    end

    if nvperp === nothing
        error("No `nvperp` found - must have distributions function outputs to plot MMS "
              * "tests")
    end

    input = Dict_to_NamedTuple(input_dict["manufactured_solns"])

    open(run_info.run_prefix * "MMS_errors.txt", "w") do io
        println_to_stdout_and_file(io, "# ", run_info.run_name)
        println_to_stdout_and_file(io, join(run_info.time, " "), " # time / (Lref/cref): ")

        for (variable_name, field_label, field_sym_label, norm_label) ∈
                (("phi", L"\tilde{\phi}", L"\tilde{\phi}^{sym}", L"\varepsilon(\tilde{\phi})"),
                 ("Er", L"\tilde{E}_r", L"\tilde{E}_r^{sym}", L"\varepsilon(\tilde{E}_r)"),
                 ("Ez", L"\tilde{E}_z", L"\tilde{E}_z^{sym}", L"\varepsilon(\tilde{E}_z)"),
                 ("density", L"\tilde{n}_i", L"\tilde{n}_i^{sym}", L"\varepsilon(\tilde{n}_i)"),
                 ("parallel_flow", L"\tilde{u}_{i,\parallel}", L"\tilde{u}_{i,\parallel}^{sym}", L"\varepsilon(\tilde{u}_{i,\parallel})"),
                 ("parallel_pressure", L"\tilde{p}_{i,\parallel}", L"\tilde{p}_{i,\parallel}^{sym}", L"\varepsilon(\tilde{p}_{i,\parallel})"),
                 ("density_neutral", L"\tilde{n}_n", L"\tilde{n}_n^{sym}", L"\varepsilon(\tilde{n}_n)"))

            if contains(variable_name, "neutral") && run_info.n_neutral_species == 0
                continue
            end
            if contains(variable_name, "Er") && run_info.r.n_global == 1
                continue
            end

            compare_moment_symbolic_test(run_info, plot_prefix, field_label, field_sym_label,
                                         norm_label, variable_name; io=io, input=input,
                                         nvperp=nvperp)
        end
    end

    return nothing
end

"""
    manufactured_solutions_analysis_dfns(run_info; plot_prefix)
    manufactured_solutions_analysis_dfns(run_info::Tuple; plot_prefix)

Compare computed and manufactured solutions for distribution function variables for a
'method of manufactured solutions' (MMS) test.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

Settings are read from the `[manufactured_solns]` section of the input.

While a Tuple of `run_info` can be passed for compatibility with `makie_post_process()`,
at present comparison of multiple runs is not supported - passing a Tuple of length
greater than one will result in an error.
"""
function manufactured_solutions_analysis_dfns end

function manufactured_solutions_analysis_dfns(run_info::Tuple; plot_prefix)
    if !any(ri !== nothing && ri.manufactured_solns_input.use_for_advance &&
            ri.manufactured_solns_input.use_for_init for ri ∈ run_info)
        # No manufactured solutions tests
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])
    if !any(v for v ∈ values(input) if isa(v, Bool))
        # Skip as there is nothing to do
        return nothing
    end

    if length(run_info) > 1
        println("Analysing more than one run at once not supported for"
                * "manufactured_solutions_analysis_dfns()")
        return nothing
    end
    try
        return manufactured_solutions_analysis_dfns(run_info[1]; plot_prefix=plot_prefix)
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in manufactured_solutions_analysis_dfns().")
    end
end

function manufactured_solutions_analysis_dfns(run_info; plot_prefix)
    manufactured_solns_input = run_info.manufactured_solns_input
    if !(manufactured_solns_input.use_for_advance && manufactured_solns_input.use_for_init)
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])

    open(run_info.run_prefix * "MMS_dfns_errors.txt", "w") do io
        println_to_stdout_and_file(io, "# ", run_info.run_name)
        println_to_stdout_and_file(io, join(run_info.time, " "), " # time / (Lref/cref): ")

        compare_ion_pdf_symbolic_test(run_info, plot_prefix; io=io, input=input)

        if run_info.n_neutral_species > 0
            compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=io, input=input)
        end
    end

    return nothing
end

"""
    timestep_diagnostics(run_info, run_info_dfns; plot_prefix=nothing, it=nothing)

Plot a time-trace of some adaptive-timestep diagnostics: steps per output, timestep
failures per output, how many times per output each variable caused a timestep failure,
and which factor limited the length of successful timesteps (CFL, accuracy, max_timestep).

If `plot_prefix` is passed, it gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix_timestep_diagnostics.pdf`.

`it` can be used to select a subset of the time points by passing a range.
"""
function timestep_diagnostics(run_info, run_info_dfns; plot_prefix=nothing, it=nothing,
                              electron=false)
    if !isa(run_info, Tuple)
        run_info = (run_info,)
    end

    input = Dict_to_NamedTuple(input_dict["timestep_diagnostics"])

    if input.plot || input.animate_CFL || input.plot_timestep_residual ||
            input.animate_timestep_residual || input.plot_timestep_error ||
            input.animate_timestep_error || input.plot_steady_state_residual ||
            input.animate_steady_state_residual
        if electron
            println("Making electron timestep diagnostics plots")
        else
            println("Making timestep diagnostics plots")
        end
    end

    steps_fig = nothing
    dt_fig = nothing
    CFL_fig = nothing

    if electron
        electron_prefix = "electron_"
    else
        electron_prefix = ""
    end

    if input.plot
        try
            # Plot numbers of steps and numbers of failures
            ###############################################

            steps_fig, ax = get_1d_ax(; xlabel="time", ylabel="number of steps per output")
            # Put failures a separate y-axis
            ax_failures = Axis(steps_fig[1, 1]; ylabel="number of failures per output",
                               yaxisposition = :right)
            hidespines!(ax_failures)
            hidexdecorations!(ax_failures)
            hideydecorations!(ax_failures; ticks=false, label=false, ticklabels=false)

            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end

                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end
                plot_1d(time, get_variable(ri, "$(electron_prefix)steps_per_output";
                                              it=it); label=prefix * "steps", ax=ax)
                # Fudge to create an invisible line on ax_failures that cycles the line colors
                # and adds a label for "steps_per_output" to the plot because we create the
                # legend from ax_failures.
                plot_1d([ri.time[1]], [0]; label=prefix * "steps", ax=ax_failures)
                plot_1d(time,
                        get_variable(ri, "$(electron_prefix)failures_per_output"; it=it);
                        label=prefix * "failures", ax=ax_failures)

                failure_caused_by_per_output =
                    get_variable(ri, "$(electron_prefix)failure_caused_by_per_output";
                                 it=it)
                counter = 0
                # pdf failure counter
                counter += 1
                if electron
                    label = prefix * "failures caused by f_electron"
                else
                    label = prefix * "failures caused by f_ion"
                end
                plot_1d(time, @view failure_caused_by_per_output[counter,:];
                        label=label, ax=ax_failures)
                if !electron && ri.evolve_density
                    # Ion density failure counter
                    counter += 1
                    plot_1d(time, @view failure_caused_by_per_output[counter,:];
                            linestyle=:dash, label=prefix * "failures caused by n_ion",
                            ax=ax_failures)
                end
                if !electron && ri.evolve_upar
                    # Ion flow failure counter
                    counter += 1
                    plot_1d(time, @view failure_caused_by_per_output[counter,:];
                            linestyle=:dash, label=prefix * "failures caused by u_ion",
                            ax=ax_failures)
                end
                if !electron && ri.evolve_ppar
                    # Ion parallel pressure failure counter
                    counter += 1
                    plot_1d(time, @view failure_caused_by_per_output[counter,:];
                            linestyle=:dash, label=prefix * "failures caused by p_ion",
                            ax=ax_failures)
                end
                if electron || ri.composition.electron_physics ∈ (braginskii_fluid,
                                                                  kinetic_electrons,
                                                                  kinetic_electrons_with_temperature_equation)
                    # Electron parallel pressure failure counter
                    counter += 1
                    plot_1d(time, @view failure_caused_by_per_output[counter,:];
                            linestyle=:dash, label=prefix * "failures caused by p_electron",
                            ax=ax_failures)
                    if !electron && ri.composition.electron_physics ∈ (kinetic_electrons,
                                                                       kinetic_electrons_with_temperature_equation)
                        # Kinetic electron nonlinear solver failure
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash, label=prefix * "failures caused by kinetic electron solve",
                                ax=ax_failures)
                    end
                end
                if !electron && ri.n_neutral_species > 0
                    # Neutral pdf failure counter
                    counter += 1
                    plot_1d(time, @view failure_caused_by_per_output[counter,:];
                            label=prefix * "failures caused by f_neutral", ax=ax_failures)
                    if ri.evolve_density
                        # Neutral density failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash,
                                label=prefix * "failures caused by n_neutral", ax=ax_failures)
                    end
                    if ri.evolve_upar
                        # Neutral flow failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash,
                                label=prefix * "failures caused by u_neutral", ax=ax_failures)
                    end
                    if ri.evolve_ppar
                        # Neutral flow failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash,
                                label=prefix * "failures caused by p_neutral", ax=ax_failures)
                    end
                    if occursin("ARK", ri.t_input["type"])
                        # Nonlinear iteration failed to converge in implicit part of
                        # timestep
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dot,
                                label=prefix * "nonlinear iteration convergence failure", ax=ax_failures)
                    end
                    if ri.composition.electron_physics ∈ (kinetic_electrons,
                                                          kinetic_electrons_with_temperature_equation)
                        # Kinetic electron iteration failed to converge
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dot,
                                label=prefix * "nonlinear iteration convergence failure", ax=ax_failures)
                    end
                end

                if counter > size(failure_caused_by_per_output, 1)
                    error("Tried to plot non-existent variables in "
                          * "failure_caused_by_per_output. Settings not understood "
                          * "correctly.")
                end
                if counter < size(failure_caused_by_per_output, 1)
                    error("Some variables in failure_caused_by_per_output not plotted. "
                          * "Settings not understood correctly.")
                end
            end

            put_legend_right(steps_fig, ax_failures)

            if plot_prefix !== nothing
                outfile = plot_prefix * electron_prefix * "timestep_diagnostics.pdf"
                save(outfile, steps_fig)
            else
                display(steps_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() steps_fig.")
        end

        try
            # Plot average timesteps
            ########################

            if plot_prefix !== nothing
                outfile = plot_prefix * "$(electron_prefix)successful_dt.pdf"
            else
                outfile = nothing
            end
            dt_fig = plot_vs_t(run_info, "$(electron_prefix)average_successful_dt"; outfile=outfile)

            if plot_prefix === nothing
                display(dt_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() dt_fig.")
        end

        try
            # PLot minimum CFL factors
            ##########################

            CFL_fig, ax = get_1d_ax(; xlabel="time", ylabel="(grid spacing) / speed")
            #maxval = Inf
            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                if electron
                    CFL_vars = ["minimum_CFL_electron_z", "minimum_CFL_electron_vpa"]
                    implicit_CFL_vars = String[]
                else
                    CFL_vars = String[]
                    implicit_CFL_vars = String[]

                    push!(CFL_vars, "minimum_CFL_ion_z")
                    if occursin("ARK", ri.t_input["type"]) && ri.t_input["implicit_ion_advance"]
                        push!(implicit_CFL_vars, "minimum_CFL_ion_z")
                    end
                    push!(CFL_vars, "minimum_CFL_ion_vpa")
                    if occursin("ARK", ri.t_input["type"]) && (ri.t_input["implicit_ion_advance"] || ri.t_input["implicit_vpa_advection"])
                        push!(implicit_CFL_vars, "minimum_CFL_ion_vpa")
                    end
                    if ri.n_neutral_species > 0
                        push!(CFL_vars, "minimum_CFL_neutral_z", "minimum_CFL_neutral_vz")
                    end
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end
                for varname ∈ CFL_vars
                    var = get_variable(ri, varname)
                    #maxval = NaNMath.min(maxval, NaNMath.maximum(var))
                    if occursin("neutral", varname)
                        if varname ∈ implicit_CFL_vars
                            linestyle = :dashdot
                        else
                            linestyle = :dash
                        end
                    else
                        if varname ∈ implicit_CFL_vars
                            linestyle = :dot
                        else
                            linestyle = nothing
                        end
                    end
                    plot_1d(time, var; ax=ax, label=prefix*electron_prefix*varname,
                            linestyle=linestyle, yscale=log10,
                            transform=x->positive_or_nan(x; epsilon=1.e-20))
                end
            end
            #ylims!(ax, 0.0, 10.0 * maxval)
            put_legend_right(CFL_fig, ax)

            if plot_prefix !== nothing
                outfile = plot_prefix * electron_prefix * "CFL_factors.pdf"
                save(outfile, CFL_fig)
            else
                display(CFL_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() CFL_fig.")
        end

        try
            limits_fig, ax = get_1d_ax(; xlabel="time", ylabel="number of limits per factor per output",
                                       size=(600, 500))

            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                limit_caused_by_per_output =
                    get_variable(ri, "$(electron_prefix)limit_caused_by_per_output";
                                 it=it)
                counter = 0

                # Maximum timestep increase limit counter
                counter += 1
                plot_1d(time, @view limit_caused_by_per_output[counter,:];
                        label=prefix * "max timestep increase", ax=ax)

                # Slower maximum timestep increase near last failure limit counter
                counter += 1
                plot_1d(time, @view limit_caused_by_per_output[counter,:];
                        label=prefix * "max timestep increase near last fail", ax=ax)

                # Minimum timestep limit counter
                counter += 1
                plot_1d(time, @view limit_caused_by_per_output[counter,:];
                        label=prefix * "min timestep", ax=ax)

                # Maximum timestep limit counter
                counter += 1
                plot_1d(time, @view limit_caused_by_per_output[counter,:];
                        label=prefix * "max timestep", ax=ax)

                # High nonlinear iterations count
                counter += 1
                plot_1d(time, @view limit_caused_by_per_output[counter,:];
                        label=prefix * "high nl iterations", ax=ax)

                # Accuracy limit counters
                counter += 1
                if electron
                    label = prefix * "electron pdf RK accuracy"
                else
                    label = prefix * "ion pdf RK accuracy"
                end
                plot_1d(time, @view limit_caused_by_per_output[counter,:];
                        label=label, ax=ax, linestyle=:dash)
                if !electron && ri.evolve_density
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "ion density RK accuracy", ax=ax,
                            linestyle=:dash)
                end
                if !electron && ri.evolve_upar
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "ion upar RK accuracy", ax=ax,
                            linestyle=:dash)
                end
                if !electron && ri.evolve_ppar
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "ion ppar RK accuracy", ax=ax,
                            linestyle=:dash)
                end
                if electron || ri.composition.electron_physics ∈ (braginskii_fluid,
                                                                  kinetic_electrons,
                                                                  kinetic_electrons_with_temperature_equation)
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "electron ppar RK accuracy", ax=ax,
                            linestyle=:dash)
                end
                if !electron && ri.n_neutral_species > 0
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "neutral pdf RK accuracy", ax=ax,
                            linestyle=:dash)
                    if ri.evolve_density
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "neutral density RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                    if ri.evolve_upar
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "neutral uz RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                    if ri.evolve_ppar
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "neutral pz RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                end

                if electron || !(occursin("ARK", ri.t_input["type"]) && ri.t_input["implicit_ion_advance"])
                    # Ion z advection
                    counter += 1
                    if electron
                        label = prefix * "electron z advect"
                    else
                        label = prefix * "ion z advect"
                    end
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=label, ax=ax, linestyle=:dot)
                end

                if electron || !(occursin("ARK", ri.t_input["type"]) && (ri.t_input["implicit_ion_advance"] || ri.t_input["implicit_vpa_advection"]))
                    # Ion vpa advection
                    counter += 1
                    if electron
                        label = prefix * "electron vpa advect"
                    else
                        label = prefix * "ion vpa advect"
                    end
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=label, ax=ax, linestyle=:dot)
                end

                if !electron && ri.n_neutral_species > 0
                    # Neutral z advection
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "neutral z advect", ax=ax, linestyle=:dot)

                    # Neutral vz advection
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "neutral vz advect", ax=ax, linestyle=:dot)
                end

                if counter > size(limit_caused_by_per_output, 1)
                    error("Tried to plot non-existent variables in "
                          * "limit_caused_by_per_output. Settings not understood "
                          * "correctly.")
                end
                if counter < size(limit_caused_by_per_output, 1)
                    error("Some variables in limit_caused_by_per_output not plotted. "
                          * "Settings not understood correctly.")
                end
            end

            put_legend_right(limits_fig, ax)

            if plot_prefix !== nothing
                outfile = plot_prefix * electron_prefix * "timestep_limits.pdf"
                save(outfile, limits_fig)
            else
                display(limits_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() limits_fig.")
        end

        try
            # Plot nonlinear solver diagnostics (if any)
            nl_solvers_fig, ax = get_1d_ax(; xlabel="time", ylabel="iterations per solve/nonlinear-iteration")
            has_nl_solver = false

            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                nl_nonlinear_iterations_names = Tuple(v for v ∈ ri.variable_names
                                                      if occursin("_nonlinear_iterations", v))
                if nl_nonlinear_iterations_names != ()
                    has_nl_solver = true
                    nl_prefixes = (split(v, "_nonlinear_iterations")[1]
                                   for v ∈ nl_nonlinear_iterations_names)
                    for p ∈ nl_prefixes
                        nonlinear_iterations = get_variable(ri, "$(p)_nonlinear_iterations_per_solve")
                        linear_iterations = get_variable(ri, "$(p)_linear_iterations_per_nonlinear_iteration")
                        plot_1d(time, nonlinear_iterations, label=prefix * " " * p * " NL per solve", ax=ax)
                        plot_1d(time, linear_iterations, label=prefix * " " * p * " L per NL", ax=ax)
                    end
                end
            end

            if has_nl_solver
                put_legend_right(nl_solvers_fig, ax)

                if plot_prefix !== nothing
                    outfile = plot_prefix * "nonlinear_solver_iterations.pdf"
                    save(outfile, nl_solvers_fig)
                else
                    display(nl_solvers_fig)
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() nl_solvers_fig.")
        end

        try
            # Plot electron solver diagnostics
            electron_solver_fig, ax = get_1d_ax(; xlabel="time", ylabel="electron steps per ion step")

            has_electron_solve = false
            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                if ri.composition.electron_physics ∈ (kinetic_electrons,
                                                      kinetic_electrons_with_temperature_equation)
                    has_electron_solve = true
                    electron_steps_per_ion_step = get_variable(ri, "electron_steps_per_ion_step")
                    plot_1d(time, electron_steps_per_ion_step, label=prefix * " electron steps per solve", ax=ax)
                end
            end

            if has_electron_solve
                put_legend_right(electron_solver_fig, ax)

                if has_electron_solve
                    outfile = plot_prefix * "electron_steps.pdf"
                    save(outfile, electron_solver_fig)
                else
                    display(electron_solver_fig)
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() nl_solvers_fig.")
        end
    end

    if input.animate_CFL
        try
            if plot_prefix === nothing
                error("plot_prefix is required when animate_CFL=true")
            end
            if !electron
                data = get_variable(run_info, "CFL_ion_z")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_ion_z"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_ion_z_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
                data = get_variable(run_info, "CFL_ion_vpa")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_ion_vpa"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_ion_vpa_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
            end
            if electron || any(ri.composition.electron_physics ∈ (kinetic_electrons,
                                                                  kinetic_electrons_with_temperature_equation)
                               for ri ∈ run_info)
                data = get_variable(run_info, "CFL_electron_z")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_electron_z"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_electron_z_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
                data = get_variable(run_info, "CFL_electron_vpa")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_electron_vpa"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_electron_vpa_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
            end
            if !electron && any(ri.n_neutral_species > 0 for ri ∈ run_info)
                data = get_variable(run_info, "CFL_neutral_z")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vz_z(run_info, "CFL_neutral_z"; data=data, it=it,
                                outfile=plot_prefix * "CFL_neutral_z_vs_vz_z.gif",
                                colorscale=log10,
                                transform=x->positive_or_nan(x; epsilon=1.e-30),
                                colorrange=(datamin, datamin * 1000.0),
                                axis_args=Dict(:bottomspinevisible=>false,
                                               :topspinevisible=>false,
                                               :leftspinevisible=>false,
                                               :rightspinevisible=>false))
                data = get_variable(run_info, "CFL_neutral_vz")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vz_z(run_info, "CFL_neutral_vz"; data=data, it=it,
                                outfile=plot_prefix * "CFL_neutral_vz_vs_vz_z.gif",
                                colorscale=log10,
                                transform=x->positive_or_nan(x; epsilon=1.e-30),
                                colorrange=(datamin, datamin * 1000.0),
                                axis_args=Dict(:bottomspinevisible=>false,
                                               :topspinevisible=>false,
                                               :leftspinevisible=>false,
                                               :rightspinevisible=>false))
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() CFL animations.")
        end
    end

    if run_info_dfns[1].dfns
        this_input_dict = input_dict_dfns
    else
        this_input_dict = input_dict
    end
    if electron
        variable_list = (v for v ∈ union((ri.evolving_variables for ri in run_info_dfns)...)
                         if occursin("electron", v))
    else
        variable_list = (v for v ∈ union((ri.evolving_variables for ri in run_info_dfns)...)
                         if !occursin("electron", v))
    end
    all_variable_names = union((ri.variable_names for ri ∈ run_info_dfns)...)

    if input.plot_timestep_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_timestep_residual"
                if variable_name == "f_neutral"
                    plot_vs_vz_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_vz_z.pdf")
                elseif variable_name ∈ ("f", "f_electron")
                    plot_vs_vpa_z(run_info_dfns, residual_name;
                                  input=this_input_dict[variable_name],
                                  outfile=plot_prefix * residual_name * "_vs_vpa_z.pdf")
                else
                    plot_vs_z(run_info_dfns, residual_name;
                              input=this_input_dict[variable_name],
                              outfile=plot_prefix * residual_name * "_vs_z.pdf")
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep residual plots.")
        end
    end

    if input.animate_timestep_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_timestep_residual"
                if variable_name == "f_neutral"
                    animate_vs_vz_z(run_info_dfns, residual_name;
                                    input=this_input_dict[variable_name],
                                    outfile=plot_prefix * residual_name * "_vs_vz_z." * this_input_dict[variable_name]["animation_ext"])
                elseif variable_name ∈ ("f", "f_electron")
                    animate_vs_vpa_z(run_info_dfns, residual_name;
                                     input=this_input_dict[variable_name],
                                     outfile=plot_prefix * residual_name * "_vs_vpa_z." * this_input_dict[variable_name]["animation_ext"])
                else
                    animate_vs_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_z." * this_input_dict[variable_name]["animation_ext"])
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep residual animations.")
        end
    end

    if input.plot_timestep_error
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate error for this variable
                    continue
                end
                error_name = variable_name * "_timestep_error"
                if variable_name == "f_neutral"
                    plot_vs_vz_z(run_info_dfns, error_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * error_name * "_vs_vz_z.pdf")
                elseif variable_name ∈ ("f", "f_electron")
                    plot_vs_vpa_z(run_info_dfns, error_name;
                                  input=this_input_dict[variable_name],
                                  outfile=plot_prefix * error_name * "_vs_vpa_z.pdf")
                else
                    plot_vs_z(run_info_dfns, error_name;
                              input=this_input_dict[variable_name],
                              outfile=plot_prefix * error_name * "_vs_z.pdf")
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep error plots.")
        end
    end

    if input.animate_timestep_error
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate error for this variable
                    continue
                end
                error_name = variable_name * "_timestep_error"
                if variable_name == "f_neutral"
                    animate_vs_vz_z(run_info_dfns, error_name;
                                    input=this_input_dict[variable_name],
                                    outfile=plot_prefix * error_name * "_vs_vz_z." * this_input_dict[variable_name]["animation_ext"])
                elseif variable_name ∈ ("f", "f_electron")
                    animate_vs_vpa_z(run_info_dfns, error_name;
                                     input=this_input_dict[variable_name],
                                     outfile=plot_prefix * error_name * "_vs_vpa_z." * this_input_dict[variable_name]["animation_ext"])
                else
                    animate_vs_z(run_info_dfns, error_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * error_name * "_vs_z." * this_input_dict[variable_name]["animation_ext"])
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep error animations.")
        end
    end

    if input.plot_steady_state_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_steady_state_residual"
                if variable_name == "f_neutral"
                    plot_vs_vz_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_vz_z.pdf")
                elseif variable_name ∈ ("f", "f_electron")
                    plot_vs_vpa_z(run_info_dfns, residual_name;
                                  input=this_input_dict[variable_name],
                                  outfile=plot_prefix * residual_name * "_vs_vpa_z.pdf")
                else
                    plot_vs_z(run_info_dfns, residual_name;
                              input=this_input_dict[variable_name],
                              outfile=plot_prefix * residual_name * "_vs_z.pdf")
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() steady state residual plots.")
        end
    end

    if input.animate_steady_state_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_steady_state_residual"
                if variable_name == "f_neutral"
                    animate_vs_vz_z(run_info_dfns, residual_name;
                                    input=this_input_dict[variable_name],
                                    outfile=plot_prefix * residual_name * "_vs_vz_z." * this_input_dict[variable_name]["animation_ext"])
                elseif variable_name ∈ ("f", "f_electron")
                    animate_vs_vpa_z(run_info_dfns, residual_name;
                                     input=this_input_dict[variable_name],
                                     outfile=plot_prefix * residual_name * "_vs_vpa_z." * this_input_dict[variable_name]["animation_ext"])
                else
                    animate_vs_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_z." * this_input_dict[variable_name]["animation_ext"])
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() steady state residual animations.")
        end
    end

    return steps_fig, dt_fig, CFL_fig
end

"""
    timing_data(run_info; plot_prefix=nothing, threshold=nothing,
                include_patterns=nothing, exclude_patterns=nothing, ranks=nothing,
                figsize=nothing, include_legend=true)

Plot timings from different parts of the `moment_kinetics` code. Only timings from
function calls during the time evolution loop are included, not from the setup, because we
plot versus time.

To reduce clutter, timings whose total time (at the final time point) is less than
`threshold` times the overall run time will be excluded. By default, `threshold` is
`1.0e-3`.

When there is more than one MPI rank present, the timings for each rank will be plotted
separately. The lines will be labelled with the MPI rank, with the position of the labels
moving along the lines one point at a time, to try to avoid overlapping many labels. If
the curves all overlap, this will look like one curve labelled by many MPI ranks.

There are many timers, so it can be useful to filter them to see only the most relevant
ones. By default all timers will be plotted. If `include_patterns` is passed, and
`exclude_patterns` is not, then only the total time and any timers that match
`include_patterns` (matches checked using `occursin()`) will be included in the plots. If
`exclude_patterns` is passed, then any timers that match (matches checked using
`occursin()`) `exclude_patterns` will be omitted, unless they match `include_patterns` in
which case they will still be included. If `ranks` is passed, then only the MPI ranks with
indices found in `ranks` will be included.

`figsize` can be passed to customize the size of the figures that plots are made on. This
can be useful because the legends may become very large when many timers are plotted, in
which case a larger figure might be needed.

`threshold`, `exclude_patterns`, `include_patterns`, `ranks`, and `figsize` can also be
set in `this_input_dict`. When this function is called as part of
[`makie_post_process`](@ref), [`input_dict`](@ref) is passed as `this_input_dict` so that
the settings are read from the post processing input file (by default
`post_processing_input.toml`). The function arguments take precedence, if they are given.

If you load GLMakie by doing `using GLMakie` before running this function, but after
calling `using makie_post_processing` (because `CairoMakie` is loaded when the module is
loaded and would take over if you load `GLMakie` before `makie_post_processing`), the
figures will be displayed in interactive windows. When you hover over a line some useful
information will be displayed.

Pass `include_legend=false` to remove legends from the figures. This is mostly useful for
interactive figures where hovering over the lines can show what they are, so that the
legend is not needed.
"""
function timing_data(run_info::Tuple; plot_prefix=nothing, threshold=nothing,
                     include_patterns=nothing, exclude_patterns=nothing, ranks=nothing,
                     this_input_dict=nothing, figsize=nothing, include_legend=true)

    if this_input_dict !== nothing
        input = Dict_to_NamedTuple(this_input_dict["timing_data"])
    else
        input = nothing
    end

    if input !== nothing && !input.plot
        return nothing
    end

    println("Making timing data plots")

    if figsize === nothing
        if input !== nothing
            figsize = Tuple(input.figsize)
        else
            figsize = (600,800)
        end
    end

    times_fig, times_ax, times_legend_place =
        get_1d_ax(; xlabel="time", ylabel="execution time per output step (s)", get_legend_place=:below,
                    size=figsize)
    ncalls_fig, ncalls_ax, ncalls_legend_place =
        get_1d_ax(; xlabel="time", ylabel="number of calls per output step", get_legend_place=:below,
                    size=figsize)
    allocs_fig, allocs_ax, allocs_legend_place =
        get_1d_ax(; xlabel="time", ylabel="allocations per output step (MB)", get_legend_place=:below,
                    size=figsize)

    for (irun,ri) ∈ enumerate(run_info)
        timing_data(ri; plot_prefix=plot_prefix, threshold=threshold,
                    include_patterns=include_patterns, exclude_patterns=exclude_patterns,
                    ranks=ranks, this_input_dict=this_input_dict, times_ax=times_ax,
                    ncalls_ax=ncalls_ax, allocs_ax=allocs_ax, irun=irun, figsize=figsize)
    end

    if string(Makie.current_backend()) == "GLMakie"
        # Can make interactive plots

        backend = Makie.current_backend()

        if include_legend
            Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(times_fig)
        display(backend.Screen(), times_fig)

        if include_legend
            Legend(ncalls_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(ncalls_fig)
        display(backend.Screen(), ncalls_fig)

        if include_legend
            Legend(allocs_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(allocs_fig)
        display(backend.Screen(), allocs_fig)
    elseif plot_prefix !== nothing
        if include_legend
            Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=true, merge=true)
        end
        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(times_fig.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(times_fig)
        outfile = plot_prefix * "execution_times.pdf"
        save(outfile, times_fig)

        if include_legend
            Legend(ncalls_fig[2,1], ncalls_ax; tellheight=true, tellwidth=true, merge=true)
        end
        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(ncalls_fig.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(ncalls_fig)
        outfile = plot_prefix * "ncalls.pdf"
        save(outfile, ncalls_fig)

        if include_legend
            Legend(allocs_fig[2,1], allocs_ax; tellheight=true, tellwidth=true, merge=true)
        end
        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(allocs_fig.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(allocs_fig)
        outfile = plot_prefix * "allocations.pdf"
        save(outfile, allocs_fig)
    end

    return times_fig, ncalls_fig, allocs_fig
end

function timing_data(run_info; plot_prefix=nothing, threshold=nothing,
                     include_patterns=nothing, exclude_patterns=nothing, ranks=nothing,
                     this_input_dict=nothing, times_ax=nothing, ncalls_ax=nothing,
                     allocs_ax=nothing, irun=1, figsize=nothing,
                     include_legend=true)

    if this_input_dict !== nothing
        input = Dict_to_NamedTuple(this_input_dict["timing_data"])
    else
        input = nothing
    end

    if input !== nothing && !input.plot
        return nothing
    end

    if figsize === nothing
        if input !== nothing
            figsize = Tuple(input.figsize)
        else
            figsize = (600,800)
        end
    end

    if threshold === nothing
        if input !== nothing
            threshold = input.threshold
        else
            threshold = 1.0e-2
        end
    end
    if isa(include_patterns, AbstractString)
        include_patterns = [include_patterns]
    end
    if isa(exclude_patterns, AbstractString)
        exclude_patterns = [exclude_patterns]
    end
    if input !== nothing && include_patterns === nothing
        include_patterns = input.include_patterns
        if length(include_patterns) == 0
            include_patterns = nothing
        end
    end
    if input !== nothing && exclude_patterns === nothing
        exclude_patterns = input.exclude_patterns
        if length(exclude_patterns) == 0
            exclude_patterns = nothing
        end
    end
    if input !== nothing && ranks === nothing
        ranks = input.ranks
    end

    if times_ax === nothing
        times_fig, times_ax, times_legend_place =
            get_1d_ax(; xlabel="time", ylabel="execution time per output step (s)",
                      get_legend_place=:below, size=figsize)
    else
        times_fig = nothing
    end
    if ncalls_ax === nothing
        ncalls_fig, ncalls_ax, ncalls_legend_place =
            get_1d_ax(; xlabel="time", ylabel="number of calls per output step", get_legend_place=:below)
    else
        ncalls_fig = nothing
    end
    if allocs_ax === nothing
        allocs_fig, allocs_ax, allocs_legend_place =
            get_1d_ax(; xlabel="time", ylabel="allocations per output step (MB)", get_legend_place=:below)
    else
        allocs_fig = nothing
    end

    linestyles = linestyle=[:solid, :dash, :dot, :dashdot, :dashdotdot]
    time_advance_timer_variables = [v for v ∈ run_info.timing_variable_names if occursin("time_advance! step", v)]
    time_variables = [v for v ∈ time_advance_timer_variables if startswith(v, "time:")]
    ncalls_variables = [v for v ∈ time_advance_timer_variables if startswith(v, "ncalls:")]
    allocs_variables = [v for v ∈ time_advance_timer_variables if startswith(v, "allocs:")]

    timing_group = "timing_data"

    function label_irank(ax, variable, irank, color, unit_conversion=1)
        if run_info.nrank > 1
            # Label curves with irank so we can tell which is which
            index = ((irank + 1) % (length(variable) - 1)) + 1
            with_theme
            text!(ax, run_info.time[index],
                  variable[index] * unit_conversion;
                  text="$irank", color=color)
        end
    end

    function check_include_exclude(variable_name)
        explicitly_included = (include_patterns !== nothing &&
                               any(occursin(p, variable_name) for p ∈ include_patterns))
        if exclude_patterns === nothing && include_patterns !== nothing
            excluded = !explicitly_included
        elseif exclude_patterns !== nothing
            if !explicitly_included &&
                    any(occursin(p, variable_name) for p ∈ exclude_patterns)
                excluded = true
            else
                excluded = false
            end
        else
            excluded = false
        end
        return excluded, explicitly_included
    end

    # Plot the total time
    time_unit_conversion = 1.0e-9 # ns to s
    total_time_variable_name = "time:moment_kinetics;time_advance! step"
    total_time = get_variable(run_info, total_time_variable_name * "_per_step",
                              group=timing_group)
    for irank ∈ 0:run_info.nrank-1
        label = "time_advance! step"
        irank_slice = total_time[irank+1,:]
        lines!(times_ax, run_info.time, irank_slice .* time_unit_conversion;
               color=:black, linestyle=linestyles[irun], label=label,
               inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
        label_irank(times_ax, irank_slice, irank, :black, time_unit_conversion)
    end
    mean_total_time = mean(total_time)
    for (variable_counter, variable_name) ∈ enumerate(time_variables)
        if variable_name == total_time_variable_name
            # Plotted this already
            continue
        end
        excluded, explicitly_included = check_include_exclude(variable_name)
        if excluded
            continue
        end
        variable = get_variable(run_info, variable_name * "_per_step",
                                group=timing_group)
        if !explicitly_included && mean(variable) < threshold * mean_total_time
            # This variable takes a very small amount of time, so skip.
            continue
        end
        for irank ∈ 0:run_info.nrank-1
            label = split(variable_name, "time_advance! step;")[2]
            irank_slice = variable[irank+1,:]
            l = lines!(times_ax, run_info.time, irank_slice .* time_unit_conversion;
                       color=Cycled(variable_counter), linestyle=linestyles[irun],
                       label=label, inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
            label_irank(times_ax, irank_slice, irank, l.color, time_unit_conversion)
        end
    end

    # Plot the number of calls
    total_ncalls_variable_name = "ncalls:moment_kinetics;time_advance! step"
    total_ncalls = get_variable(run_info, total_ncalls_variable_name * "_per_step",
                              group=timing_group)
    for irank ∈ 0:run_info.nrank-1
        label = "time_advance! step"
        irank_slice = total_ncalls[irank+1,:]
        lines!(ncalls_ax, run_info.time, irank_slice; color=:black,
               linestyle=linestyles[irun], label=label,
               inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
        label_irank(ncalls_ax, irank_slice, irank, :black)
    end
    mean_total_ncalls = mean(total_ncalls)
    for (variable_counter, variable_name) ∈ enumerate(ncalls_variables)
        if variable_name == total_ncalls_variable_name
            # Plotted this already
            continue
        end
        excluded, explicitly_included = check_include_exclude(variable_name)
        if excluded
            continue
        end
        variable = get_variable(run_info, variable_name * "_per_step",
                                group=timing_group)
        if !explicitly_included && mean(variable) < threshold * mean_total_ncalls
            # This variable takes a very small number of calls, so skip.
            continue
        end
        for irank ∈ 0:run_info.nrank-1
            label = split(variable_name, "time_advance! step;")[2]
            irank_slice = variable[irank+1,:]
            l = lines!(ncalls_ax, run_info.time, irank_slice;
                       color=Cycled(variable_counter), linestyle=linestyles[irun],
                       label=label, inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
            label_irank(ncalls_ax, irank_slice, irank, l.color)
        end
    end

    # Plot the total allocs
    allocs_unit_conversion = 2^(-20) # bytes to MB
    total_allocs_variable_name = "allocs:moment_kinetics;time_advance! step"
    total_allocs = get_variable(run_info, total_allocs_variable_name * "_per_step",
                                group=timing_group)
    for irank ∈ 0:run_info.nrank-1
        label = "time_advance! step"
        irank_slice = total_allocs[irank+1,:]
        lines!(allocs_ax, run_info.time, irank_slice .* allocs_unit_conversion;
               color=:black, linestyle=linestyles[irun], label=label,
               inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
        label_irank(allocs_ax, irank_slice, irank, :black, allocs_unit_conversion)
    end
    mean_total_allocs = mean(total_allocs)
    for (variable_counter, variable_name) ∈ enumerate(allocs_variables)
        if variable_name == total_allocs_variable_name
            # Plotted this already
            continue
        end
        excluded, explicitly_included = check_include_exclude(variable_name)
        if excluded
            continue
        end
        variable = get_variable(run_info, variable_name * "_per_step",
                                group=timing_group)
        if !explicitly_included && mean(variable) < threshold * mean_total_allocs
            # This variable represents a very small amount of allocs, so skip.
            continue
        end
        for irank ∈ 0:run_info.nrank-1
            label = split(variable_name, "time_advance! step;")[2]
            irank_slice = variable[irank+1,:]
            l = lines!(allocs_ax, run_info.time, irank_slice .* allocs_unit_conversion;
                       color=Cycled(variable_counter), linestyle=linestyles[irun],
                       label=label, inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
            label_irank(allocs_ax, irank_slice, irank, l.color, allocs_unit_conversion)
        end
    end

    if times_fig !== nothing && plot_prefix === nothing &&
            string(Makie.current_backend()) == "GLMakie"

        # Can make interactive plots

        backend = Makie.current_backend()

        if include_legend
            Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(times_fig)
        display(backend.Screen(), times_fig)

        if include_legend
            Legend(ncalls_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(ncalls_fig)
        display(backend.Screen(), ncalls_fig)

        if include_legend
            Legend(allocs_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(allocs_fig)
        display(backend.Screen(), allocs_fig)
    else
        if times_fig !== nothing
            if include_legend
                Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=true, merge=true)
            end
            # Ensure the first row width is 3/4 of the column width so that the plot does not
            # get squashed by the legend
            rowsize!(times_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(times_fig)
            if plot_prefix !== nothing
                outfile = plot_prefix * "execution_times.pdf"
                save(outfile, times_fig)
            end
        end

        if ncalls_fig !== nothing
            if include_legend
                Legend(ncalls_fig[2,1], ncalls_ax; tellheight=true, tellwidth=true, merge=true)
            end
            # Ensure the first row width is 3/4 of the column width so that the plot does not
            # get squashed by the legend
            rowsize!(ncalls_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(ncalls_fig)
            if plot_prefix !== nothing
                outfile = plot_prefix * "ncalls.pdf"
                save(outfile, ncalls_fig)
            end
        end

        if allocs_fig !== nothing
            if include_legend
                Legend(allocs_fig[2,1], allocs_ax; tellheight=true, tellwidth=true, merge=true)
            end
            # Ensure the first row width is 3/4 of the column width so that the plot does not
            # get squashed by the legend
            rowsize!(allocs_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(allocs_fig)
            if plot_prefix !== nothing
                outfile = plot_prefix * "allocations.pdf"
                save(outfile, allocs_fig)
            end
        end
    end

    return times_fig, ncalls_fig, allocs_fig
end


# Utility functions
###################
#
# These are more-or-less generic, but only used in this module for now, so keep them here.

"""
    clear_Dict!(d::AbstractDict)

Remove all entries from an AbstractDict, leaving it empty
"""
function clear_Dict!(d::AbstractDict)
    # This is one way to clear all entries from a dict, by using a filter which is false
    # for every entry
    if !isempty(d)
        filter!(x->false, d)
    end

    return d
end

"""
    convert_to_OrderedDicts!(d::AbstractDict)

Recursively convert an AbstractDict to OrderedDict.

Any nested AbstractDicts are also converted to OrderedDict.
"""
function convert_to_OrderedDicts!(d::AbstractDict)
    for (k, v) ∈ d
        if isa(v, AbstractDict)
            d[k] = convert_to_OrderedDicts!(v)
        end
    end
    return OrderedDict(d)
end

"""
    println_to_stdout_and_file(io, stuff...)

Print `stuff` both to stdout and to a file `io`.
"""
function println_to_stdout_and_file(io, stuff...)
    println(stuff...)
    if io !== nothing
        println(io, stuff...)
    end
end

"""
    positive_or_nan(x; epsilon=0)

If the argument `x` is zero or negative, replace it with NaN, otherwise return `x`.

`epsilon` can be passed if the number should be forced to be above some value (typically
we would assume epsilon is small and positive, but nothing about this function forces it
to be).
"""
function positive_or_nan(x; epsilon=0)
    return x > epsilon ? x : NaN
end

end
