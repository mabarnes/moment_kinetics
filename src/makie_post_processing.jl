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
       setup_makie_post_processing_input!, get_run_info
export animate_f_unnorm_vs_vpa, animate_f_unnorm_vs_vpa_z, get_1d_ax, get_2d_ax,
       irregular_heatmap, irregular_heatmap!, plot_f_unnorm_vs_vpa,
       plot_f_unnorm_vs_vpa_z, positive_or_nan, postproc_load_variable, positive_or_nan,
       put_legend_above, put_legend_below, put_legend_left, put_legend_right

using ..analysis: analyze_fields_data, check_Chodura_condition, get_r_perturbation,
                  get_Fourier_modes_2D, get_Fourier_modes_1D, steady_state_residuals,
                  get_unnormalised_f_dzdt_1d, get_unnormalised_f_coords_2d,
                  get_unnormalised_f_1d, vpagrid_to_dzdt_2d, get_unnormalised_f_2d
using ..array_allocation: allocate_float
using ..coordinates: define_coordinate
using ..input_structs: grid_input, advection_input, set_defaults_and_check_top_level!,
                       set_defaults_and_check_section!, Dict_to_NamedTuple
using ..krook_collisions: get_collision_frequency
using ..looping: all_dimensions, ion_dimensions, neutral_dimensions
using ..manufactured_solns: manufactured_solutions, manufactured_electric_fields
using ..moment_kinetics_input: mk_input
using ..load_data: open_readonly_output_file, get_group, load_block_data,
                   load_coordinate_data, load_distributed_charged_pdf_slice,
                   load_distributed_neutral_pdf_slice, load_input, load_mk_options,
                   load_species_data, load_time_data
using ..initial_conditions: vpagrid_to_dzdt
using ..post_processing: calculate_and_write_frequencies, construct_global_zr_coords,
                         get_geometry_and_composition, read_distributed_zr_data!
using ..type_definitions: mk_float, mk_int
using ..velocity_moments: integrate_over_vspace, integrate_over_neutral_vspace

using Combinatorics
using Glob
using LaTeXStrings
using LsqFit
using MPI
using NaNMath
using OrderedCollections
using TOML

using CairoMakie

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

const em_variables = ("phi", "Er", "Ez")
const ion_moment_variables = ("density", "parallel_flow", "parallel_pressure",
                              "thermal_speed", "temperature", "parallel_heat_flux",
                              "collision_frequency", "sound_speed", "mach_number")
const neutral_moment_variables = ("density_neutral", "uz_neutral", "pz_neutral",
                                  "thermal_speed_neutral", "temperature_neutral",
                                  "qz_neutral")
const all_moment_variables = tuple(em_variables..., ion_moment_variables...,
                                   neutral_moment_variables...)

const ion_dfn_variables = ("f",)
const neutral_dfn_variables = ("f_neutral",)
const all_dfn_variables = tuple(ion_dfn_variables..., neutral_dfn_variables...)

const ion_variables = tuple(ion_moment_variables..., ion_dfn_variables)
const neutral_variables = tuple(neutral_moment_variables..., neutral_dfn_variables)
const all_variables = tuple(all_moment_variables..., all_dfn_variables...)

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
given, plots comparing the runs in `run_dir...`.

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

`run_dir` is the path to an output directory, or (to make comparison plots) a tuple of
paths to output directories.

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

    is_1D = all(ri !== nothing && ri.r.n == 1 for ri ∈ run_info_moments)

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

    moment_variable_list = tuple(em_variables..., ion_moment_variables...)
    if has_neutrals
        moment_variable_list = tuple(moment_variable_list..., neutral_moment_variables...)
    end

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

    do_steady_state_residuals = any(input_dict[v]["steady_state_residual"]
                                    for v ∈ moment_variable_list)
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

    for variable_name ∈ moment_variable_list
        plots_for_variable(run_info, variable_name; plot_prefix=plot_prefix, is_1D=is_1D,
                           is_1V=is_1V,
                           steady_state_residual_fig_axes=steady_state_residual_fig_axes)
    end

    if do_steady_state_residuals
        _save_residual_plots(steady_state_residual_fig_axes, plot_prefix)
    end

    # Plots from distribution function variables
    ############################################
    if any(ri !== nothing for ri in run_info_dfns)
        dfn_variable_list = ion_dfn_variables
        if has_neutrals
            dfn_variable_list = tuple(dfn_variable_list..., neutral_dfn_variables...)
        end
        for variable_name ∈ dfn_variable_list
            plots_for_dfn_variable(run_info_dfns, variable_name; plot_prefix=plot_prefix,
                                   is_1D=is_1D, is_1V=is_1V)
        end
    end

    plot_charged_pdf_2D_at_wall(run_info_dfns; plot_prefix=plot_prefix)
    if has_neutrals
        plot_neutral_pdf_2D_at_wall(run_info_dfns; plot_prefix=plot_prefix)
    end

    if !is_1D
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

The `run_info` that you are using (as returned by [`get_run_info`](@ref)) should be passed
to `run_info_moments` (if it contains only the moments), or `run_info_dfns` (if it also
contains the distributions functions), or both (if you have loaded both sets of output).
This allows default values to be set based on the grid sizes and number of time points
read from the output files. Note that `setup_makie_post_processing_input!()` is called by
default at the end of `get_run_info()`, for conveinence in interactive use.

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
        nvperp_min = minimum(ri.vperp.n for ri in run_info if ri !== nothing)
        nvpa_min = minimum(ri.vpa.n for ri in run_info if ri !== nothing)
        nvzeta_min = minimum(ri.vzeta.n for ri in run_info if ri !== nothing)
        nvr_min = minimum(ri.vr.n for ri in run_info if ri !== nothing)
        nvz_min = minimum(ri.vz.n for ri in run_info if ri !== nothing)
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
    time_index_options = ("itime_min", "itime_max", "itime_skip", "itime_min_dfns",
                          "itime_max_dfns", "itime_skip_dfns")

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
      )

    section_defaults = OrderedDict(k=>v for (k,v) ∈ this_input_dict
                                   if !isa(v, AbstractDict) &&
                                      !(k ∈ time_index_options))
    for variable_name ∈ all_moment_variables
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
    if dfns
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
    end

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf";
        plot=false,
        animate=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "wall_pdf_neutral";
        plot=false,
        animate=false,
        colormap=this_input_dict["colormap"],
        animation_ext=this_input_dict["animation_ext"],
       )

    set_defaults_and_check_section!(
        this_input_dict, "Chodura_condition";
        plot_vs_t=false,
        plot_vs_r=false,
        plot_vs_r_t=false,
        it0=this_input_dict["it0"],
        ir0=this_input_dict["ir0"],
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

    return nothing
end

"""
    get_run_info(run_dir...; itime_min=1, itime_max=0,
                 itime_skip=1, dfns=false, do_setup=true, setup_input_file=nothing)
    get_run_info((run_dir, restart_index)...; itime_min=1, itime_max=0,
                 itime_skip=1, dfns=false, do_setup=true, setup_input_file=nothing)

Get file handles and other info for a single run

`run_dir` is the directory to read output from.

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
functions files.

The `itime_min`, `itime_max` and `itime_skip` options can be used to select only a slice
of time points when loading data. In `makie_post_process` these options are read from the
input (if they are set) before `get_run_info()` is called, so that the `run_info` returned
can be passed to [`setup_makie_post_processing_input!`](@ref), to be used for defaults for
the remaining options. If either `itime_min` or `itime_max` are ≤0, their values are used
as offsets from the final time index of the run.

By default `setup_makie_post_processing_input!()` is called at the end of
`get_run_info()`, for convenience when working interactively. Pass `do_setup=false` to
disable this. A post-processing input file can be passed to `setup_input_file` that will
be passed to `setup_makie_post_processing_input!()`  if you do not want to use the default
input file.
"""
function get_run_info(run_dir::Union{AbstractString,Tuple{AbstractString,Union{Int,Nothing}}}...;
                      itime_min=1, itime_max=0, itime_skip=1, dfns=false, do_setup=true,
                      setup_input_file=nothing)
    if length(run_dir) == 0
        error("No run_dir passed")
    end
    if length(run_dir) > 1
        run_info = Tuple(get_run_info(r; itime_min=itime_min, itime_max=itime_max,
                                      itime_skip=itime_skip, dfns=dfns, do_setup=false)
                         for r ∈ run_dir)
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

    this_run_dir = run_dir[1]
    if isa(this_run_dir, Tuple)
        if length(this_run_dir) != 2
            error("When a Tuple is passed for run_dir, expect it to have length 2. Got "
                  * "$this_run_dir")
        end
        this_run_dir, restart_index = this_run_dir
    else
        restart_index = nothing
    end

    if !isa(this_run_dir, AbstractString) || !isa(restart_index, Union{Int,Nothing})
        error("Expected all `run_dir` arguments to be `String` or `(String, Int)` or "
              * "`(String, Nothing)`. Got $run_dir")
    end

    if !isdir(this_run_dir)
        error("$this_run_dir is not a directory")
    end

    # Normalise by removing any trailing slash - with a slash basename() would return an
    # empty string
    this_run_dir = rstrip(this_run_dir, '/')

    run_name = basename(this_run_dir)
    base_prefix = joinpath(this_run_dir, run_name)
    if restart_index === nothing
        # Find output files from all restarts in the directory
        counter = 1
        run_prefixes = Vector{String}()
        while true
            # Test if output files exist for this value of counter
            prefix_with_count = base_prefix * "_$counter"
            if length(glob(basename(prefix_with_count) * ".*.h5", dirname(prefix_with_count))) > 0 ||
                length(glob(basename(prefix_with_count) * ".*.cdf", dirname(prefix_with_count))) > 0

                push!(run_prefixes, prefix_with_count)
            else
                # No more output files found
                break
            end
            counter += 1
        end
        # Add the final run which does not have a '_$counter' suffix
        push!(run_prefixes, base_prefix)
        run_prefixes = tuple(run_prefixes...)
    elseif restart_index == -1
        run_prefixes = (base_prefix,)
    elseif restart_index > 0
        run_prefixes = (base_prefix * "_$restart_index",)
    else
        error("Invalid restart_index=$restart_index")
    end

    if dfns
        ext = "dfns"
    else
        ext = "moments"
    end

    has_data = all(length(glob(basename(p) * ".$ext*.h5", dirname(p))) > 0
                   for p ∈ run_prefixes)
    if !has_data
        println("No $ext data found for $run_prefixes, skipping $ext")
        return nothing
    end

    fids0 = Tuple(open_readonly_output_file(r, ext, printout=false)
                         for r ∈ run_prefixes)
    nblocks = Tuple(load_block_data(f)[1] for f ∈ fids0)
    if all(n == 1 for n ∈ nblocks)
        # Did not use distributed memory, or used parallel_io
        parallel_io = true
    else
        parallel_io = false
    end

    nt_unskipped, time, restarts_nt = load_time_data(fids0)
    if itime_min <= 0
        itime_min = nt_unskipped + itime_min
    end
    if itime_max <= 0
        itime_max = nt_unskipped + itime_max
    end
    time = time[itime_min:itime_skip:itime_max]
    nt = length(time)

    # Get input and coordinates from the final restart
    file_final_restart = fids0[end]

    input = load_input(file_final_restart)

    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    io_input, evolve_moments, t_input, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
        composition, species, collisions, geometry, drive_input, external_source_settings,
        num_diss_params, manufactured_solns_input = mk_input(input)

    n_ion_species, n_neutral_species = load_species_data(file_final_restart)
    evolve_density, evolve_upar, evolve_ppar = load_mk_options(file_final_restart)

    z_local, z_local_spectral, z_chunk_size =
        load_coordinate_data(file_final_restart, "z")
    r_local, r_local_spectral, r_chunk_size =
        load_coordinate_data(file_final_restart, "r")
    r, r_spectral, z, z_spectral = construct_global_zr_coords(r_local, z_local)

    if dfns
        vperp, vperp_spectral, vperp_chunk_size =
            load_coordinate_data(file_final_restart, "vperp")
        vpa, vpa_spectral, vpa_chunk_size =
            load_coordinate_data(file_final_restart, "vpa")

        if n_neutral_species > 0
            vzeta, vzeta_spectral, vzeta_chunk_size =
                load_coordinate_data(file_final_restart, "vzeta")
            vr, vr_spectral, vr_chunk_size =
                load_coordinate_data(file_final_restart, "vr")
            vz, vz_spectral, vz_chunk_size =
                load_coordinate_data(file_final_restart, "vz")
        else
            dummy_adv_input = advection_input("default", 1.0, 0.0, 0.0)
            dummy_comm = MPI.COMM_NULL
            dummy_input = grid_input("dummy", 1, 1, 1, 1, 0, 1.0,
                                     "chebyshev_pseudospectral", "", "periodic",
                                     dummy_adv_input, dummy_comm, "uniform")
            vzeta, vzeta_spectral = define_coordinate(dummy_input)
            vzeta_chunk_size = 1
            vr, vr_spectral = define_coordinate(dummy_input)
            vr_chunk_size = 1
            vz, vz_spectral = define_coordinate(dummy_input)
            vz_chunk_size = 1
        end
    end

    if parallel_io
        files = fids0
    else
        # Don't keep open files as read_distributed_zr_data!(), etc. open the files
        # themselves
        files = run_prefixes
    end

    if dfns
        run_info = (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                    ext=ext, nblocks=nblocks, files=files, input=input,
                    n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                    evolve_moments=evolve_moments, composition=composition,
                    species=species, collisions=collisions, geometry=geometry,
                    drive_input=drive_input, num_diss_params=num_diss_params,
                    evolve_density=evolve_density, evolve_upar=evolve_upar,
                    evolve_ppar=evolve_ppar,
                    manufactured_solns_input=manufactured_solns_input, nt=nt,
                    nt_unskipped=nt_unskipped, restarts_nt=restarts_nt,
                    itime_min=itime_min, itime_skip=itime_skip, itime_max=itime_max,
                    time=time, r=r, z=z, vperp=vperp, vpa=vpa, vzeta=vzeta, vr=vr, vz=vz,
                    r_local=r_local, z_local=z_local, r_spectral=r_spectral,
                    z_spectral=z_spectral, vperp_spectral=vperp_spectral,
                    vpa_spectral=vpa_spectral, vzeta_spectral=vzeta_spectral,
                    vr_spectral=vr_spectral, vz_spectral=vz_spectral,
                    r_chunk_size=r_chunk_size, z_chunk_size=z_chunk_size,
                    vperp_chunk_size=vperp_chunk_size, vpa_chunk_size=vpa_chunk_size,
                    vzeta_chunk_size=vzeta_chunk_size, vr_chunk_size=vr_chunk_size,
                    vz_chunk_size=vz_chunk_size, dfns=dfns)
        if do_setup
            setup_makie_post_processing_input!(
                setup_input_file; run_info_dfns=run_info,
                allow_missing_input_file=(setup_input_file === nothing))
        end
    else
        run_info = (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                    ext=ext, nblocks=nblocks, files=files, input=input,
                    n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                    evolve_moments=evolve_moments, composition=composition,
                    species=species, collisions=collisions, geometry=geometry,
                    drive_input=drive_input, num_diss_params=num_diss_params,
                    evolve_density=evolve_density, evolve_upar=evolve_upar,
                    evolve_ppar=evolve_ppar,
                    manufactured_solns_input=manufactured_solns_input, nt=nt,
                    nt_unskipped=nt_unskipped, restarts_nt=restarts_nt,
                    itime_min=itime_min, itime_skip=itime_skip, itime_max=itime_max,
                    time=time, r=r, z=z, r_local=r_local, z_local=z_local,
                    r_spectral=r_spectral, z_spectral=z_spectral,
                    r_chunk_size=r_chunk_size, z_chunk_size=z_chunk_size, dfns=dfns)
        if do_setup
            setup_makie_post_processing_input!(
                setup_input_file; run_info_moments=run_info,
                allow_missing_input_file=(setup_input_file === nothing))
        end
    end

    return run_info
end

"""
    close_run_info(run_info)

Close all the files in a run_info NamedTuple.
"""
function close_run_info(run_info)
    if run_info === nothing
        return nothing
    end
    if !run_info.parallel_io
        # Files are not kept open, so nothing to do
        return nothing
    end

    for f ∈ run_info.files
        close(f)
    end

    return nothing
end

"""
    postproc_load_variable(run_info, variable_name; it=nothing, is=nothing,
                           ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                           ivzeta=nothing, ivr=nothing, ivz=nothing)

Load a variable

`run_info` is the information about a run returned by [`get_run_info`](@ref).

`variable_name` is the name of the variable to load.

The keyword arguments `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz`
can be set to an integer or a range (e.g. `3:8` or `3:2:8`) to select subsets of the data.
Only the data for the subset requested will be loaded from the output file (mostly - when
loading fields or moments from runs which used `parallel_io = false`, the full array will
be loaded and then sliced).
"""
function postproc_load_variable(run_info, variable_name; it=nothing, is=nothing,
                                ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                                ivzeta=nothing, ivr=nothing, ivz=nothing)
    nt = run_info.nt

    if it === nothing
        it = run_info.itime_min:run_info.itime_skip:run_info.itime_max
    elseif isa(it, mk_int)
        nt = 1
        it = collect(run_info.itime_min:run_info.itime_skip:run_info.itime_max)[it]
    else
        nt = length(it)
    end
    if is === nothing
        # Can't use 'n_species' in a similar way to the way we treat other dims, because
        # we don't know here if the variable is for ions or neutrals.
        # Use Colon operator `:` when slice argument is `nothing` as when we pass that as
        # an 'index', it selects the whole dimension. Brackets are needed around the `:`
        # when assigning it to variables, etc. to avoid an error "LoadError: syntax:
        # newline not allowed after ":" used for quoting".
        is = (:)
    elseif isa(is, mk_int)
        nspecies = 1
    else
        nspecies = length(is)
    end
    if ir === nothing
        nr = run_info.r.n
        ir = 1:nr
    elseif isa(ir, mk_int)
        nr = 1
    else
        nr = length(ir)
    end
    if iz === nothing
        nz = run_info.z.n
        iz = 1:nz
    elseif isa(iz, mk_int)
        nz = 1
    else
        nz = length(iz)
    end
    if ivperp === nothing
        if :vperp ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvperp = run_info.vperp.n
            ivperp = 1:nvperp
        else
            nvperp = nothing
            ivperp = nothing
        end
    elseif isa(ivperp, mk_int)
        nvperp = 1
    else
        nvperp = length(ivperp)
    end
    if ivpa === nothing
        if :vpa ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvpa = run_info.vpa.n
            ivpa = 1:nvpa
        else
            nvpa = nothing
            ivpa = nothing
        end
    elseif isa(ivpa, mk_int)
        nvpa = 1
    else
        nvpa = length(ivpa)
    end
    if ivzeta === nothing
        if :vzeta ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvzeta = run_info.vzeta.n
            ivzeta = 1:nvzeta
        else
            nvzeta = nothing
            ivzeta = nothing
        end
    elseif isa(ivzeta, mk_int)
        nvzeta = 1
    else
        nvzeta = length(ivzeta)
    end
    if ivr === nothing
        if :vr ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvr = run_info.vr.n
            ivr = 1:nvr
        else
            nvr = nothing
            ivr = nothing
        end
    elseif isa(ivr, mk_int)
        nvr = 1
    else
        nvr = length(ivr)
    end
    if ivz === nothing
        if :vz ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvz = run_info.vz.n
            ivz = 1:nvz
        else
            nvz = nothing
            ivz = nothing
        end
    elseif isa(ivz, mk_int)
        nvz = 1
    else
        nvz = length(ivz)
    end

    if run_info.parallel_io
        # Get HDF5/NetCDF variables directly and load slices
        variable = Tuple(get_group(f, "dynamic_data")[variable_name]
                         for f ∈ run_info.files)
        nd = ndims(variable[1])

        if nd == 3
            # EM variable with dimensions (z,r,t)
            dims = Vector{mk_int}()
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 4
            # moment variable with dimensions (z,r,s,t)
            # Get nspecies from the variable, not from run_info, because it might be
            # either ion or neutral
            dims = Vector{mk_int}()
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            if is === (:)
                nspecies = size(variable[1], 3)
                push!(dims, nspecies)
            elseif !isa(is, mk_int)
                push!(dims, nspecies)
            end
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 6
            # ion distribution function variable with dimensions (vpa,vperp,z,r,s,t)
            nspecies = size(variable[1], 5)
            dims = Vector{mk_int}()
            !isa(ivpa, mk_int) && push!(dims, nvpa)
            !isa(ivperp, mk_int) && push!(dims, nvperp)
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            if is === (:)
                nspecies = size(variable[1], 3)
                push!(dims, nspecies)
            elseif !isa(is, mk_int)
                push!(dims, nspecies)
            end
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 7
            # neutral distribution function variable with dimensions (vz,vr,vzeta,z,r,s,t)
            nspecies = size(variable[1], 6)
            dims = Vector{mk_int}()
            !isa(ivz, mk_int) && push!(dims, nvz)
            !isa(ivr, mk_int) && push!(dims, nvr)
            !isa(ivzeta, mk_int) && push!(dims, nvzeta)
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            if is === (:)
                nspecies = size(variable[1], 3)
                push!(dims, nspecies)
            elseif !isa(is, mk_int)
                push!(dims, nspecies)
            end
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        else
            error("Unsupported number of dimensions ($nd) for '$variable_name'.")
        end

        local_it_start = 1
        global_it_start = 1
        for v ∈ variable
            # For restarts, the first time point is a duplicate of the last time
            # point of the previous restart. Use `offset` to skip this point.
            offset = local_it_start == 1 ? 0 : 1
            local_nt = size(v, nd) - offset
            local_it_end = local_it_start+local_nt-1

            if isa(it, mk_int)
                tind = it - local_it_start + 1
                if tind < 1
                    error("Trying to select time index before the beginning of this "
                          * "restart, should have finished already")
                elseif tind <= local_nt
                    # tind is within this restart's time range, so get result
                    if nd == 3
                        result .= v[iz,ir,tind]
                    elseif nd == 4
                        result .= v[iz,ir,is,tind]
                    elseif nd == 6
                        result .= v[ivpa,ivperp,iz,ir,is,tind]
                    elseif nd == 7
                        result .= v[ivz,ivr,ivzeta,iz,ir,is,tind]
                    else
                        error("Unsupported combination nd=$nd, ir=$ir, iz=$iz, ivperp=$ivperp "
                              * "ivpa=$ivpa, ivzeta=$ivzeta, ivr=$ivr, ivz=$ivz.")
                    end

                    # Already got the data for `it`, so end loop
                    break
                end
            else
                tinds = collect(i - local_it_start + 1 + offset for i ∈ it
                                if local_it_start <= i <= local_it_end)
                # Convert tinds to slice, as we know the spacing is constant
                if length(tinds) != 0
                    # There is some data in this file
                    if length(tinds) > 1
                        tstep = tinds[2] - tinds[begin]
                    else
                        tstep = 1
                    end
                    tinds = tinds[begin]:tstep:tinds[end]
                    global_it_end = global_it_start + length(tinds) - 1

                    if nd == 3
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[iz,ir,tinds]
                    elseif nd == 4
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[iz,ir,is,tinds]
                    elseif nd == 6
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[ivpa,ivperp,iz,ir,is,tinds]
                    elseif nd == 7
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[ivz,ivr,ivzeta,iz,ir,is,tinds]
                    else
                        error("Unsupported combination nd=$nd, ir=$ir, iz=$iz, ivperp=$ivperp "
                              * "ivpa=$ivpa, ivzeta=$ivzeta, ivr=$ivr, ivz=$ivz.")
                    end

                    global_it_start = global_it_end + 1
                end
            end

            local_it_start = local_it_end + 1
        end
    else
        # Use existing distributed I/O loading functions
        if variable_name ∈ em_variables
            nd = 3
        elseif variable_name ∈ ion_dfn_variables
            nd = 6
        elseif variable_name ∈ neutral_dfn_variables
            nd = 7
        else
            # Ion or neutral moment variable
            nd = 4
        end

        if nd == 3
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.nt)
            read_distributed_zr_data!(result, variable_name, run_info.files,
                                      run_info.ext, run_info.nblocks, run_info.z_local.n,
                                      run_info.r_local.n, run_info.itime_skip)
            result = result[iz,ir,it]
        elseif nd == 4
            # If we ever have neutrals included but n_neutral_species != n_ion_species,
            # then this will fail - in that case would need some way to specify that we
            # need to read a neutral moment variable rather than an ion moment variable
            # here.
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.n_ion_species,
                                    run_info.nt)
            read_distributed_zr_data!(result, variable_name, run_info.files,
                                      run_info.ext, run_info.nblocks, run_info.z_local.n,
                                      run_info.r_local.n, run_info.itime_skip)
            result = result[iz,ir,is,it]
        elseif nd === 6
            result = load_distributed_charged_pdf_slice(run_info.files, run_info.nblocks,
                                                        it, run_info.n_ion_species,
                                                        run_info.r_local,
                                                        run_info.z_local, run_info.vperp,
                                                        run_info.vpa;
                                                        is=(is === (:) ? nothing : is),
                                                        ir=ir, iz=iz, ivperp=ivperp,
                                                        ivpa=ivpa)
        elseif nd === 7
            result = load_distributed_neutral_pdf_slice(run_info.files, run_info.nblocks,
                                                        it, run_info.n_ion_species,
                                                        run_info.r_local,
                                                        run_info.z_local, run_info.vzeta,
                                                        run_info.vr, run_info.vz;
                                                        isn=(is === (:) ? nothing : is),
                                                        ir=ir, iz=iz, ivzeta=ivzeta,
                                                        ivr=ivr, ivz=ivz)
        end
    end

    return result
end

"""
    get_variable(run_info::Tuple, variable_name; kwargs...)
    get_variable(run_info, variable_name; kwargs...)

Get an array (or Tuple of arrays, if `run_info` is a Tuple) of the data for
`variable_name` from `run_info`.

Some derived variables need to be calculated from the saved output, not just loaded from
file (with `postproc_load_variable`). This function takes care of that calculation, and
handles the case where `run_info` is a Tuple (which `postproc_load_data` does not handle).

`kwargs...` are passed through to `postproc_load_variable()`.
"""
function get_variable end

function get_variable(run_info::Tuple, variable_name; kwargs...)
    return Tuple(get_variable(ri, variable_name; kwargs...) for ri ∈ run_info)
end

function get_variable(run_info, variable_name; kwargs...)
    if variable_name == "temperature"
        vth = postproc_load_variable(run_info, "thermal_speed")
        variable = vth.^2
    elseif variable_name == "collision_frequency"
        n = postproc_load_variable(run_info, "density")
        vth = postproc_load_variable(run_info, "thermal_speed")
        variable = get_collision_frequency(run_info.collisions, n, vth)
    elseif variable_name == "temperature_neutral"
        vth = postproc_load_variable(run_info, "thermal_speed_neutral")
        variable = vth.^2
    elseif variable_name == "sound_speed"
        T_e = run_info.composition.T_e
        T_i = get_variable(run_info, "temperature"; kwargs...)

        # Adiabatic index. Not too clear what value should be (see e.g. [Riemann 1991,
        # below eq. (39)], or discussion of Bohm criterion in Stangeby's book.
        gamma = 3.0

        # Factor of 0.5 needed because temperatures are normalised to mi*cref^2, not Tref
        variable = @. sqrt(0.5*(T_e + gamma*T_i))
    elseif variable_name == "mach_number"
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        cs = get_variable(run_info, "sound_speed"; kwargs...)
        variable = upar ./ cs
    else
        variable = postproc_load_variable(run_info, variable_name)
    end

    return variable
end

const chunk_size_1d = 10000
const chunk_size_2d = 100
struct VariableCache{T1,T2,N}
    run_info::T1
    variable_name::String
    t_chunk_size::mk_int
    n_tinds::mk_int
    tinds_range_global::Union{UnitRange{mk_int},StepRange{mk_int}}
    tinds_chunk::Union{Base.RefValue{UnitRange{mk_int}},Base.RefValue{StepRange{mk_int}}}
    data_chunk::Array{mk_float,N}
    dim_slices::T2
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
    data_chunk = postproc_load_variable(run_info, variable_name;
                                        it=tinds_range_global[tinds_chunk], dim_slices...)

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
            postproc_load_variable(variable_cache.run_info, variable_cache.variable_name;
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
    plots_for_variable(run_info, variable_name; plot_prefix, is_1D=false,
                       is_1V=false, steady_state_residual_fig_axes=nothing)

Make plots for the EM field or moment variable `variable_name`.

Which plots to make are determined by the settings in the section of the input whose
heading is the variable name.

`run_info` is the information returned by [`get_run_info`](@ref).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`is_1D` and/or `is_1V` can be passed to allow the function to skip some plots that do not
make sense for 1D or 1V simulations (regardless of the settings).

`steady_state_residual_fig_axes` contains the figure, axes and legend places for steady
state residual plots.
"""
function plots_for_variable(run_info, variable_name; plot_prefix, is_1D=false,
                            is_1V=false, steady_state_residual_fig_axes=nothing)
    input = Dict_to_NamedTuple(input_dict[variable_name])

    # test if any plot is needed
    if !(any(v for (k,v) in pairs(input) if
           startswith(String(k), "plot") || startswith(String(k), "animate") ||
           k == :steady_state_residual))
        return nothing
    end

    if is_1D && variable_name == "Er"
        return nothing
    elseif variable_name == "collision_frequency" &&
            all(ri.collisions.krook_collisions_option == "none" for ri ∈ run_info)
        # No Krook collisions active, so do not make plots.
        return nothing
    end

    println("Making plots for $variable_name")
    flush(stdout)

    variable = get_variable(run_info, variable_name)

    if variable_name ∈ em_variables
        species_indices = (nothing,)
    elseif variable_name ∈ neutral_moment_variables ||
           variable_name ∈ neutral_dfn_variables
        species_indices = 1:maximum(ri.n_neutral_species for ri ∈ run_info)
    else
        species_indices = 1:maximum(ri.n_ion_species for ri ∈ run_info)
    end
    for is ∈ species_indices
        if is !== nothing
            variable_prefix = plot_prefix * variable_name * "_spec$(is)_"
            log_variable_prefix = plot_prefix * "log" * variable_name * "_spec$(is)_"
        else
            variable_prefix = plot_prefix * variable_name * "_"
            log_variable_prefix = plot_prefix * "log" * variable_name * "_"
        end
        if variable_name == "Er" && is_1D
            # Skip if there is no r-dimension
            continue
        end
        if !is_1D && input.plot_vs_r_t
            plot_vs_r_t(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_r_t.pdf")
        end
        if input.plot_vs_z_t
            plot_vs_z_t(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_t.pdf")
        end
        if !is_1D && input.plot_vs_r
            plot_vs_r(run_info, variable_name, is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_r.pdf")
        end
        if input.plot_vs_z
            plot_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_z.pdf")
        end
        if !is_1D && input.plot_vs_z_r
            plot_vs_z_r(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_r.pdf")
        end
        if input.animate_vs_z
            animate_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_z." * input.animation_ext)
        end
        if !is_1D && input.animate_vs_r
            animate_vs_r(run_info, variable_name, is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_r." * input.animation_ext)
        end
        if !is_1D && input.animate_vs_z_r
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
    plots_for_dfn_variable(run_info, variable_name; plot_prefix, is_1D=false,
                           is_1V=false)

Make plots for the distribution function variable `variable_name`.

Which plots to make are determined by the settings in the section of the input whose
heading is the variable name.

`run_info` is the information returned by [`get_run_info()`](@ref). The `dfns=true` keyword
argument must have been passed to [`get_run_info()`](@ref) so that output files containing
the distribution functions are being read.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`is_1D` and/or `is_1V` can be passed to allow the function to skip some plots that do not
make sense for 1D or 1V simulations (regardless of the settings).
"""
function plots_for_dfn_variable(run_info, variable_name; plot_prefix, is_1D=false,
                                is_1V=false)
    input = Dict_to_NamedTuple(input_dict_dfns[variable_name])

    is_neutral = variable_name ∈ neutral_dfn_variables

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
    if is_1D
        animate_dims = setdiff(animate_dims, (:r,))
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
                    if input[Symbol(:plot, log, :_unnorm_vs_vz_z)]
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
                    if input[Symbol(:animate, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true, is=is,
                                                  outfile=outfile, colorscale=yscale,
                                                  transform=transform)
                    end
                else
                    if input[Symbol(:plot, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, is=is, outfile=outfile,
                                             yscale=yscale, transform=transform)
                    end
                    if input[Symbol(:plot, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input, is=is, outfile=outfile,
                                               colorscale=yscale, transform=transform)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input, is=is, outfile=outfile,
                                                yscale=yscale, transform=transform)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input, is=is, outfile=outfile,
                                                  colorscale=yscale, transform=transform)
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
        fn = postproc_load_variable(run_info, "f_neutral")
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
        f = postproc_load_variable(run_info, "f")
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
    spaces = " " ^ length(function_name_str)
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
             function $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                      $($spaces)input=nothing, outfile=nothing, yscale=nothing,
                      transform=identity, axis_args=Dict{Symbol,Any}(), it=nothing,
                      $($spaces)ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                      $($spaces)ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
             function $($function_name_str)(run_info, var_name; is=1, data=nothing,
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
                     println("$($function_name_str) failed for $var_name, is=$is. Error was $e")
                     return nothing
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
                         input = input_dict_dfns[var_name]
                     else
                         input = input_dict[var_name]
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
                     data = postproc_load_variable(run_info, var_name; dim_slices...)
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
    spaces = " " ^ length(function_name_str)
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
             function $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                      $($spaces)input=nothing, outfile=nothing, colorscale=identity,
                      $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                      $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                      $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                      $($spaces)kwargs...)
             function $($function_name_str)(run_info, var_name; is=1, data=nothing,
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
                     println("$($function_name_str) failed for $var_name, is=$is. Error was $e")
                     return nothing
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
                         input = input_dict_dfns[var_name]
                     else
                         input = input_dict[var_name]
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
                     data = postproc_load_variable(run_info, var_name; dim_slices...)
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
    spaces = " " ^ length(function_name_str)
    dim_str = String(dim)
    dim_grid = :( run_info.$dim.grid )
    idim = Symbol(:i, dim)
    eval(quote
             export $function_name

             """
             function $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                      $($spaces)input=nothing, outfile=nothing, yscale=nothing,
                      $($spaces)transform=identity, ylims=nothing,
                      $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing, iz=nothing,
                      $($spaces)ivperp=nothing, ivpa=nothing, ivzeta=nothing, ivr=nothing,
                      $($spaces)ivz=nothing, kwargs...)
             function $($function_name_str)(run_info, var_name; is=1, data=nothing,
                      $($spaces)input=nothing, frame_index=nothing, ax=nothing,
                      $($spaces)fig=nothing, outfile=nothing, yscale=nothing,
                      $($spaces)transform=identity, ylims=nothing,
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
                         title = lift(i->string("t = ", run_info[1].time[i]), frame_index)
                     else
                         title = lift(i->join((string("t", irun, " = ", ri.time[i])
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
                     println("$($function_name_str)() failed for $var_name, is=$is. Error was $e")
                     return nothing
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     fig=nothing, outfile=nothing, yscale=nothing,
                                     ylims=nothing, axis_args=Dict{Symbol,Any}(),
                                     it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         input = input_dict_dfns[var_name]
                     else
                         input = input_dict[var_name]
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
                     title = lift(i->string("t = ", run_info.time[i]), ind)
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale, title=title, axis_args...)
                 else
                     fig = nothing
                 end

                 x = $dim_grid
                 if $idim !== nothing
                     x = x[$idim]
                 end
                 animate_1d(x, data; ax=ax, ylims=ylims, frame_index=ind,
                            label=run_info.run_name, kwargs...)

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
    spaces = " " ^ length(function_name_str)
    dim1_str = String(dim1)
    dim2_str = String(dim2)
    dim1_grid = :( run_info.$dim1.grid )
    dim2_grid = :( run_info.$dim2.grid )
    idim1 = Symbol(:i, dim1)
    idim2 = Symbol(:i, dim2)
    eval(quote
             export $function_name

             """
             function $($function_name_str)(run_info::Tuple, var_name; is=1, data=nothing,
                      $($spaces)input=nothing, outfile=nothing, colorscale=identity,
                      $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                      $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                      $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                      $($spaces)kwargs...)
             function $($function_name_str)(run_info, var_name; is=1, data=nothing,
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
                         subtitles = (lift(i->string(ri.run_name, "\nt = ", ri.time[i]),
                                           frame_index)
                                      for ri ∈ run_info)
                     else
                         title = lift(i->string(get_variable_symbol(var_name), "\nt = ",
                                                run_info[1].time[i]),
                                      frame_index)
                         subtitles = nothing
                     end
                     fig, ax, colorbar_places = get_2d_ax(length(run_info),
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
                     println("$($function_name_str) failed for $var_name, is=$is. Error was $e")
                     return nothing
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
                         input = input_dict_dfns[var_name]
                     else
                         input = input_dict[var_name]
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
              get_legend_place=nothing, kwargs...)

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

`resolution` is passed through to the `Figure` constructor. Its default value is
`(600, 400)` if `n` is not passed, or `(600*n, 400)` if `n` is passed.

Extra `kwargs` are passed to the `Axis()` constructor.
"""
function get_1d_ax(n=nothing; title=nothing, subtitles=nothing, yscale=nothing,
                   get_legend_place=nothing, resolution=(600, 400), kwargs...)
    valid_legend_places = (nothing, :left, :right, :above, :below)
    if get_legend_place ∉ valid_legend_places
        error("get_legend_place=$get_legend_place is not one of $valid_legend_places")
    end
    if yscale !== nothing
        kwargs = tuple(kwargs..., :yscale=>yscale)
    end
    if n == nothing
        if resolution == nothing
            resolution = (600, 400)
        end
        fig = Figure(resolution=resolution)
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
        if resolution == nothing
            resolution = (600*n, 400)
        end
        fig = Figure(resolution=resolution)
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
    get_2d_ax(n=nothing; title=nothing, subtitles=nothing, kwargs...)

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

`resolution` is passed through to the `Figure` constructor. Its default value is
`(600, 400)` if `n` is not passed, or `(600*n, 400)` if `n` is passed.

Extra `kwargs` are passed to the `Axis()` constructor.
"""
function get_2d_ax(n=nothing; title=nothing, subtitles=nothing, resolution=nothing,
                   kwargs...)
    if n == nothing
        if resolution == nothing
            resolution = (600, 400)
        end
        fig = Figure(resolution=resolution)
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
        if resolution == nothing
            resolution = (600*n, 400)
        end
        fig = Figure(resolution=resolution)

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
    if yscale !== nothing
        ax.yscale = yscale
    end

    if transform !== identity
        # Use transform to allow user to do something like data = abs.(data)
        # Don't actually apply identity transform in case this function is called with
        # `data` being a Makie Observable (in which case transform.(data) would be an
        # error).
        data = transform.(data)
    end

    l = lines!(ax, xcoord, data; kwargs...)

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

    return mesh!(ax, vertices, faces; color = colors, shading = false, kwargs...)
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

function select_slice(variable::AbstractArray{T,6}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivperp=nothing,
                      ivpa=nothing, kwargs...) where T
    # Array is (z,r,species,t)

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
    if ivpa !== nothing
        ivpa0 = ivpa
    elseif input === nothing || :ivpa0 ∉ input
        ivpa0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivpa0 = input.ivpa0
    end
    if ivperp !== nothing
        ivperp0 = ivperp
    elseif input === nothing || :ivperp0 ∉ input
        ivperp0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivperp0 = input.ivperp0
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
    # Array is (z,r,species,t)

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
    for (key, fa) ∈ fig_axes
        for (ax, lp) ∈ zip(fa[2], fa[3])
            Legend(lp, ax)
        end
        save(plot_prefix * replace(key, " "=>"_") * ".pdf", fa[1])
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
end

function calculate_steady_state_residual(run_info, variable_name; is=1, data=nothing,
                                         plot_prefix=nothing, fig_axes=nothing,
                                         i_run=1)

    if data === nothing
        data = postproc_load_variable(run_info, variable_name; is=is)
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
    plot_f_unnorm_vs_vpa(run_info; input=nothing, neutral=false, it=nothing, is=1,
                         iz=nothing, fig=nothing, ax=nothing, outfile=nothing,
                         yscale=identity, transform=identity,
                         axis_args=Dict{Symbol,Any}(), kwargs...)

Plot an unnormalized distribution function against \$v_\\parallel\$ at a fixed z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are overlayed on the same axis.

By default plots the ion distribution function. If `neutrals=true` is passed, plots the
neutral distribution function instead.

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

function plot_f_unnorm_vs_vpa(run_info::Tuple; f_over_vpa2=false, neutral=false,
                              outfile=nothing, axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        species_label = neutral ? "n" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, axis_args...)

        for ri ∈ run_info
            plot_f_unnorm_vs_vpa(ri; f_over_vpa2=f_over_vpa2, neutral=neutral, ax=ax,
                                 kwargs...)
        end

        if n_runs > 1
            put_legend_above(fig, ax)
        end

        if outfile !== nothing
            save(outfile, fig)
        end

        return fig
    catch e
        println("Error in plot_f_unnorm_vs_vpa(). Error was ", e)
    end
end

function plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=false, input=nothing, neutral=false,
                              it=nothing, is=1, iz=nothing, fig=nothing, ax=nothing,
                              outfile=nothing, transform=identity,
                              axis_args=Dict{Symbol,Any}(), kwargs...)
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
        species_label = neutral ? "n" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, axis_args...)
    end

    if neutral
        f = postproc_load_variable(run_info, "f_neutral"; it=it, is=is, ir=input.ir0,
                                   iz=iz, ivzeta=input.ivzeta0, ivr=input.ivr0)
        density = postproc_load_variable(run_info, "density_neutral"; it=it, is=is,
                                         ir=input.ir0, iz=iz)
        upar = postproc_load_variable(run_info, "uz_neutral"; it=it, is=is, ir=input.ir0,
                                      iz=iz)
        vth = postproc_load_variable(run_info, "thermal_speed_neutral"; it=it, is=is,
                                     ir=input.ir0, iz=iz)
    else
        f = postproc_load_variable(run_info, "f"; it=it, is=is, ir=input.ir0, iz=iz,
                                   ivperp=input.ivperp0)
        density = postproc_load_variable(run_info, "density"; it=it, is=is, ir=input.ir0,
                                         iz=iz)
        upar = postproc_load_variable(run_info, "parallel_flow"; it=it, is=is, ir=input.ir0, iz=iz)
        vth = postproc_load_variable(run_info, "thermal_speed"; it=it, is=is,
                                     ir=input.ir0, iz=iz)
    end

    f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(f, run_info.vpa.grid, density, upar, vth,
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
    plot_f_unnorm_vs_vpa_z(run_info; input=nothing, neutral=false, it=nothing, is=1,
                           fig=nothing, ax=nothing, outfile=nothing, yscale=identity,
                           transform=identity, rasterize=true, subtitles=nothing,
                           axis_args=Dict{Symbol,Any}(), kwargs...)

Plot unnormalized distribution function against \$v_\\parallel\$ and z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

By default plots the ion distribution function. If `neutrals=true` is passed, plots the
neutral distribution function instead.

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

function plot_f_unnorm_vs_vpa_z(run_info::Tuple; neutral=false, outfile=nothing,
                                axis_args=Dict{Symbol,Any}(), title=nothing,
                                subtitles=nothing, kwargs...)
    try
        n_runs = length(run_info)
        if subtitles === nothing
            subtitles = Tuple(nothing for _ ∈ 1:n_runs)
        end
        if title !== nothing
            title = neutral ? L"f_{n,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        end
        fig, axes, colorbar_places =
            get_2d_ax(n_runs; title=title, xlabel=L"v_\parallel", ylabel=L"z",
                      axis_args...)

        for (ri, ax, colorbar_place, st) ∈ zip(run_info, axes, colorbar_places, subtitles)
            plot_f_unnorm_vs_vpa_z(ri; neutral=neutral, ax=ax, colorbar_place=colorbar_place,
                                   title=st, kwargs...)
        end

        if outfile !== nothing
            save(outfile, fig)
        end

        return fig
    catch e
        println("Error in plot_f_unnorm_vs_vpa_z(). Error was ", e)
    end
end

function plot_f_unnorm_vs_vpa_z(run_info; input=nothing, neutral=false, it=nothing, is=1,
                                fig=nothing, ax=nothing, colorbar_place=nothing, title=nothing,
                                outfile=nothing, transform=identity, rasterize=true,
                                axis_args=Dict{Symbol,Any}(), kwargs...)
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
            title = neutral ? L"f_{n,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
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
        f = postproc_load_variable(run_info, "f_neutral"; it=it, is=is, ir=input.ir0,
                                   ivzeta=input.ivzeta0, ivr=input.ivr0)
        density = postproc_load_variable(run_info, "density_neutral"; it=it, is=is,
                                         ir=input.ir0)
        upar = postproc_load_variable(run_info, "uz_neutral"; it=it, is=is, ir=input.ir0)
        vth = postproc_load_variable(run_info, "thermal_speed_neutral"; it=it, is=is,
                                     ir=input.ir0)
        vpa_grid = run_info.vz.grid
    else
        f = postproc_load_variable(run_info, "f"; it=it, is=is, ir=input.ir0,
                                   ivperp=input.ivperp0)
        density = postproc_load_variable(run_info, "density"; it=it, is=is, ir=input.ir0)
        upar = postproc_load_variable(run_info, "parallel_flow"; it=it, is=is, ir=input.ir0)
        vth = postproc_load_variable(run_info, "thermal_speed"; it=it, is=is,
                                     ir=input.ir0)
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
    animate_f_unnorm_vs_vpa(run_info; input=nothing, neutral=false, is=1, iz=nothing,
                            fig=nothing, ax=nothing, frame_index=nothing,
                            outfile=nothing, yscale=identity, transform=identity,
                            axis_args=Dict{Symbol,Any}(), kwargs...)

Plot an unnormalized distribution function against \$v_\\parallel\$ at a fixed z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to animate is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are overlayed on the same axis.

By default animates the ion distribution function. If `neutrals=true` is passed, animates
the neutral distribution function instead.

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

Any extra `kwargs` are passed to [`plot_1d`](@ref) (which is used to create the plot, as
we have to handle time-varying coordinates so cannot use [`animate_1d`](@ref)).
"""
function animate_f_unnorm_vs_vpa end

function animate_f_unnorm_vs_vpa(run_info::Tuple; f_over_vpa2=false, neutral=false,
                                 outfile=nothing, axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        frame_index = Observable(1)

        species_label = neutral ? "n" : "i"
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
            animate_f_unnorm_vs_vpa(ri; f_over_vpa2=f_over_vpa2, neutral=neutral, ax=ax,
                                    frame_index=frame_index, kwargs...)
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
        println("Error in animate_f_unnorm_vs_vpa(). Error was ", e)
    end
end

function animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=false, input=nothing,
                                 neutral=false, is=1, iz=nothing, fig=nothing, ax=nothing,
                                 frame_index=nothing, outfile=nothing, transform=identity,
                                 axis_args=Dict{Symbol,Any}(), kwargs...)
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
        density = postproc_load_variable(run_info, "density_neutral"; is=is, ir=input.ir0,
                                         iz=iz)
        upar = postproc_load_variable(run_info, "uz_neutral"; is=is, ir=input.ir0, iz=iz)
        vth = postproc_load_variable(run_info, "thermal_speed_neutral"; is=is,
                                     ir=input.ir0, iz=iz)
    else
        f = VariableCache(run_info, "f", chunk_size_2d; it=nothing, is=is, ir=input.ir0, iz=iz,
                          ivperp=input.ivperp0, ivpa=nothing, ivzeta=nothing, ivr=nothing,
                          ivz=nothing)
        density = postproc_load_variable(run_info, "density"; is=is, ir=input.ir0, iz=iz)
        upar = postproc_load_variable(run_info, "parallel_flow"; is=is, ir=input.ir0, iz=iz)
        vth = postproc_load_variable(run_info, "thermal_speed"; is=is, ir=input.ir0, iz=iz)
    end

    function get_this_f_unnorm(it)
        f_unnorm = get_unnormalised_f_1d(get_cache_slice(f, it), density[it], vth[it],
                                         run_info.evolve_density, run_info.evolve_ppar)

        if f_over_vpa2
            dzdt = vpagrid_to_dzdt(run_info.vpa.grid, vth[it], upar[it],
                                   run_info.evolve_ppar, run_info.evolve_upar)
            dzdt2 = dzdt.^2
            for i ∈ eachindex(dzdt2)
                if dzdt2[i] == 0.0
                    dzdt2[i] = 1.0
                end
            end

            f_unnorm = @. copy(f_unnorm) / dzdt2
        end

        return f_unnorm
    end

    # Get extrema of dzdt
    dzdtmin = Inf
    dzdtmax = -Inf
    fmin = Inf
    fmax = -Inf
    for it ∈ 1:run_info.nt
        this_dzdt = vpagrid_to_dzdt(run_info.vpa.grid, vth[it], upar[it],
                                    run_info.evolve_ppar, run_info.evolve_upar)
        this_dzdtmin, this_dzdtmax = extrema(this_dzdt)
        dzdtmin = min(dzdtmin, this_dzdtmin)
        dzdtmax = max(dzdtmax, this_dzdtmax)

        this_f_unnorm = get_this_f_unnorm(it)

        this_fmin, this_fmax = NaNMath.extrema(transform(this_f_unnorm))
        fmin = min(fmin, this_fmin)
        fmax = max(fmax, this_fmax)
    end
    yheight = fmax - fmin
    xwidth = dzdtmax - dzdtmin
    limits!(ax, dzdtmin - 0.01*xwidth, dzdtmax + 0.01*xwidth,
            fmin - 0.01*yheight, fmax + 0.01*yheight)

    dzdt = @lift vpagrid_to_dzdt(run_info.vpa.grid, vth[$frame_index], upar[$frame_index],
                                 run_info.evolve_ppar, run_info.evolve_upar)
    f_unnorm = @lift transform.(get_this_f_unnorm($frame_index))

    l = plot_1d(dzdt, f_unnorm; ax=ax, label=run_info.run_name, kwargs...)

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
    animate_f_unnorm_vs_vpa_z(run_info; input=nothing, neutral=false, is=1,
                              fig=nothing, ax=nothing, frame_index=nothing,
                              outfile=nothing, yscale=identity, transform=identity,
                              axis_args=Dict{Symbol,Any}(), kwargs...)

Animate an unnormalized distribution function against \$v_\\parallel\$ and z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

By default animates the ion distribution function. If `neutrals=true` is passed, animates
the neutral distribution function instead.

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

function animate_f_unnorm_vs_vpa_z(run_info::Tuple; neutral=false, outfile=nothing,
                                   axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        frame_index = Observable(1)

        var_name = neutral ? L"f_{n,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
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
            animate_f_unnorm_vs_vpa_z(ri; neutral=neutral, ax=ax,
                                      colorbar_place=colorbar_place, frame_index=frame_index,
                                      kwargs...)
        end

        if outfile !== nothing
            nt = minimum(ri.nt for ri ∈ run_info)
            save_animation(fig, frame_index, nt, outfile)
        end

        return fig
    catch e
        println("Error in animate_f_unnorm_vs_vpa_z(). Error was ", e)
    end
end

function animate_f_unnorm_vs_vpa_z(run_info; input=nothing, neutral=false, is=1,
                                   fig=nothing, ax=nothing, colorbar_place=nothing,
                                   frame_index=nothing, outfile=nothing,
                                   transform=identity, axis_args=Dict{Symbol,Any}(),
                                   kwargs...)
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
        f = VariableCache(run_info, "f", chunk_size_2d; it=nothing, is=is, ir=input.ir0,
                          iz=nothing, ivperp=input.ivperp0, ivpa=nothing, ivzeta=nothing,
                          ivr=nothing, ivz=nothing)
        density = VariableCache(run_info, "density", chunk_size_1d; it=nothing, is=is,
                                ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                                ivzeta=nothing, ivr=nothing, ivz=nothing)
        upar = VariableCache(run_info, "parallel_flow", chunk_size_1d; it=nothing, is=is,
                             ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                             ivzeta=nothing, ivr=nothing, ivz=nothing)
        vth = VariableCache(run_info, "thermal_speed", chunk_size_1d; it=nothing, is=is,
                            ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
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
    plot_charged_pdf_2D_at_wall(run_info; plot_prefix)

Make plots/animations of the charged particle distribution function at wall boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Tuple, comparison plots are made where line
plots/animations from the different runs are overlayed on the same axis, and heatmap
plots/animations are displayed in a horizontal row.

Settings are read from the `[wall_pdf]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`. When `run_info`
is not a Tuple, `plot_prefix` is optional - plots/animations will be saved only if it is
passed.
"""
function plot_charged_pdf_2D_at_wall(run_info; plot_prefix)
    input = Dict_to_NamedTuple(input_dict_dfns["wall_pdf"])
    if !(input.plot || input.animate)
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

    println("Making plots of ion distribution function at walls")
    flush(stdout)

    is_1D = all(ri !== nothing && ri.r.n == 1 for ri ∈ run_info)
    is_1V = all(ri !== nothing && ri.vperp.n == 1 for ri ∈ run_info)
    moment_kinetic = any(ri !== nothing
                         && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                         for ri ∈ run_info)

    for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+8, "wall-"),
                               (z_upper, z_upper-8:z_upper, "wall+"))
        f_input = copy(input_dict_dfns["f"])
        f_input["iz0"] = z

        if input.plot
            plot_vs_vpa(run_info, "f"; is=1, input=f_input,
                        outfile=plot_prefix * "pdf_$(label)_vs_vpa.pdf")

            if moment_kinetic
                plot_f_unnorm_vs_vpa(run_info; input=f_input, is=1,
                                     outfile=plot_prefix * "pdf_unnorm_$(label)_vs_vpa.pdf")
            end

            plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1,
                                 outfile=plot_prefix * "pdf_unnorm_over_vpa2_$(label)_vs_vpa.pdf")

            if !is_1V
                plot_vs_vpa_vperp(run_info, "f"; is=1, input=f_input,
                                  outfile=plot_prefix * "pdf_$(label)_vs_vpa_vperp.pdf")
            end

            plot_vs_vpa_z(run_info, "f"; is=1, input=f_input, iz=z_range,
                          outfile=plot_prefix * "pdf_$(label)_vs_vpa_z.pdf")

            if !is_1D
                plot_vs_z_r(run_info, "f"; is=1, input=f_input, iz=z_range,
                            outfile=plot_prefix * "pdf_$(label)_vs_z_r.pdf")

                plot_vs_vpa_r(run_info, "f"; is=1, input=f_input,
                              outfile=plot_prefix * "pdf_$(label)_vs_vpa_r.pdf")
            end
        end

        if input.animate
            animate_vs_vpa(run_info, "f"; is=1, input=f_input,
                           outfile=plot_prefix * "pdf_$(label)_vs_vpa." * input.animation_ext)

            if moment_kinetic
                animate_f_unnorm_vs_vpa(run_info; input=f_input, is=1,
                                        outfile=plot_prefix * "pdf_unnorm_$(label)_vs_vpa." * input.animation_ext)
            end

            animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1,
                                    outfile=plot_prefix * "pdf_unnorm_over_vpa2_$(label)_vs_vpa." * input.animation_ext)

            if !is_1V
                animate_vs_vpa_vperp(run_info, "f"; is=1, input=f_input,
                                     outfile=plot_prefix * "pdf_$(label)_vs_vpa_vperp." * input.animation_ext)
            end

            animate_vs_vpa_z(run_info, "f"; is=1, input=f_input, iz=z_range,
                             outfile=plot_prefix * "pdf_$(label)_vs_vpa_z." * input.animation_ext)

            if !is_1D
                animate_vs_z_r(run_info, "f"; is=1, input=f_input, iz=z_range,
                               outfile=plot_prefix * "pdf_$(label)_vs_z_r." * input.animation_ext)

                animate_vs_vpa_r(run_info, "f"; is=1, input=f_input,
                                 outfile=plot_prefix * "pdf_$(label)_vs_vpa_r." * input.animation_ext)
            end
        end
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
    input = Dict_to_NamedTuple(input_dict_dfns["wall_pdf_neutral"])
    if !(input.plot || input.animate)
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

    is_1D = all(ri !== nothing && ri.r.n == 1 for ri ∈ run_info)
    is_1V = all(ri !== nothing && ri.vzeta.n == 1 && ri.vr.n == 1 for ri ∈ run_info)
    moment_kinetic = any(ri !== nothing
                         && (ri.evolve_density || ri.evolve_upar || ri.evolve_ppar)
                         for ri ∈ run_info)

    for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+8, "wall-"),
                               (z_upper, z_upper-8:z_upper, "wall+"))
        f_neutral_input = copy(input_dict_dfns["f_neutral"])
        f_neutral_input["iz0"] = z

        if input.plot
            plot_vs_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                       outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz.pdf")

            if moment_kinetic
                plot_f_unnorm_vs_vpa(run_info; input=f_neutral_input, neutral=true, is=1,
                                     outfile=plot_prefix * "pdf_neutral_unnorm_$(label)_vs_vpa.pdf")
            end

            if !is_1V
                plot_vs_vzeta_vr(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_vzeta.pdf")
                plot_vs_vzeta_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vzeta.pdf")
                plot_vs_vr_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                              outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vr.pdf")
            end

            plot_vs_vz_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                         outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_z.pdf")

            if !is_1V
                plot_vs_vzeta_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_z.pdf")
                plot_vs_vr_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                             outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_z.pdf")
            end

            if !is_1D
                plot_vs_z_r(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                            outfile=plot_prefix * "pdf_neutral_$(label)_vs_z_r.pdf")

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
            animate_vs_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                          outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz." * input.animation_ext)

            if moment_kinetic
                animate_f_unnorm_vs_vpa(run_info; input=f_neutral_input, neutral=true, is=1,
                                        outfile=plot_prefix * "pdf_neutral_unnorm_$(label)_vs_vz." * input.animation_ext)
            end

            if !is_1V
                animate_vs_vzeta_vr(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_vzeta." * input.animation_ext)
                animate_vs_vzeta_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vzeta." * input.animation_ext)
                animate_vs_vr_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vr." * input.animation_ext)
            end

            animate_vs_vz_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                            outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_z." * input.animation_ext)

            if !is_1V
                animate_vs_vzeta_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                   outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_z." * input.animation_ext)
                animate_vs_vr_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_z." * input.animation_ext)
            end

            if !is_1D
                animate_vs_z_r(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                               outfile=plot_prefix * "pdf_neutral_$(label)_vs_z_r." * input.animation_ext)

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
    end

    return nothing
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

        for (ri, ax) ∈ zip(run_info, axes)
            Chodura_condition_plots(ri; axes=ax)
        end

        if input.plot_vs_t
            fig = figs[1]
            ax = axes[1][1]
            put_legend_right(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_t.pdf")
            save(outfile, fig)

            fig = figs[2]
            ax = axes[2][1]
            put_legend_right(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_t.pdf")
            save(outfile, fig)
        end
        if input.plot_vs_r
            fig = figs[3]
            ax = axes[3][1]
            put_legend_right(fig, ax)
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r.pdf")
            save(outfile, fig)

            fig = figs[4]
            ax = axes[4][1]
            put_legend_right(fig, ax)
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
    catch e
        println("Error in Chodura_condition_plots(). Error was ", e)
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
    density = postproc_load_variable(run_info, "density")
    upar = postproc_load_variable(run_info, "parallel_flow")
    vth = postproc_load_variable(run_info, "thermal_speed")
    Er = postproc_load_variable(run_info, "Er")
    f_lower = postproc_load_variable(run_info, "f", iz=1)
    f_upper = postproc_load_variable(run_info, "f", iz=run_info.z.n_global)

    Chodura_ratio_lower, Chodura_ratio_upper =
        check_Chodura_condition(run_info.r_local, run_info.z_local, run_info.vperp,
                                run_info.vpa, density, upar, vth, run_info.composition,
                                Er, run_info.geometry, run_info.z.bc, nothing;
                                evolve_density=run_info.evolve_density,
                                evolve_upar=run_info.evolve_upar,
                                evolve_ppar=run_info.evolve_ppar,
                                f_lower=f_lower, f_upper=f_upper)

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
        println("Error in sound_wave_plots(). Error was ", e)
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
        phi = postproc_load_variable(run_info, "phi"; ir=input.ir0)
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
        variable = postproc_load_variable(run_info, "thermal_speed").^2
    else
        variable = postproc_load_variable(run_info, variable_name)
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
            println("Warning: error in 1D Fourier analysis for $variable_name. Error was $e")
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
            println("Warning: error in 2D Fourier analysis for $variable_name. Error was $e")
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
            println("Warning: error in perturbation animation for $variable_name. Error was $e")
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
                                   run_info.r.bc, run_info.z.bc, run_info.geometry,
                                   run_info.composition, run_info.species, run_info.r.n,
                                   nvperp)
    end

    variable_func = manufactured_funcs[func_name_lookup[variable_name]]

    variable = postproc_load_variable(run_info, String(variable_name); it=tinds, is=1,
                                      ir=ir, iz=iz, ivperp=ivperp, ivpa=ivpa,
                                      ivzeta=ivzeta, ivr=ivr, ivz=ivz)
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

    is_1D = (r.n == 1)

    if !is_1D && input.wall_plots
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
    if !is_1D && input.plot_vs_r
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
    if input.plot_vs_z
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
    if !is_1D && input.plot_vs_r_t
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
    if input.plot_vs_z_t
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
    if !is_1D && input.plot_vs_z_r
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
    if !is_1D && input.animate_vs_r
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
    if input.animate_vs_z
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
    if !is_1D && input.animate_vs_z_r
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
[`compare_charged_pdf_symbolic_test`](@ref) and
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
    compare_charged_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
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
function compare_charged_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
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

    is_1D = (r.n == 1)
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

            if !is_1D
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
    if is_1D
        animate_dims = setdiff(animate_dims, (:r,))
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

    is_1D = (r.n == 1)
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

            if !is_1D
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
    if is_1D
        animate_dims = setdiff(animate_dims, (:r,))
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
        println("Error in manufactured_solutions_analysis(). Error was ", e)
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
        println("Error in manufactured_solutions_analysis_dfns(). Error was ", e)
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

        compare_charged_pdf_symbolic_test(run_info, plot_prefix; io=io, input=input)

        if run_info.n_neutral_species > 0
            compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=io, input=input)
        end
    end

    return nothing
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
