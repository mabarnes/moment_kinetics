"""
Post processing functions using Makie.jl
"""
module makie_post_processing

export makie_post_process

using ..analysis: analyze_fields_data, check_Chodura_condition, get_r_perturbation,
                  get_Fourier_modes_2D, get_Fourier_modes_1D
using ..array_allocation: allocate_float
using ..coordinates: define_coordinate
using ..input_structs: grid_input, advection_input
using ..looping: all_dimensions, ion_dimensions, neutral_dimensions
using ..moment_kinetics_input: mk_input, set_defaults_and_check_top_level!,
                               set_defaults_and_check_section!, Dict_to_NamedTuple
using ..load_data: open_readonly_output_file, get_group, load_block_data,
                   load_coordinate_data, load_input, load_mk_options, load_species_data,
                   load_time_data
using ..post_processing: calculate_and_write_frequencies, construct_global_zr_coords,
                         get_geometry_and_composition
using ..type_definitions: mk_float, mk_int

using Combinatorics
using Glob
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
const input_dict_dfns = OrderedDict{String,Any}()

const em_variables = ("phi", "Er", "Ez")
const ion_moment_variables = ("density", "parallel_flow", "parallel_pressure",
                              "thermal_speed", "temperature", "parallel_heat_flux")
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

"""
Run post processing with input read from a TOML file

If file does not exist, prints warning and uses default options.
"""
function makie_post_process(run_prefix...;
                            input_file::String=default_input_file_name, kwargs...)
    if isfile(input_file)
        new_input_dict = TOML.parsefile(input_file)
    else
        println("Warning: $input_file does not exist, using default post-processing "
                * "options")
        new_input_dict = Dict{String,Any}()
    end

    return makie_post_process(run_prefix, new_input_dict; kwargs...)
end

"""
Run post prossing, with (non-default) input given in a Dict

`run_dir` is the path to an output directory, or (to make comparison plots) a tuple of
paths to output directories.

`input_dict` is a dictionary containing settings for the post-processing.

`restart_index` specifies which restart to read if there are multiple restarts. The
default (`nothing`) reads all restarts and concatenates them. An integer value reads the
restart with that index - `-1` indicates the latest restart (which does not have an
index). A tuple with the same length as `run_dir` can also be passed to give a different
`restart_index` for each run.
"""
function makie_post_process(run_dir::Union{String,Tuple},
                            new_input_dict::AbstractDict{String,Any};
                            restart_index::Union{Nothing,mk_int,Tuple}=nothing)
    if isa(run_dir, String)
        # Make run_dir a one-element tuple if it is not a tuple
        run_dir = (run_dir,)
    end
    # Normalise by removing any trailing slashes - with a slash basename() would return an
    # empty string
    run_dir = Tuple(rstrip(ri, '/') for ri ∈ run_dir)

    if !isa(restart_index, Tuple)
        # Convert scalar restart_index to Tuple so we can treat everything the same below
        restart_index = Tuple(restart_index for _ ∈ run_dir)
    end

    # Special handling for itime_* and itime_*_dfns because they are needed in order to
    # set up `time` and `time_dfns` in run_info, but run_info is needed to set several
    # other default values in setup_makie_post_processing_input!().
    itime_min = get(new_input_dict, "itime_min", 1)
    itime_max = get(new_input_dict, "itime_max", -1)
    itime_skip = get(new_input_dict, "itime_skip", 1)
    itime_min_dfns = get(new_input_dict, "itime_min_dfns", 1)
    itime_max_dfns = get(new_input_dict, "itime_max_dfns", -1)
    itime_skip_dfns = get(new_input_dict, "itime_skip_dfns", 1)
    run_info_moments = Tuple(get_run_info(p, i, itime_min=itime_min, itime_max=itime_max,
                                          itime_skip=itime_skip,
                                          itime_min_dfns=itime_min_dfns,
                                          itime_max_dfns=itime_max_dfns,
                                          itime_skip_dfns=itime_skip_dfns)
                             for (p,i) in zip(run_dir, restart_index))
    run_info_dfns = Tuple(get_run_info(p, i, itime_min=itime_min, itime_max=itime_max,
                                       itime_skip=itime_skip,
                                       itime_min_dfns=itime_min_dfns,
                                       itime_max_dfns=itime_max_dfns,
                                       itime_skip_dfns=itime_skip_dfns, dfns=true)
                          for (p,i) in zip(run_dir, restart_index))

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

    if length(run_info) == 1
        plot_prefix = joinpath(run_dir[1], basename(run_dir[1])) * "_"
    else
        plot_prefix = "comparison_plots/compare_"
    end

    for variable_name ∈ moment_variable_list
        plots_for_variable(run_info, variable_name, plot_prefix=plot_prefix, is_1D=is_1D,
                           is_1V=is_1V)
    end

    # Plots from distribution function variables
    ############################################
    if any(ri !== nothing for ri in run_info_dfns)
        dfn_variable_list = ion_dfn_variables
        if has_neutrals
            dfn_variable_list = tuple(dfn_variable_list..., neutral_dfn_variables...)
        end
        for variable_name ∈ dfn_variable_list
            plots_for_dfn_variable(run_info_dfns, variable_name, plot_prefix=plot_prefix,
                                   is_1D=is_1D, is_1V=is_1V)
        end
    end

    plot_charged_pdf_2D_at_wall(run_info_dfns; plot_prefix=plot_prefix)

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

    Chodura_condition_plots(run_info_dfns, plot_prefix=plot_prefix)

    sound_wave_plots(run_info; plot_prefix=plot_prefix)

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

    original_input = deepcopy(input_dict)
    original_input_dfns = deepcopy(input_dict_dfns)

    # Set up input_dict and input_dict_dfns with all-default parameters
    setup_makie_post_processing_input!(Dict{String,Any}())

    # Merge input_dict and input_dict_dfns, then convert to a String formatted as the
    # contents of a TOML file
    combined_input_dict = merge(input_dict_dfns, input_dict)
    buffer = IOBuffer()
    TOML.print(buffer, combined_input_dict)
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

    # Restore original state of input_dict and input_dict_dfns
    clear_Dict!(input_dict)
    clear_Dict!(input_dict_dfns)
    merge!(input_dict, original_input)
    merge!(input_dict_dfns, original_input_dfns)

    return nothing
end

"""
    setup_makie_post_processing_input!(
        input_file::String=default_input_file_name; run_info_moment=nothing,
        run_info_dfns=nothing, ignore_missing_file::Bool=false)
    setup_makie_post_processing_input!(new_input_dict::AbstractDict{String,Any};
                                       run_info_moments=run_info_moments,
                                       run_info_dfns=run_info_dfns)

Set up input, storing in the global `input_dict`
"""
function setup_makie_post_processing_input! end

function setup_makie_post_processing_input!(
        input_file::String=default_input_file_name; run_info_moments=nothing,
        run_info_dfns=nothing, ignore_missing_file::Bool=false)

    if isfile(input_file)
        new_input_dict = TOML.parsefile(input_file)
    else
        if ignore_missing_file
            new_input_dict = Dict{String,Any}()
        else
            error("$input_file does not exist")
        end
    end
    setup_makie_post_processing_input!(new_input_dict, run_info_moments=run_info_moments,
                                       run_info_dfns=run_info_dfns)

    return nothing
end

function setup_makie_post_processing_input!(new_input_dict::AbstractDict{String,Any};
                                            run_info_moments=nothing,
                                            run_info_dfns=nothing)
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

function _setup_single_input!(this_input_dict::AbstractDict{String,Any},
                              new_input_dict::AbstractDict{String,Any}, run_info,
                              dfns::Bool)
    # Remove all existing entries from this_input_dict
    clear_Dict!(this_input_dict)

    # Put entries from new_input_dict into this_input_dict
    merge!(this_input_dict, new_input_dict)

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
    time_index_options = ("it0", "it0_dfns", "itime_min", "itime_max", "itime_skip",
                          "itime_min_dfns", "itime_max_dfns", "itime_skip_dfns")

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
      )

    for variable_name ∈ all_moment_variables
        section_defaults = OrderedDict(k=>v for (k,v) ∈ this_input_dict
                                       if !isa(v, AbstractDict) &&
                                          !(k ∈ time_index_options))
        if dfns
            section_defaults["it0"] = this_input_dict["it0_dfns"]
        else
            section_defaults["it0"] = this_input_dict["it0"]
        end
        set_defaults_and_check_section!(
            this_input_dict, variable_name;
            OrderedDict(Symbol(k)=>v for (k,v) ∈ section_defaults)...)
    end

    if dfns
        for variable_name ∈ all_dfn_variables
            section_defaults = OrderedDict(k=>v for (k,v) ∈ this_input_dict
                                           if !isa(v, AbstractDict) &&
                                              !(k ∈ time_index_options))
            section_defaults["it0"] = this_input_dict["it0_dfns"]
            set_defaults_and_check_section!(
                this_input_dict, variable_name;
                check_moments=false,
                plot_log_vs_r=false,
                plot_log_vs_z=false,
                plot_vs_vperp=false,
                plot_log_vs_vperp=false,
                plot_vs_vpa=false,
                plot_log_vs_vpa=false,
                plot_vs_vzeta=false,
                plot_log_vs_vzeta=false,
                plot_vs_vr=false,
                plot_log_vs_vr=false,
                plot_vs_vz=false,
                plot_log_vs_vz=false,
                plot_log_vs_r_t=false,
                plot_log_vs_z_t=false,
                plot_vs_vperp_t=false,
                plot_log_vs_vperp_t=false,
                plot_vs_vpa_t=false,
                plot_log_vs_vpa_t=false,
                plot_vs_vzeta_t=false,
                plot_log_vs_vzeta_t=false,
                plot_vs_vr_t=false,
                plot_log_vs_vr_t=false,
                plot_vs_vz_t=false,
                plot_log_vs_vz_t=false,
                plot_log_vs_z_r=false,
                plot_vs_vperp_r=false,
                plot_log_vs_vperp_r=false,
                plot_vs_vpa_r=false,
                plot_log_vs_vpa_r=false,
                plot_vs_vperp_z=false,
                plot_log_vs_vperp_z=false,
                plot_vs_vpa_z=false,
                plot_log_vs_vpa_z=false,
                plot_vs_vpa_vperp=false,
                plot_log_vs_vpa_vperp=false,
                plot_vs_vzeta_r=false,
                plot_log_vs_vzeta_r=false,
                plot_vs_vr_r=false,
                plot_log_vs_vr_r=false,
                plot_vs_vz_r=false,
                plot_log_vs_vz_r=false,
                plot_vs_vzeta_z=false,
                plot_log_vs_vzeta_z=false,
                plot_vs_vr_z=false,
                plot_log_vs_vr_z=false,
                plot_vs_vz_z=false,
                plot_log_vs_vz_z=false,
                plot_vs_vr_vzeta=false,
                plot_log_vs_vr_vzeta=false,
                plot_vs_vz_vzeta=false,
                plot_log_vs_vz_vzeta=false,
                plot_vs_vz_vr=false,
                plot_log_vs_vz_vr=false,
                animate_log_vs_r=false,
                animate_log_vs_z=false,
                animate_log_vs_z_r=false,
                animate_vs_vperp=false,
                animate_log_vs_vperp=false,
                animate_vs_vperp_r=false,
                animate_log_vs_vperp_r=false,
                animate_vs_vperp_z=false,
                animate_log_vs_vperp_z=false,
                animate_vs_vpa=false,
                animate_log_vs_vpa=false,
                animate_vs_vpa_r=false,
                animate_log_vs_vpa_r=false,
                animate_vs_vpa_z=false,
                animate_log_vs_vpa_z=false,
                animate_vs_vpa_vperp=false,
                animate_log_vs_vpa_vperp=false,
                animate_vs_vzeta=false,
                animate_log_vs_vzeta=false,
                animate_vs_vzeta_r=false,
                animate_log_vs_vzeta_r=false,
                animate_vs_vzeta_z=false,
                animate_log_vs_vzeta_z=false,
                animate_vs_vr=false,
                animate_log_vs_vr=false,
                animate_vs_vr_r=false,
                animate_log_vs_vr_r=false,
                animate_vs_vr_z=false,
                animate_log_vs_vr_z=false,
                animate_vs_vz=false,
                animate_log_vs_vz=false,
                animate_vs_vz_r=false,
                animate_log_vs_vz_r=false,
                animate_vs_vz_z=false,
                animate_log_vs_vz_z=false,
                animate_vs_vr_vzeta=false,
                animate_log_vs_vr_vzeta=false,
                animate_vs_vz_vzeta=false,
                animate_log_vs_vz_vzeta=false,
                animate_vs_vz_vr=false,
                animate_log_vs_vz_vr=false,
                OrderedDict(Symbol(k)=>v for (k,v) ∈ section_defaults)...)
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
        this_input_dict, "Chodura_condition";
        plot_vs_t=false,
        plot_vs_r_t=false,
        ir0=input_dict["ir0"],
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
        ir0=input_dict["ir0"],
        iz0=input_dict["iz0"],
       )

    return nothing
end

"""
Get file handles and other info for a single run

By default load data from moments files, pass `dfns=true` to load from distribution
functions files.
"""
function get_run_info(run_dir, restart_index; itime_min=1, itime_max=-1, itime_skip=1,
                      itime_min_dfns=1, itime_max_dfns=-1, itime_skip_dfns=1, dfns=false)
    if !isdir(run_dir)
        error("$run_dir is not a directory")
    end

    # Normalise by removing any trailing slash - with a slash basename() would return an
    # empty string
    run_dir = rstrip(run_dir, '/')

    run_name = basename(run_dir)
    base_prefix = joinpath(run_dir, run_name)
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
    if itime_max > 0
        time = time[itime_min:itime_skip:itime_max]
    else
        time = time[itime_min:itime_skip:end]
    end
    nt = length(time)

    # Get input and coordinates from the final restart
    file_final_restart = fids0[end]

    input = load_input(file_final_restart)

    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    io_input, evolve_moments, t_input, z_input, r_input, vpa_input, vperp_input,
        gyrophase_input, vz_input, vr_input, vzeta_input, composition, species,
        collisions, geometry, drive_input, num_diss_params, manufactured_solns_input =
        mk_input(input)

    n_ion_species, n_neutral_species = load_species_data(file_final_restart)
    evolve_density, evolve_upar, evolve_ppar = load_mk_options(file_final_restart)

    z_local, z_local_spectral = load_coordinate_data(file_final_restart, "z")
    r_local, r_local_spectral = load_coordinate_data(file_final_restart, "r")
    r, r_spectral, z, z_spectral = construct_global_zr_coords(r_local, z_local)

    if dfns
        vperp, vperp_spectral = load_coordinate_data(file_final_restart, "vperp")
        vpa, vpa_spectral = load_coordinate_data(file_final_restart, "vpa")

        if n_neutral_species > 0
            vzeta, vzeta_spectral = load_coordinate_data(file_final_restart, "vzeta")
            vr, vr_spectral = load_coordinate_data(file_final_restart, "vr")
            vz, vz_spectral = load_coordinate_data(file_final_restart, "vz")
        else
            dummy_adv_input = advection_input("default", 1.0, 0.0, 0.0)
            dummy_comm = MPI.COMM_NULL
            dummy_input = grid_input("dummy", 1, 1, 1, 1, 0, 1.0,
                                     "chebyshev_pseudospectral", "", "periodic",
                                     dummy_adv_input, dummy_comm)
            vzeta, vzeta_spectral = define_coordinate(dummy_input)
            vr, vr_spectral = define_coordinate(dummy_input)
            vz, vz_spectral = define_coordinate(dummy_input)
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
        return (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                ext=ext, nblocks=nblocks, files=files, input=input,
                n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                evolve_moments=evolve_moments, composition=composition, species=species,
                collisions=collisions, geometry=geometry, drive_input=drive_input,
                num_diss_params=num_diss_params,
                manufactured_solns_input=manufactured_solns_input, nt=nt,
                nt_unskipped=nt_unskipped, restarts_nt=restarts_nt, itime_skip=itime_skip,
                time=time, r=r, z=z, vperp=vperp, vpa=vpa, vzeta=vzeta, vr=vr, vz=vz,
                r_local=r_local, z_local=z_local, r_spectral=r_spectral,
                z_spectral=z_spectral, vperp_spectral=vperp_spectral,
                vpa_spectral=vpa_spectral, vzeta_spectral=vzeta_spectral,
                vr_spectral=vr_spectral, vz_spectral=vz_spectral)
    else
        return (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                ext=ext, nblocks=nblocks, files=files, input=input,
                n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                evolve_moments=evolve_moments, composition=composition, species=species,
                collisions=collisions, geometry=geometry, drive_input=drive_input,
                num_diss_params=num_diss_params,
                manufactured_solns_input=manufactured_solns_input, nt=nt,
                nt_unskipped=nt_unskipped, restarts_nt=restarts_nt, itime_skip=itime_skip,
                time=time, r=r, z=z, r_local=r_local, z_local=z_local,
                r_spectral=r_spectral, z_spectral=z_spectral)
    end
end

"""
Load a variable

The result always has a time dimension, even if the slice `it` is `mk_int` (in which case
the time dimension will have size 1).
"""
function postproc_load_variable(run_info, variable_name; it=nothing, is=nothing,
                                ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                                ivzeta=nothing, ivr=nothing, ivz=nothing)
    nt = run_info.nt

    if it === nothing
        it = 1:nt
    elseif isa(it, mk_int)
        nt = 1
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
    elseif isa(it, mk_int)
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

    if isa(it, mk_int)
        nt = 1
    elseif it === (:)
        it = 1:nt
    else
        nt = length(it)
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
                if length(tinds) == 0
                    # Nothing to do in this file
                    continue
                elseif length(tinds) > 1
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

        if run_info.itime_skip != 1 && length(run_info.files) > 1
            error("Setting itime_skip!=1 and combining multiple restarts of a single run "
                  * "not currently supported when parallel_io=false")
        end
        if nd == 3
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.nt)
            local_start = 1
            for (f, this_nt) ∈ zip(run_info.files, run_info.restarts_nt)
                read_distributed_zr_data!(
                    @view(result[:,:,local_start:local_start+this_nt-1]), variable_name, f,
                    run_info.ext, run_info.nblocks, run_info.z_local.n,
                    run_info.r_local.n, run_info.itime_skip)
                local_start += this_nt - 1
            end
            result = result[iz,ir,:]
        elseif nd == 4
            # If we ever have neutrals included but n_neutral_species != n_ion_species,
            # then this will fail - in that case would need some way to specify that we
            # need to read a neutral moment variable rather than an ion moment variable
            # here.
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.n_ion_species,
                                    run_info.nt)
            local_start = 1
            for (f, this_nt) ∈ zip(run_info.files, run_info.restarts_nt)
                read_distributed_zr_data!(
                    @view(result[:,:,:,local_start:local_start+this_nt-1]), variable_name,
                    f, run_info.ext, run_info.nblocks, run_info.z_local.n,
                    run_info.r_local.n, run_info.itime_skip)
                local_start += this_nt - 1
            end
            result = result[iz,ir,is,:]
        elseif nd === 6
            parts = Vector{Array{mk_float,6}}()
            local_it_start = 1
            for (f, this_nt) ∈ zip(run_info.files, run_info.restarts_nt)
                local_it_end = local_it_start+this_nt-1

                tinds = collect(i - local_it_start + 1 + offset for i ∈ it
                                if local_it_start <= i <= local_it_end)
                push!(parts, load_distributed_charged_pdf_slice(
                                 f, run_info.nblocks, tinds, run_info.n_ion_species,
                                 run_info.r, run_info.z, run_info.vperp, run_info.vpa;
                                 is=is, ir=ir, iz=iz, ivperp=ivperp, ivpa=ivpa))
                local_it_start = local_it_end + 1
            end
            result = cat(parts...; dims=6)
        elseif nd === 7
            parts = Vector{Array{mk_float,7}}()
            local_it_start = 1
            for (f, this_nt) ∈ zip(run_info.files, run_info.restarts_nt)
                local_it_end = local_it_start+this_nt-1

                tinds = collect(i - local_it_start + 1 + offset for i ∈ it
                                if local_it_start <= i <= local_it_end)
                push!(parts, load_distributed_neutral_pdf_slice(
                                 f, run_info.nblocks, tinds, run_info.n_ion_species,
                                 run_info.r, run_info.z, run_info.vperp, run_info.vpa;
                                 is=(is === (:) ? nothing : is),
                                 ir=(ir === (:) ? nothing : ir),
                                 iz=(iz === (:) ? nothing : iz),
                                 ivzeta=(ivzeta === (:) ? nothing : ivzeta),
                                 ivr=(ivr === (:) ? nothing : ivr),
                                 ivz=(ivz === (:) ? nothing : ivz)))
                local_it_start = local_it_end + 1
            end
            if length(run_info.files) == 1
                # Special case to avoid an extra allocation/copy of a potentially large
                # array when we don't actually need to `cat()`
                result = parts[1]
            else
                result = cat(parts...; dims=7)
            end
        end
    end

    return result
end

function plots_for_variable(run_info, variable_name; plot_prefix, is_1D=false,
                            is_1V=false)
    println("Making plots for $variable_name")
    flush(stdout)

    input = Dict_to_NamedTuple(input_dict[variable_name])
    # Use the global settings for "itime_*" to be consistent with the `time` in
    # `run_info`.
    tinds = input_dict["itime_min"]:input_dict["itime_skip"]:input_dict["itime_max"]

    # test if any plot is needed
    if any(v for (k,v) in pairs(input) if
           startswith(String(k), "plot") || startswith(String(k), "animate"))
        if variable_name == "temperature"
            vth = Tuple(postproc_load_variable(ri, "thermal_speed"; it=tinds)
                        for ri ∈ run_info)
            variable = Tuple(v.^2 for v ∈ vth)
        elseif variable_name == "temperature_neutral"
            vth = Tuple(postproc_load_variable(ri, "thermal_speed_neutral"; it=tinds)
                        for ri ∈ run_info)
            variable = Tuple(v.^2 for v ∈ vth)
        else
            variable = Tuple(postproc_load_variable(ri, variable_name; it=tinds)
                             for ri ∈ run_info)
        end
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
            if input.plot_vs_z_r
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
        end
    end

    return nothing
end

function plots_for_dfn_variable(run_info, variable_name; plot_prefix, is_1D=false,
                                is_1V=false)
    println("Making plots for $variable_name")
    flush(stdout)

    input = Dict_to_NamedTuple(input_dict_dfns[variable_name])
    # Use the global settings for "itime_*" to be consistent with the `time` in
    # `run_info`.
    tinds = input_dict_dfns["itime_min_dfns"]:input_dict_dfns["itime_skip_dfns"]:input_dict_dfns["itime_max_dfns"]

    # test if any plot is needed
    if any(v for (k,v) in pairs(input) if
           startswith(String(k), "plot") || startswith(String(k), "animate"))
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
                plot_vs_r_t(run_info, variable_name, is=is, input=input,
                            outfile=variable_prefix * "vs_r_t.pdf")
            end
            if input.plot_vs_z_t
                plot_vs_z_t(run_info, variable_name, is=is, input=input,
                            outfile=variable_prefix * "vs_z_t.pdf")
            end
            if !is_1D && input.plot_vs_r
                plot_vs_r(run_info, variable_name, is=is, input=input,
                          outfile=variable_prefix * "vs_r.pdf")
            end
            if input.plot_vs_z
                plot_vs_z(run_info, variable_name, is=is, input=input,
                          outfile=variable_prefix * "vs_z.pdf")
            end
            if input.plot_vs_z_r
                plot_vs_z_r(run_info, variable_name, is=is, input=input,
                            outfile=variable_prefix * "vs_z_r.pdf")
            end
            if input.animate_vs_z
                animate_vs_z(run_info, variable_name, is=is, input=input,
                             outfile=variable_prefix * "vs_z." * input.animation_ext)
            end
            if !is_1D && input.animate_vs_r
                animate_vs_r(run_info, variable_name, is=is, input=input,
                             outfile=variable_prefix * "vs_r." * input.animation_ext)
            end
            if !is_1D && input.animate_vs_z_r
                animate_vs_z_r(run_info, variable_name, is=is, input=input,
                               outfile=variable_prefix * "vs_r." * input.animation_ext)
            end

            if variable_name ∈ all_dfn_variables
                if input.check_moments
                    error("checking moments using analyze_pdf_data() not implemented yet")
                end

                if !is_1D && input.plot_log_vs_r
                    plot_vs_r(run_info, variable_name, is=is, input=input,
                              outfile=log_variable_prefix * "vs_r.pdf", yscale=log10,
                              transform=positive_or_nan)
                end
                if input.plot_log_vs_z
                    plot_vs_z(run_info, variable_name, is=is, input=input,
                              outfile=log_variable_prefix * "vs_z.pdf", yscale=log10,
                              transform=positive_or_nan)
                end
                if !is_1D && input.plot_log_vs_r_t
                    plot_vs_r_t(run_info, variable_name, is=is, input=input,
                                outfile=log_variable_prefix * "vs_r_t.pdf",
                                colorscale=log10, transform=positive_or_nan)
                end
                if input.plot_log_vs_z_t
                    plot_vs_z_t(run_info, variable_name, is=is, input=input,
                                outfile=log_variable_prefix * "vs_z_t.pdf",
                                colorscale=log10, transform=positive_or_nan)
                end
                if input.plot_log_vs_z_r
                    plot_vs_z_r(run_info, variable_name, is=is, input=input,
                                outfile=log_variable_prefix * "vs_z_r.pdf",
                                colorscale=log10, transform=positive_or_nan)
                end

                if input.animate_log_vs_z
                    # Note that we use `yscale=log10` and `transform=positive_or_nan`
                    # rather than defining a custom scaling function (which would return
                    # NaN for negative values) because it messes up the automatic minimum
                    # value for the colorscale: The transform removes any zero or negative
                    # values from the data, so the minimum value for the colorscale is set
                    # by the smallest positive value; with only the custom colorscale, the
                    # minimum would be negative and the corresponding color would be the
                    # color for NaN, which does not go on the Colorbar and so causes an
                    # error.
                    animate_vs_z(run_info, variable_name, is=is, input=input,
                                 outfile=log_variable_prefix * "vs_z." * input.animation_ext,
                                 yscale=log10, transform=positive_or_nan)
                end
                if !is_1D && input.animate_log_vs_r
                    animate_vs_z(run_info, variable_name, is=is, input=input,
                                 outfile=log_variable_prefix * "vs_z." * input.animation_ext,
                                 yscale=log10, transform=positive_or_nan)
                end
                if !is_1D && input.animate_log_vs_z_r
                    animate_vs_z_r(run_info, variable_name, is=is, input=input,
                                   outfile=log_variable_prefix * "vs_z." *
                                   input.animation_ext, colorscale=log10,
                                   transform=positive_or_nan)
                end
            end
            if variable_name ∈ ion_dfn_variables
                if !is_1V && input.plot_vs_vperp
                    plot_vs_vperp(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vperp.pdf")
                end
                if !is_1V && input.plot_log_vs_vperp
                    plot_vs_vperp(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vperp.pdf",
                                  yscale=log10, transform=positive_or_nan)
                end
                if input.plot_vs_vpa
                    plot_vs_vpa(run_info, variable_name, is=is, input=input,
                                outfile=variable_prefix * "vs_vpa.pdf")
                end
                if input.plot_log_vs_vpa
                    plot_vs_vpa(run_info, variable_name, is=is, input=input,
                                outfile=log_variable_prefix * "vs_vpa.pdf", yscale=log10,
                                transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vperp_t
                    plot_vs_vperp_t(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vperp_t.pdf")
                end
                if !is_1V && input.plot_log_vs_vperp_t
                    plot_vs_vperp_t(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vperp_t.pdf",
                                    colorscale=log10, transform=positive_or_nan)
                end
                if input.plot_vs_vpa_t
                    plot_vs_vpa_t(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vpa_t.pdf")
                end
                if input.plot_log_vs_vpa_t
                    plot_vs_vpa_t(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vpa_t.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1D && !is_1V && input.plot_vs_vperp_r
                    plot_vs_vperp_r(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vperp_r.pdf")
                end
                if !is_1D && !is_1V && input.plot_log_vs_vperp_r
                    plot_vs_vperp_r(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vperp_r.pdf",
                                    colorscale=log10, transform=positive_or_nan)
                end
                if !is_1D && input.plot_vs_vpa_r
                    plot_vs_vpa_r(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vpa_r.pdf")
                end
                if !is_1D && input.plot_log_vs_vpa_r
                    plot_vs_vpa_r(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vpa_r.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vperp_z
                    plot_vs_vperp_z(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vperp_z.pdf")
                end
                if !is_1V && input.plot_log_vs_vperp_z
                    plot_vs_vperp_z(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vperp_z.pdf",
                                    colorscale=log10, transform=positive_or_nan)
                end
                if input.plot_vs_vpa_z
                    plot_vs_vpa_z(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vpa_z.pdf")
                end
                if input.plot_log_vs_vpa_z
                    plot_vs_vpa_z(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vpa_z.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vpa_vperp
                    plot_vs_vpa_vperp(run_info, variable_name, is=is, input=input,
                                      outfile=variable_prefix * "vs_vpa_vperp.pdf")
                end
                if !is_1V && input.plot_log_vs_vpa_vperp
                    plot_vs_vpa_vperp(run_info, variable_name, is=is, input=input,
                                      outfile=log_variable_prefix * "vs_vpa_vperp.pdf",
                                      colorscale=log10, transform=positive_or_nan)
                end

                if !is_1V && input.animate_vs_vperp
                    animate_vs_vperp(run_info, variable_name, is=is, input=input,
                                     outfile=variable_prefix * "vs_vperp." *
                                     input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vperp
                    animate_vs_vperp(run_info, variable_name, is=is, input=input,
                                     outfile=log_variable_prefix * "vs_vperp." *
                                     input.animation_ext, yscale=log10,
                                     transform=positive_or_nan)
                end
                if !is_1D && !is_1V && input.animate_vs_vperp_z
                    animate_vs_vperp_r(run_info, variable_name, is=is, input=input,
                                       outfile=variable_prefix * "vs_vperp_r." *
                                       input.animation_ext)
                end
                if !is_1D && !is_1V && input.animate_log_vs_vperp_z
                    animate_vs_vperp_r(run_info, variable_name, is=is, input=input,
                                       outfile=log_variable_prefix * "vs_vperp_r." *
                                       input.animation_ext, colorscale=log10,
                                       transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vperp_z
                    animate_vs_vperp_z(run_info, variable_name, is=is, input=input,
                                       outfile=variable_prefix * "vs_vperp_z." *
                                       input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vperp_z
                    animate_vs_vperp_z(run_info, variable_name, is=is, input=input,
                                       outfile=log_variable_prefix * "vs_vperp_z." *
                                       input.animation_ext, colorscale=log10,
                                       transform=positive_or_nan)
                end
                if input.animate_vs_vpa
                    animate_vs_vpa(run_info, variable_name, is=is, input=input,
                                   outfile=variable_prefix * "vs_vpa." * input.animation_ext)
                end
                if input.animate_log_vs_vpa
                    animate_vs_vpa(run_info, variable_name, is=is, input=input,
                                   outfile=log_variable_prefix * "vs_vpa." * input.animation_ext,
                                   yscale=log10, transform=positive_or_nan)
                end
                if !is_1D && input.animate_vs_vpa_r
                    animate_vs_vpa_r(run_info, variable_name, is=is, input=input,
                                     outfile=variable_prefix * "vs_vpa_r." * input.animation_ext)
                end
                if !is_1D && input.animate_log_vs_vpa_r
                    animate_vs_vpa_r(run_info, variable_name, is=is, input=input,
                                     outfile=log_variable_prefix * "vs_vpa_r." * input.animation_ext,
                                     colorscale=log10, transform=positive_or_nan)
                end
                if input.animate_vs_vpa_z
                    animate_vs_vpa_z(run_info, variable_name, is=is, input=input,
                                     outfile=variable_prefix * "vs_vpa_z." * input.animation_ext)
                end
                if input.animate_log_vs_vpa_z
                    animate_vs_vpa_z(run_info, variable_name, is=is, input=input,
                                     outfile=log_variable_prefix * "vs_vpa_z." * input.animation_ext,
                                     colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vpa_vperp
                    animate_vs_vpa_vperp(run_info, variable_name, is=is, input=input,
                                         outfile=variable_prefix * "vs_vpa_vperp." *
                                         input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vpa_vperp
                    animate_vs_vpa_vperp(run_info, variable_name, is=is, input=input,
                                         outfile=log_variable_prefix * "vs_vpa_vperp." *
                                         input.animation_ext, colorscale=log10,
                                         transform=positive_or_nan)
                end
            end
            if variable_name ∈ neutral_dfn_variables
                if !is_1V && input.plot_vs_vzeta_t
                    plot_vs_vzeta_t(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vzeta_t.pdf")
                end
                if !is_1V && input.plot_log_vs_vzeta_t
                    plot_vs_vzeta_t(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vzeta_t.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vr_t
                    plot_vs_vr_t(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vr_t.pdf")
                end
                if !is_1V && input.plot_log_vs_vr_t
                    plot_vs_vr_t(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vr_t.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if input.plot_vs_vz_t
                    plot_vs_vz_t(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vz_t.pdf")
                end
                if input.plot_log_vs_vz_t
                    plot_vs_vz_t(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vz_t.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1D && !is_1V && input.plot_vs_vzeta_r
                    plot_vs_vzeta_r(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vzeta_r.pdf")
                end
                if !is_1D && !is_1V && input.plot_log_vs_vzeta_r
                    plot_vs_vzeta_r(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vzeta_r.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1D && !is_1V && input.plot_vs_vr_r
                    plot_vs_vr_r(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vr_r.pdf")
                end
                if !is_1D && !is_1V && input.plot_log_vs_vr_r
                    plot_vs_vr_r(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vr_r.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1D && input.plot_vs_vz_r
                    plot_vs_vz_r(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vz_r.pdf")
                end
                if !is_1D && input.plot_log_vs_vz_r
                    plot_vs_vz_r(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vz_r.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vzeta_z
                    plot_vs_vzeta_z(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vzeta_z.pdf")
                end
                if !is_1V && input.plot_log_vs_vzeta_z
                    plot_vs_vzeta_z(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vzeta_z.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vr_z
                    plot_vs_vr_z(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vr_z.pdf")
                end
                if !is_1V && input.plot_log_vs_vr_z
                    plot_vs_vr_z(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vr_z.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if input.plot_vs_vz_z
                    plot_vs_vz_z(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vz_z.pdf")
                end
                if input.plot_log_vs_vz_z
                    plot_vs_vz_z(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vz_z.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vr_vzeta
                    plot_vs_vr_vzeta(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vr_vzeta.pdf")
                end
                if !is_1V && input.plot_log_vs_vr_vzeta
                    plot_vs_vr_vzeta(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vr_vzeta.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vz_vzeta
                    plot_vs_vz_vzeta(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vz_vzeta.pdf")
                end
                if !is_1V && input.plot_log_vs_vz_vzeta
                    plot_vs_vz_vzeta(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vz_vzeta.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.plot_vs_vz_vr
                    plot_vs_vz_vr(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vz_vr.pdf")
                end
                if !is_1V && input.plot_log_vs_vz_vr
                    plot_vs_vz_vr(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vz_vr.pdf",
                                  colorscale=log10, transform=positive_or_nan)
                end

                if !is_1V && input.animate_vs_vzeta
                    animate_vs_vzeta(run_info, variable_name, is=is, input=input,
                                     outfile=variable_prefix * "vs_vzeta." *
                                     input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vzeta
                    animate_vs_vzeta(run_info, variable_name, is=is, input=input,
                                     outfile=log_variable_prefix * "vs_vzeta." *
                                     input.animation_ext, yscale=log10,
                                     transform=positive_or_nan)
                end
                if !is_1V && !is_1D && input.animate_vs_vzeta_r
                    animate_vs_vzeta_r(run_info, variable_name, is=is, input=input,
                                       outfile=variable_prefix * "vs_vzeta_r." *
                                       input.animation_ext)
                end
                if !is_1V && !is_1D && input.animate_log_vs_vzeta_r
                    animate_vs_vzeta_r(run_info, variable_name, is=is, input=input,
                                       outfile=log_variable_prefix * "vs_vzeta_r." *
                                       input.animation_ext, colorscale=log10,
                                       transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vzeta_z
                    animate_vs_vzeta_z(run_info, variable_name, is=is, input=input,
                                       outfile=variable_prefix * "vs_vzeta_z." *
                                       input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vzeta_z
                    animate_vs_vzeta_z(run_info, variable_name, is=is, input=input,
                                       outfile=log_variable_prefix * "vs_vzeta_z." *
                                       input.animation_ext, colorscale=log10,
                                       transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vr
                    animate_vs_vr(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vr." * input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vr
                    animate_vs_vr(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vr." * input.animation_ext,
                                  yscale=log10, transform=positive_or_nan)
                end
                if !is_1V && !is_1D && input.animate_vs_vr_r
                    animate_vs_vr_r(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vr_r." * input.animation_ext)
                end
                if !is_1V && !is_1D && input.animate_log_vs_vr_r
                    animate_vs_vr_r(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vr_r." * input.animation_ext,
                                    colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vr_z
                    animate_vs_vr_z(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vr_z." * input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vr_z
                    animate_vs_vr_z(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vr_z." * input.animation_ext,
                                    colorscale=log10, transform=positive_or_nan)
                end
                if input.animate_vs_vz
                    animate_vs_vz(run_info, variable_name, is=is, input=input,
                                  outfile=variable_prefix * "vs_vz." * input.animation_ext)
                end
                if input.animate_log_vs_vz
                    animate_vs_vz(run_info, variable_name, is=is, input=input,
                                  outfile=log_variable_prefix * "vs_vz." * input.animation_ext,
                                  yscale=log10, transform=positive_or_nan)
                end
                if !is_1D && input.animate_vs_vz_r
                    animate_vs_vz_r(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vz_r." * input.animation_ext)
                end
                if !is_1D && input.animate_log_vs_vz_r
                    animate_vs_vz_r(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vz_r." * input.animation_ext,
                                    colorscale=log10, transform=positive_or_nan)
                end
                if input.animate_vs_vz_z
                    animate_vs_vz_z(run_info, variable_name, is=is, input=input,
                                    outfile=variable_prefix * "vs_vz_z." * input.animation_ext)
                end
                if input.animate_log_vs_vz_z
                    animate_vs_vz_z(run_info, variable_name, is=is, input=input,
                                    outfile=log_variable_prefix * "vs_vz_z." * input.animation_ext,
                                    colorscale=log10, transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vr_vzeta
                    animate_vs_vr_vzeta(run_info, variable_name, is=is, input=input,
                                        outfile=variable_prefix * "vs_vr_vzeta." *
                                        input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vr_vzeta
                    animate_vs_vr_vzeta(run_info, variable_name, is=is, input=input,
                                        outfile=log_variable_prefix * "vs_vr_vzeta." *
                                        input.animation_ext, colorscale=log10,
                                        transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vz_vzeta
                    animate_vs_vz_vzeta(run_info, variable_name, is=is, input=input,
                                        outfile=variable_prefix * "vs_vz_vzeta." *
                                        input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vz_vzeta
                    animate_vs_vz_vzeta(run_info, variable_name, is=is, input=input,
                                        outfile=log_variable_prefix * "vs_vz_vzeta." *
                                        input.animation_ext, colorscale=log10,
                                        transform=positive_or_nan)
                end
                if !is_1V && input.animate_vs_vz_vr
                    animate_vs_vz_vr(run_info, variable_name, is=is, input=input,
                                     outfile=variable_prefix * "vs_vz_vr." *
                                     input.animation_ext)
                end
                if !is_1V && input.animate_log_vs_vz_vr
                    animate_vs_vz_vr(run_info, variable_name, is=is, input=input,
                                     outfile=log_variable_prefix * "vs_vz_vr." *
                                     input.animation_ext, colorscale=log10,
                                     transform=positive_or_nan)
                end
            end
        end
    end

    return nothing
end

# Generate 1d plot functions for each dimension
for dim ∈ (:t, setdiff(all_dimensions, (:s, :sn))...)
    function_name_str = "plot_vs_$dim"
    function_name = Symbol(function_name_str)
    dim_str = String(dim)
    idim = Symbol(:i, dim)
    range_name = Symbol(dim, :_range)
    eval(quote
             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, yscale=nothing,
                                     transform=identity, $range_name=nothing, kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end

                     n_runs = length(run_info)

                     fig, ax = get_1d_ax(xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale)
                     for (d, ri) ∈ zip(data, run_info)
                         $function_name(ri, var_name, is=is, data=d, input=input, ax=ax,
                                        transform=transform, $range_name=$range_name,
                                        label=ri.run_name, kwargs...)
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
                                     input=nothing, ax=nothing, label=nothing,
                                     outfile=nothing, transform=identity,
                                     $range_name=nothing, kwargs...)
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices($(QuoteNode(dim));
                                                              input=input, is=is)
                     data = postproc_load_variable(run_info, var_name; $idim=$range_name,
                                                   dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim)); input=input, is=is,
                                         $idim=$range_name)
                 end

                 # Use transform to allow user to do something like data = abs.(data)
                 data = transform.(data)

                 x = run_info.$dim.grid
                 if $range_name !== nothing
                     x = x[$range_name]
                 end
                 fig = plot_1d(x, data, xlabel="$($dim_str)",
                               ylabel=get_variable_symbol(var_name), label=label, ax=ax,
                               kwargs...)

                 if outfile !== nothing
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save(outfile, fig)
                 end

                 return nothing
             end
         end)
end

# Generate 2d plot functions for all combinations of dimensions
const dimension_combinations_2d = Tuple(
         Tuple(c) for c in
         unique((combinations((:t, setdiff(ion_dimensions, (:s,))...), 2)...,
                 combinations((:t, setdiff(neutral_dimensions, (:sn,))...), 2)...)))
for (dim1, dim2) ∈ dimension_combinations_2d
    function_name_str = "plot_vs_$(dim2)_$(dim1)"
    function_name = Symbol(function_name_str)
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
    range_name1 = Symbol(dim1, :_range)
    range_name2 = Symbol(dim2, :_range)
    eval(quote
             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, transform=identity,
                                     $range_name1=nothing, $range_name2=nothing,
                                     kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end
                     fig, ax, colorbar_places = get_2d_ax(length(run_info),
                                                          title=get_variable_symbol(var_name))
                     for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
                         $function_name(ri, var_name; is=is, data=d, input=input, ax=a,
                                        transform=transform, $range_name1=$range_name1,
                                        $range_name2=$range_name2, colorbar_place=cp,
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
                                     colorbar_place=colorbar_place, title=nothing,
                                     outfile=nothing, transform=identity,
                                     $range_name1=nothing, $range_name2=nothing,
                                     kwargs...)
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices($(QuoteNode(dim1)),
                                                              $(QuoteNode(dim2));
                                                              input=input, is=is)
                     data = postproc_load_variable(run_info, var_name;
                                                   $idim1=$range_name1,
                                                   $idim2=$range_name2, dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim2)), $(QuoteNode(dim1));
                                         input=input, is=is, $idim1=$range_name1,
                                         $idim2=$range_name2)
                 end
                 if input === nothing
                     colormap = "reverse_deep"
                 else
                     colormap = input.colormap
                 end
                 if title === nothing
                     title = get_variable_symbol(var_name)
                 end

                 # Use transform to allow user to do something like data = abs.(data)
                 data = transform.(data)

                 x = $dim2_grid
                 if $range_name2 !== nothing
                     x = x[$range_name2]
                 end
                 y = $dim1_grid
                 if $range_name1 !== nothing
                     y = y[$range_name1]
                 end
                 fig = plot_2d(x, y, data; xlabel="$($dim2_str)", ylabel="$($dim1_str)",
                               title=title, ax=ax, colorbar_place=colorbar_place,
                               colormap=colormap, kwargs...)

                 if outfile !== nothing
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save(outfile, fig)
                 end

                 return nothing
             end
         end)
end

# Generate 1d animation functions for each dimension
for dim ∈ setdiff(all_dimensions, (:s, :sn))
    function_name_str = "animate_vs_$dim"
    function_name = Symbol(function_name_str)
    dim_str = String(dim)
    idim = Symbol(:i, dim)
    range_name = Symbol(dim, :_range)
    eval(quote
             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, yscale=nothing,
                                     transform=identity, $range_name=nothing,
                                     ylims=nothing, kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end

                     n_runs = length(run_info)

                     fig, ax = get_1d_ax(xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale)
                     frame_index = Observable(1)

                     for (d, ri) ∈ zip(data, run_info)
                         $function_name(ri, var_name; is=is, data=d, input=input,
                                        transform=transform, $range_name=$range_name,
                                        ylims=ylims, frame_index=frame_index, ax=ax,
                                        kwargs...)
                     end
                     if n_runs > 1
                         put_legend_above(fig, ax)
                     end

                     nt = minimum(ri.nt for ri ∈ run_info)
                     save_animation(fig, frame_index, nt, outfile)

                     return fig
                 catch e
                     println("$($function_name_str)() failed for $var_name, is=$is. Error was $e")
                     return nothing
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     transform=identity, $range_name=nothing,
                                     outfile=nothing, yscale=nothing, ylims=nothing,
                                     kwargs...)
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices(:t, $(QuoteNode(dim));
                                                              input=input, is=is)
                     data = postproc_load_variable(run_info, var_name; $idim=$range_name,
                                                   dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim)), :t; input=input, is=is,
                                         $idim=$range_name)
                 end
                 if frame_index === nothing
                     ind = Observable(1)
                 else
                     ind = frame_index
                 end
                 if ax === nothing
                     fig, ax = get_1d_ax(xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale)
                 else
                     fig = nothing
                 end

                 # Use transform to allow user to do something like data = abs.(data)
                 data = transform.(data)

                 nt = size(data, 2)

                 x = run_info.$dim.grid
                 if $range_name !== nothing
                     x = x[$range_name]
                 end
                 animate_1d(x, data; ax=ax, ylims=ylims, frame_index=ind,
                            label=run_info.run_name, kwargs...)

                 if frame_index === nothing
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save_animation(fig, ind, nt, outfile)
                 end

                 return fig
             end
         end)
end

# Generate 2d animation functions for all combinations of dimensions
const dimension_combinations_2d_no_t = Tuple(
          Tuple(c) for c in unique((combinations(setdiff(ion_dimensions, (:s,)), 2)...,
                                    combinations(setdiff(neutral_dimensions, (:sn,)), 2)...)))
for (dim1, dim2) ∈ dimension_combinations_2d_no_t
    function_name_str = "animate_vs_$(dim2)_$(dim1)"
    function_name = Symbol(function_name_str)
    dim1_str = String(dim1)
    dim2_str = String(dim2)
    dim1_grid = :( run_info.$dim1.grid )
    dim2_grid = :( run_info.$dim2.grid )
    idim1 = Symbol(:i, dim1)
    idim2 = Symbol(:i, dim2)
    range_name1 = Symbol(dim1, :_range)
    range_name2 = Symbol(dim2, :_range)
    eval(quote
             function $function_name(run_info::Tuple, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, transform=identity,
                                     $range_name1=nothing, $range_name2=nothing,
                                     kwargs...)

                 try
                     if data === nothing
                         data = Tuple(nothing for _ in run_info)
                     end
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end

                     fig, ax, colorbar_places = get_2d_ax(length(run_info),
                                                          title=get_variable_symbol(var_name))
                     frame_index = Observable(1)

                     for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
                         $function_name(ri, var_name; is=is, data=d, input=input,
                                        transform=transform, $range_name1=$range_name1,
                                        $range_name2=$range_name2,
                                        frame_index=frame_index, ax=a, colorbar_place=cp,
                                        title=ri.run_name, kwargs...)
                     end

                     nt = minimum(ri.nt for ri ∈ run_info)
                     save_animation(fig, frame_index, nt, outfile)

                     return fig
                 catch e
                     println("$($function_name_str) failed for $var_name, is=$is. Error was $e")
                     return nothing
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     transform=identity, $range_name1=nothing,
                                     $range_name2=nothing, colorbar_place=colorbar_place,
                                     title=nothing, outfile=nothing, kwargs...)
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices(:t, $(QuoteNode(dim1)),
                                                              $(QuoteNode(dim2));
                                                              input=input, is=is)
                     data = postproc_load_variable(run_info, var_name;
                                                   $idim1=$range_name1,
                                                   $idim2=$range_name2, dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim2)), $(QuoteNode(dim1)), :t;
                                         input=input, is=is, $idim1=$range_name1,
                                         $idim2=$range_name2)
                 end
                 if input === nothing
                     colormap = "reverse_deep"
                 else
                     colormap = input.colormap
                 end
                 if title === nothing
                     title = get_variable_symbol(var_name)
                 end

                 # Use transform to allow user to do something like data = abs.(data)
                 data = transform.(data)

                 x = $dim2_grid
                 if $range_name2 !== nothing
                     x = x[$range_name2]
                 end
                 y = $dim1_grid
                 if $range_name1 !== nothing
                     y = y[$range_name1]
                 end
                 fig = animate_2d(x, y, data; xlabel="$($dim2_str)",
                                  ylabel="$($dim1_str)", title=title,
                                  frame_index=frame_index, ax=ax,
                                  colorbar_place=colorbar_place, colormap=colormap,
                                  kwargs...)

                 if frame_index === nothing
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     nt = size(data, 3)
                     save_animation(fig, ind, nt, outfile)
                 end

                 return fig
             end
         end)
end

function get_1d_ax(n=nothing; title=nothing, yscale=nothing, kwargs...)
    if yscale !== nothing
        kwargs = tuple(kwargs..., :yscale=>yscale)
    end
    if n == nothing
        fig = Figure(resolution=(600, 400))
        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)

            ax = Axis(fig[2,1]; kwargs...)
        else
            ax = Axis(fig[1,1]; kwargs...)
        end
    else
        fig = Figure(resolution=(600*n, 400))

        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)

            plot_layout = fig[2,1] = GridLayout()
        else
            plot_layout = fig[1,1] = GridLayout()
        end

        ax = [Axis(plot_layout[1,i]; kwargs...) for i in 1:n]
        if yscale !== nothing
            for a ∈ ax
                a.yscale = yscale
            end
        end
    end

    return fig, ax
end

function get_2d_ax(n=nothing; title=nothing, kwargs...)
    if n == nothing
        fig = Figure(resolution=(600, 400))
        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)
            irow = 2
        else
            irow = 1
        end
        ax = Axis(fig[irow,1]; kwargs...)
        colorbar_places = fig[irow,2]
    else
        fig = Figure(resolution=(600*n, 400))

        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)

            plot_layout = fig[2,1] = GridLayout()
        else
            plot_layout = fig[1,1] = GridLayout()
        end
        ax = [Axis(plot_layout[1,2*i-1]; kwargs...) for i in 1:n]
        colorbar_places = [plot_layout[1,2*i] for i in 1:n]
    end

    return fig, ax, colorbar_places
end

function plot_1d(xcoord, data; ax=nothing, xlabel=nothing,
                 ylabel=nothing, title=nothing, yscale=nothing, kwargs...)
    if ax === nothing
        fig, ax = get_1d_ax()
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

    l = lines!(ax, xcoord, data; kwargs...)

    if ax === nothing
        return fig
    else
        return l
    end
end

function plot_2d(xcoord, ycoord, data; ax=nothing, colorbar_place=nothing, xlabel=nothing,
                 ylabel=nothing, title=nothing, colormap="reverse_deep", kwargs...)
    if ax === nothing
        fig, ax, colorbar_place = get_2d_ax()
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

    # Convert grid point values to 'cell face' values for heatmap
    xcoord = grid_points_to_faces(xcoord)
    ycoord = grid_points_to_faces(ycoord)

    hm = heatmap!(ax, xcoord, ycoord, data; kwargs...)
    if colorbar_place === nothing
        println("Warning: colorbar_place argument is required to make a color bar")
    else
        Colorbar(colorbar_place, hm)
    end

    if ax === nothing
        return fig
    else
        return nothing
    end
end

function animate_1d(xcoord, data; frame_index=nothing, ax=nothing, fig=nothing,
                    xlabel=nothing, ylabel=nothing, title=nothing, yscale=nothing,
                    ylims=nothing, outfile=nothing, kwargs...)

    if frame_index === nothing
        ind = Observable(1)
    else
        ind = frame_index
    end

    if ax === nothing
        fig, ax = get_1d_ax(title=title, xlabel=xlabel, ylabel=ylabel, yscale=yscale)
    end

    if ylims === nothing
        datamin, datamax = NaNMath.extrema(data)
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

    line_data = @lift(@view data[:,$ind])
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

function animate_2d(xcoord, ycoord, data; frame_index=nothing, ax=nothing, fig=nothing,
                    colorbar_place=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
                    outfile=nothing, colormap="reverse_deep", kwargs...)
    colormap = parse_colormap(colormap)

    if ax === nothing
        fig, ax, colorbar_place = get_2d_ax()
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
    if title !== nothing
        ax.title = title
    end

    xcoord = grid_points_to_faces(xcoord)
    ycoord = grid_points_to_faces(ycoord)
    heatmap_data = @lift(@view data[:,:,$ind])
    hm = heatmap!(ax, xcoord, ycoord, heatmap_data; colormap=colormap, kwargs...)
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

function save_animation(fig, frame_index, nt, outfile)
    record(fig, outfile, 1:nt, framerate=5) do it
        frame_index[] = it
    end
    return nothing
end

function put_legend_above(fig, ax; kwargs...)
    return Legend(fig[0,1], ax; tellheight=true, tellwidth=false, kwargs...)
end

function put_legend_below(fig, ax; kwargs...)
    return Legend(fig[end+1,1], ax; tellheight=true, tellwidth=false, kwargs...)
end

function put_legend_left(fig, ax; kwargs...)
    return Legend(fig[end,0], ax; kwargs...)
end

function put_legend_right(fig, ax; kwargs...)
    return Legend(fig[end,end+1], ax; kwargs...)
end

function select_slice(variable::AbstractArray{T,1}, dims::Symbol...; input=nothing, is=nothing) where T
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

function select_slice(variable::AbstractArray{T,2}, dims::Symbol...; input=nothing, is=nothing) where T
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

function select_slice(variable::AbstractArray{T,3}, dims::Symbol...; input=nothing, is=nothing) where T
    # Array is (z,r,t)

    if length(dims) > 3
        error("Tried to get a slice of 3d variable with dimensions $dims")
    end

    if input === nothing || :it0 ∉ input
        it0 = size(variable, 3)
    else
        it0 = input.it0
    end
    if input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 2) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims
        slice = selectdim(slice, 3, it0)
    end
    if :r ∉ dims
        slice = selectdim(slice, 2, ir0)
    end
    if :z ∉ dims
        slice = selectdim(slice, 1, iz0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,4}, dims::Symbol...; input=nothing, is=1) where T
    # Array is (z,r,species,t)

    if input === nothing || :it0 ∉ input
        it0 = size(variable, 4)
    else
        it0 = input.it0
    end
    if input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 2) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims
        slice = selectdim(slice, 4, it0)
    end
    slice = selectdim(slice, 3, is)
    if :r ∉ dims
        slice = selectdim(slice, 2, ir0)
    end
    if :z ∉ dims
        slice = selectdim(slice, 1, iz0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,6}, dims::Symbol...; input=nothing, is=1) where T
    # Array is (z,r,species,t)

    if input === nothing || :it0 ∉ input
        it0 = size(variable, 6)
    else
        it0 = input.it0
    end
    if input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 4) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 3) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if input === nothing || :ivpa0 ∉ input
        ivpa0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivpa0 = input.ivpa0
    end
    if input === nothing || :ivperp0 ∉ input
        ivperp0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivperp0 = input.ivperp0
    end

    slice = variable
    if :t ∉ dims
        slice = selectdim(slice, 6, it0)
    end
    slice = selectdim(slice, 5, is)
    if :r ∉ dims
        slice = selectdim(slice, 4, ir0)
    end
    if :z ∉ dims
        slice = selectdim(slice, 3, iz0)
    end
    if :vperp \nin∉ dims
        slice = selectdim(slice, 2, ivperp0)
    end
    if :vpa \nin  ∉ dims
        slice = selectdim(slice, 1, ivpa0)
    end

    return slice
end

function select_slice(variable::AbstractArray{T,7}, dims::Symbol...; input=nothing, is=1) where T
    # Array is (z,r,species,t)

    if input === nothing || :it0 ∉ input
        it0 = size(variable, 7)
    else
        it0 = input.it0
    end
    if input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 5) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 4) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if input === nothing || :ivzeta0 ∉ input
        ivzeta0 = max(size(variable, 3) ÷ 3, 1)
    else
        ivzeta0 = input.ivzeta0
    end
    if input === nothing || :ivr0 ∉ input
        ivr0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivr0 = input.ivr0
    end
    if input === nothing || :ivz0 ∉ input
        ivz0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivz0 = input.ivz0
    end

    slice = variable
    if :t ∉ dims
        slice = selectdim(slice, 7, it0)
    end
    slice = selectdim(slice, 6, is)
    if :r ∉ dims
        slice = selectdim(slice, 5, ir0)
    end
    if :z ∉ dims
        slice = selectdim(slice, 4, iz0)
    end
    if :vzeta ∉ dims
        slice = selectdim(slice, 3, ivzeta0)
    end
    if :vr ∉ dims
        slice = selectdim(slice, 2, ivr0)
    end
    if :vz ∉ dims
        slice = selectdim(slice, 1, ivz0)
    end

    return slice
end

"""
Turn grid points into 'cell faces'

Returns `faces`, which has a length one greater than `coord`. The first and last values of
`faces` are the first and last values of `coord`. The intermediate values are the mid
points between grid points.
"""
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

"""
Make plots/animations of the charged particle distribution function at wall boundaries
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

    println("Making plots of f at walls")

    z_lower = 1
    z_upper = run_info[1].z.n
    if !all(ri.z.n == z_upper for ri ∈ run_info)
        println("Cannot run plot_charged_pdf_2D_at_wall() for runs with different "
                * "z-grid sizes. Got $(Tuple(ri.z.n for ri ∈ run_info))")
        return nothing
    end

    is_1D = all(ri !== nothing && ri.r.n == 1 for ri ∈ run_info)
    is_1V = all(ri !== nothing && ri.vperp.n == 1 && ri.vzeta.n == 1 && ri.vr.n == 1
                for ri ∈ run_info)

    for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+8, "wall-"),
                               (z_upper, z_upper-8:z_upper, "wall+"))
        f_input = copy(input_dict_dfns["f"])
        f_input["iz0"] = z

        if input.plot
            plot_vs_vpa(run_info, "f"; is=1, input=f_input,
                        outfile=plot_prefix * "pdf_$(label)_vs_vpa.pdf")

            if !is_1V
                plot_vs_vpa_vperp(run_info, "f"; is=1, input=f_input,
                                  outfile=plot_prefix * "pdf_$(label)_vs_vpa_vperp.pdf")
            end

            plot_vs_vpa_z(run_info, "f"; is=1, input=f_input, z_range=z_range,
                          outfile=plot_prefix * "pdf_$(label)_vs_vpa_z.pdf")

            if !is_1D
                plot_vs_z_r(run_info, "f"; is=1, input=f_input, z_range=z_range,
                            outfile=plot_prefix * "pdf_$(label)_vs_z_r.pdf")

                plot_vs_vpa_r(run_info, "f"; is=1, input=f_input,
                              outfile=plot_prefix * "pdf_$(label)_vs_vpa_r.pdf")
            end
        end

        if input.animate
            animate_vs_vpa(run_info, "f"; is=1, input=f_input,
                           outfile=plot_prefix * "pdf_$(label)_vs_vpa." * input.animation_ext)

            if !is_1V
                animate_vs_vpa_vperp(run_info, "f"; is=1, input=f_input,
                                     outfile=plot_prefix * "pdf_$(label)_vs_vpa_vperp." * input.animation_ext)
            end

            animate_vs_vpa_z(run_info, "f"; is=1, input=f_input, z_range=z_range,
                             outfile=plot_prefix * "pdf_$(label)_vs_vpa_z." * input.animation_ext)

            if !is_1D
                animate_vs_z_r(run_info, "f"; is=1, input=f_input, z_range=z_range,
                               outfile=plot_prefix * "pdf_$(label)_vs_z_r." * input.animation_ext)

                animate_vs_vpa_r(run_info, "f"; is=1, input=f_input,
                                 outfile=plot_prefix * "pdf_$(label)_vs_vpa_r." * input.animation_ext)
            end
        end
    end

    return nothing
end

"""
Plot the criterion from the Chodura condition at the sheath boundaries
"""
function Chodura_condition_plots(run_info::Tuple; plot_prefix, ax_lower=nothing,
                                 ax_upper=nothing)
    input = Dict_to_NamedTuple(input_dict_dfns["Chodura_condition"])

    println("Making Chodura condition plots")

    if !any(v for (k,v) ∈ pairs(input) if startswith(String(k), "plot"))
        # No plots to make here
        return nothing
    end
    if !any(ri !== nothing for ri ∈ run_info)
        println("Warning: no distribution function output, skipping Chodura "
                * "condition plots")
        return nothing
    end

    if n_runs == 1
        Chodura_condition_plots(run_info[1], plot_prefix=plot_prefix)
        return nothing
    end

    n_runs = length(run_info)

    figs = []
    axes = ([] for _ ∈ run_info)
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
        outfile = string(plot_prefix, "_Chodura_ratio_lower_vs_t.pdf")
        save(outfile, fig)

        fig = figs[2]
        ax = axes[2][1]
        put_legend_right(fig, ax)
        outfile = string(plot_prefix, "_Chodura_ratio_upper_vs_t.pdf")
        save(outfile, fig)
    end
    if input.plot_vs_r_t
        fig = figs[3]
        outfile = string(plot_prefix, "_Chodura_ratio_lower_vs_r_t.pdf")
        save(outfile, fig)

        fig = figs[4]
        outfile = string(plot_prefix, "_Chodura_ratio_upper_vs_r_t.pdf")
        save(outfile, fig)
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

    tinds = input_dict_dfns["itime_min"]:input_dict_dfns["itime_skip"]:input_dict_dfns["itime_max"]

    time = run_info.time
    density = postproc_load_variable(run_info, "density"; it=tinds)
    Er = postproc_load_variable(run_info, "Er"; it=tinds)
    f_lower = postproc_load_variable(run_info, "f"; it=tinds, iz=1)
    f_upper = postproc_load_variable(run_info, "f"; it=tinds, iz=run_info.z.n_global)

    Chodura_ratio_lower, Chodura_ratio_upper =
        check_Chodura_condition(run_info.r_local, run_info.z_local, run_info.vperp,
                                run_info.vpa, density, run_info.composition.T_e, Er,
                                run_info.geometry, run_info.z.bc, nothing;
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
            outfile = string(plot_prefix, "_Chodura_ratio_lower_vs_t.pdf")
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
            outfile = string(plot_prefix, "_Chodura_ratio_upper_vs_t.pdf")
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
            ax, colorbar_place = axes[3]
            title = run_info.run_name
        end
        plot_2d(run_info.r.grid, time, Chodura_ratio_lower, ax=ax,
                colorbar_place=colorbar_place, title=title)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "_Chodura_ratio_lower_vs_r_t.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax, colorbar_place = get_2d_ax(title="Chodura ratio at z=+L/2",
                                                xlabel="r", ylabel="time")
            title = nothing
        else
            fig = nothing
            ax, colorbar_place = axes[4]
            title = run_info.run_name
        end
        plot_2d(run_info.r.grid, time, Chodura_ratio_upper, ax=ax,
                colorbar_place=colorbar_place, title=title)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "_Chodura_ratio_upper_vs_r_t.pdf")
            save(outfile, fig)
        end
    end

    return nothing
end

function sound_wave_plots(run_info::Tuple; plot_prefix)
    input = Dict_to_NamedTuple(input_dict["sound_wave_fit"])

    println("Making sound wave fit plots")

    if !input.calculate_frequency && !input.plot
        return nothing
    end

    #try
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
    #catch e
    #    println("Error in sound_wave_plots(). Error was ", e)
    #end

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

    tinds = input_dict["itime_min"]:input_dict["itime_skip"]:input_dict["itime_max"]
    time = run_info.time

    # This analysis is only designed for 1D cases, so only use phi[:,ir0,:]
    if phi === nothing
        phi = postproc_load_variable(run_info, "phi"; it=tinds, ir=input.ir0)
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

        @views lines!(ax, time, abs.(delta_phi[input.iz0,:]), label=delta_phi_label)

        if input.calculate_frequency
            @views lines!(ax, time, abs.(fitted_delta_phi), label=fit_label)
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
Make plots for analysis of 2D instability

If `zind` is not passed, it is calculated as the z-index where the mode seems to have
the maximum growth rate for this variable.
Returns `zind`.
"""
function instability2D_plots(run_info::Tuple, variable_name; plot_prefix, zind)
    println("Making instability plots for $variable_name")
    flush(stdout)

    n_runs = length(run_info)
    var_symbol = get_variable_symbol(variable_name)
    instability2D_options = Dict_to_NamedTuple(input_dict["instability2D"])

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
        println("Saving 2D Fourier components movie for $variable_name")
        frame_index = axes_and_observables[1][4][3]
        nt = minimum(ri.nt for ri ∈ run_info)
        outfile = plot_prefix * variable_name * "_Fourier." *
                  instability2D_options.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    fig = figs[5]
    if fig !== nothing
        println("Saving 2D perturbation movie for $variable_name")
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

    tinds = input_dict["itime_min"]:input_dict["itime_skip"]:input_dict["itime_max"]
    time = run_info.time

    if variable_name == "temperature"
        variable = postproc_load_variable(run_info, "thermal_speed"; it=tinds).^2
    else
        variable = postproc_load_variable(run_info, variable_name; it=tinds)
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
        println("Doing 2D Fourier analysis for $variable_name")
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
            println("making perturbation movie $variable_name")
            flush(stdout)
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

"""
Get indices for dimensions to slice

The indices are taken from `input`, unless they are passed as keyword arguments

The dimensions in `keep_dims` are not given a slice (those are the dimensions we want in
the variable after slicing).
"""
function get_dimension_slice_indices(keep_dims...; input, slice_indices...)
    if isa(input, AbstractDict)
        input = Dict_to_NamedTuple(input)
    end
    slice_names = union((:t,), setdiff(all_dimensions, (:sn,)))
    return Tuple(Symbol(:i, sn)=>get(slice_indices, sn, input[Symbol(:i, sn, :0)]) for sn ∈ slice_names if sn ∉ keep_dims)
end

"""
Get a symbol corresponding to a variable name

If the symbol has not been defined, just return the variable name
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
Parse colormap option

Allows us to have a string option and still use Reverse, etc. conveniently
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

# Utility functions
###################
#
# These are more-or-less generic, but only used in this module for now, so keep them here.

"""
Remove all entries from a Dict, leaving it empty
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
If the argument is zero or negative, replace it with NaN
"""
function positive_or_nan(x)
    return x > 0 ? x : NaN
end

end
