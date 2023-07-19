"""
Post processing functions using Makie.jl
"""
module makie_post_processing

export makie_post_process

using ..array_allocation: allocate_float
using ..coordinates: define_coordinate
using ..input_structs: grid_input, advection_input
using ..moment_kinetics_input: set_defaults_and_check_top_level!,
                               set_defaults_and_check_section!, Dict_to_NamedTuple
using ..load_data: open_readonly_output_file, get_group, load_block_data,
                   load_coordinate_data, load_input, load_mk_options, load_species_data,
                   load_time_data
using ..post_processing: construct_global_zr_coords
using ..type_definitions: mk_float, mk_int

using Glob
using LsqFit
using MPI
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

const em_variables = ("phi", "Er", "Ez")
const ion_moment_variables = ("density", "parallel_flow", "parallel_pressure",
                              "thermal_speed", "parallel_heat_flux", )
const neutral_moment_variables = ("density_neutral", "uz_neutral", "pz_neutral",
                                  "thermal_speed_neutral", "qz_neutral")
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

    run_label = Tuple(basename(r) for r ∈ run_dir)

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
        # Default to plotting moments from 'moments' files
        run_info = run_info_moments
    else
        # Fall back to trying to plot from 'dfns' files if those are all we have
        run_info = run_info_dfns
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
        for variable_name ∈ all_dfn_variables
            #plots_for_dfn_variable(run_info_dfns, variable_name, plot_prefix=plot_prefix,
            #                       is_1D=is_1D, is_1V=is_1V)
        end
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

    original_input = deepcopy(input_dict)

    # Set up input_dict with all-default parameters
    setup_makie_post_processing_input!(Dict{String,Any}())

    # Convert input_dict to a String formatted as the contents of a TOML file
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

    # Restore original state of input_dict
    clear_Dict!(input_dict)
    merge!(input_dict, original_input)

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
        input_file::String=default_input_file_name; run_info_moment=nothing,
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
                                            run_info_moments=run_info_moments,
                                            run_info_dfns=run_info_dfns)
    # Remove all existing entries from the global `input_dict`
    clear_Dict!(input_dict)

    # Put entries from new_input_dict into input_dict
    merge!(input_dict, new_input_dict)

    if !isa(run_info_moments, Tuple)
        # Make sure run_info is a Tuple
        run_info_moments = (run_info_moments,)
    end
    if !isa(run_info_dfns, Tuple)
        # Make sure run_info is a Tuple
        run_info_dfns = (run_info_dfns,)
    end

    has_moments = any(ri !== nothing for ri ∈ run_info_moments)
    has_dfns = any(ri !== nothing for ri ∈ run_info_dfns)

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

    if has_moments
        nt_unskipped_min = minimum(ri.nt_unskipped for ri in run_info_moments
                                                   if ri !== nothing)
        nt_min = minimum(ri.nt for ri in run_info_moments if ri !== nothing)
        nr_min = minimum(ri.r.n for ri in run_info_moments if ri !== nothing)
        nz_min = minimum(ri.z.n for ri in run_info_moments if ri !== nothing)
    else
        nt_unskipped_min = 1
        nt_min = 1
        nr_min = 1
        nz_min = 1
    end
    if has_dfns
        nt_unskipped_dfns_min = minimum(ri.nt_unskipped for ri in run_info_dfns
                                                             if ri !== nothing)
        nt_dfns_min = minimum(ri.nt for ri in run_info_dfns if ri !== nothing)
        nvperp_min = minimum(ri.vperp.n for ri in run_info_dfns if ri !== nothing)
        nvpa_min = minimum(ri.vpa.n for ri in run_info_dfns if ri !== nothing)
        nvzeta_min = minimum(ri.vzeta.n for ri in run_info_dfns if ri !== nothing)
        nvr_min = minimum(ri.vr.n for ri in run_info_dfns if ri !== nothing)
        nvz_min = minimum(ri.vz.n for ri in run_info_dfns if ri !== nothing)
    else
        nt_unskipped_dfns_min = 1
        nt_dfns_min = 1
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
    only_global_options = ("itime_min", "itime_max", "itime_skip", "itime_skip_dfns")

    set_defaults_and_check_top_level!(input_dict;
       # Options that only apply at the global level (not per-variable)
       ################################################################
       # Options that provide the defaults for per-variable settings
       #############################################################
       colormap="reverse_deep",
       # Slice t to this value when making time-independent plots
       it0=nt_min,
       it0_dfns=nt_dfns_min,
       # Slice r to this value when making reduced dimensionality plots
       ir0=max(nr_min÷3, 1),
       # Slice z to this value when making reduced dimensionality plots
       iz0=max(nz_min÷3, 1),
       # Slice vperp to this value when making reduced dimensionality plots
       ivperp0=max(nvperp_min ÷ 3, 1),
       # Slice vpa to this value when making reduced dimensionality plots
       ivpa0=max(nvpa_min ÷ 3, 1),
       # Slice vzeta to this value when making reduced dimensionality plots
       ivzeta0=max(nvzeta_min ÷ 3, 1),
       # Slice vr to this value when making reduced dimensionality plots
       ivr0=max(nvr_min ÷ 3, 1),
       # Slice vz to this value when making reduced dimensionality plots
       ivz0=max(nvz_min ÷ 3, 1),
       # Time index to start from
       itime_min=1,
       # Time index to end at
       itime_max=nt_unskipped_min,
       # Load every `time_skip` time points for EM and moment variables, to save memory
       itime_skip=1,
       # Time index to start from for distribution functions
       itime_min_dfns=1,
       # Time index to end at for distribution functions
       itime_max_dfns=nt_unskipped_dfns_min,
       # Load every `time_skip` time points for distribution function variables, to save
       # memory
       itime_skip_dfns=1,
       plot_vs_z_t=true,
       animate_vs_z=true,
      )

    for variable_name ∈ all_variables
        set_defaults_and_check_section!(
            input_dict, variable_name;
            OrderedDict(Symbol(k)=>v for (k,v) ∈ input_dict
                        if !isa(v, AbstractDict) && !(k ∈ only_global_options))...)
    end


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

    has_data = all(length(glob(p * ".$ext*.h5")) > 0 for p ∈ run_prefixes)
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
    if parallel_io
        files = fids0
        nt_unskipped, time = load_time_data(files)
        if itime_max > 0
            time = time[itime_min:itime_skip:itime_max]
        else
            time = time[itime_min:itime_skip:end]
        end
        nt = length(time)

        # Get input and coordinates from the final restart
        file_final_restart = files[end]

        input = load_input(file_final_restart)

        n_ion_species, n_neutral_species = load_species_data(file_final_restart)
        evolve_density, evolve_upar, evolve_ppar =
            load_mk_options(file_final_restart)

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
    else
        error("parallel_io=false not implemented yet")
        files = run_prefixes
    end

    if dfns
        return (run_name=run_name, parallel_io=parallel_io, nblocks=nblocks, files=files,
                input=input, n_ion_species=n_ion_species,
                n_neutral_species=n_neutral_species, nt=nt, nt_unskipped=nt_unskipped,
                time=time, r=r, z=z, vperp=vperp, vpa=vpa, vzeta=vzeta, vr=vr, vz=vz,
                r_local=r_local, z_local=z_local, r_spectral=r_spectral,
                z_spectral=z_spectral, vperp_spectral=vperp_spectral,
                vpa_spectral=vpa_spectral, vzeta_spectral=vzeta_spectral,
                vr_spectral=vr_spectral, vz_spectral=vz_spectral)
    else
        return (run_name=run_name, parallel_io=parallel_io, nblocks=nblocks, files=files,
                input=input, n_ion_species=n_ion_species,
                n_neutral_species=n_neutral_species, nt=nt, nt_unskipped=nt_unskipped,
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

    if run_info.parallel_io
        # Get HDF5/NetCDF variables directly and load slices
        variable = Tuple(get_group(f, "dynamic_data")[variable_name]
                         for f ∈ run_info.files)
        nd = ndims(variable[1])

        if isa(it, mk_int)
            nt = 1
        elseif it === nothing
            it = 1:nt
        else
            nt = length(it)
        end

        if nd == 3
            # EM variable with dimensions (z,r,t)
            not_allowed_slices = (ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                                  ivz=ivz)
            if any(i !== nothing for i ∈ values(not_allowed_slices))
                error("Got slice for non-existing dimension of 2d variable. "
                      * "All of $not_allowed_slices should be `nothing`.")
            end
            dims = Vector{mk_int}()
            iz === nothing && push!(dims, run_info.z.n)
            ir === nothing && push!(dims, run_info.r.n)
            push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 4
            # moment variable with dimensions (z,r,s,t)
            not_allowed_slices = (ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                                  ivz=ivz)
            if any(i !== nothing for i ∈ values(not_allowed_slices))
                error("Got slice for non-existing dimension of 2d variable. "
                      * "All of $not_allowed_slices should be `nothing`.")
            end
            # Get nspecies from the variable, not from run_info, because it might be
            # either ion or neutral
            nspecies = size(variable[1], 3)
            dims = Vector{mk_int}()
            iz === nothing && push!(dims, run_info.z.n)
            ir === nothing && push!(dims, run_info.r.n)
            is === nothing && push!(dims, nspecies)
            push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 6
            # ion distribution function variable with dimensions (vpa,vperp,z,r,s,t)
            not_allowed_slices = (ivzeta=ivzeta, ivr=ivr, ivz=ivz)
            if any(i !== nothing for i ∈ values(not_allowed_slices))
                error("Got slice for non-existing dimension of 4d variable. "
                      * "All of $not_allowed_slices should be `nothing`.")
            end
            dims = Vector{mk_int}()
            ivpa === nothing && push!(dims, run_info.vpa.n)
            ivperp === nothing && push!(dims, run_info.vperp.n)
            iz === nothing && push!(dims, run_info.z.n)
            ir === nothing && push!(dims, run_info.r.n)
            is === nothing && push!(dims, nspecies)
            push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 7
            # neutral distribution function variable with dimensions (vz,vr,vzeta,z,r,s,t)
            not_allowed_slices = (ivperp=ivperp, ivpa=ivpa)
            if any(i !== nothing for i ∈ values(not_allowed_slices))
                error("Got slice for non-existing dimension of 5d variable. "
                      * "All of $not_allowed_slices should be `nothing`.")
            end
            dims = Vector{mk_int}()
            ivpz === nothing && push!(dims, run_info.vpz.n)
            ivr === nothing && push!(dims, run_info.vr.n)
            ivzeta === nothing && push!(dims, run_info.vzeta.n)
            iz === nothing && push!(dims, run_info.z.n)
            ir === nothing && push!(dims, run_info.r.n)
            is === nothing && push!(dims, nspecies)
            push!(dims, nt)
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

            # Is there a nicer way to cover all the possible combinations of slices here?
            if nd == 3 && ir === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,:,tinds]
            elseif nd == 3 && iz === nothing
                result[:,global_it_start:global_it_end] = v[:,ir,tinds]
            elseif nd == 3 && ir === nothing
                result[:,global_it_start:global_it_end] = v[iz,:,tinds]
            elseif nd == 3
                result[global_it_start:global_it_end] = v[iz,ir,tinds]
            elseif nd == 4 && is === nothing && ir === nothing && iz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,:,:,tinds]
            elseif nd == 4 && is === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ir,:,tinds]
            elseif nd == 4 && is === nothing && ir === nothing
                result[:,:,global_it_start:global_it_end] = v[iz,:,:,tinds]
            elseif nd == 4 && is === nothing
                result[:,global_it_start:global_it_end] = v[iz,ir,:,tinds]
            elseif nd == 4 && ir === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,:,is,tinds]
            elseif nd == 4 && iz === nothing
                result[:,global_it_start:global_it_end] = v[:,ir,is,tinds]
            elseif nd == 4 && ir === nothing
                result[:,global_it_start:global_it_end] = v[iz,:,is,tinds]
            elseif nd == 4
                result[:,global_it_start:global_it_end] = v[iz,ir,is,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && iz === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,:,:,global_it_start:global_it_end] = v[:,:,:,:,:,tinds]
            elseif nd == 6 && is === nothing && iz === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,:,ir,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,iz,:,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && iz === nothing && ivpa === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,ivperp,:,:,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && iz === nothing && iverp === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[ivpa,:,:,:,:,tinds]
            elseif nd == 6 && is === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,:,iz,ir,:,tinds]
            elseif nd == 6 && is === nothing && iz === nothing && ivpa === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivperp,:,ir,:,tinds]
            elseif nd == 6 && is === nothing && iz === nothing && ivperp === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivpa,:,:,ir,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && ivpa === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivperp,iz,:,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && ivperp === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivpa,:,iz,:,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing && iz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivpa,ivperp,:,:,:,tinds]
            elseif nd == 6 && is === nothing && ivpa === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivperp,iz,ir,:,tinds]
            elseif nd == 6 && is === nothing && ivperp === nothing
                result[:,:,global_it_start:global_it_end] = v[ivpa,:,iz,ir,:,tinds]
            elseif nd == 6 && is === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[ivpa,ivperp,:,ir,:,tinds]
            elseif nd == 6 && is === nothing && ir === nothing
                result[:,:,global_it_start:global_it_end] = v[ivpa,ivperp,iz,:,:,tinds]
            elseif nd == 6 && is === nothing
                result[:,global_it_start:global_it_end] = v[ivpa,ivperp,ir,iz,:,tinds]
            elseif nd == 6 && ir === nothing && iz === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,:,:,is,tinds]
            elseif nd == 6 && iz === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,:,:,ir,is,tinds]
            elseif nd == 6 && ir === nothing && ivperp === nothing && ivpa === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,:,iz,:,is,tinds]
            elseif nd == 6 && ir === nothing && iz === nothing && ivpa === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivperp,:,:,is,tinds]
            elseif nd == 6 && ir === nothing && iz === nothing && iverp === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivpa,:,:,:,is,tinds]
            elseif nd == 6 && ivperp === nothing && ivpa === nothing
                result[:,:,global_it_start:global_it_end] = v[:,:,iz,ir,is,tinds]
            elseif nd == 6 && iz === nothing && ivpa === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivperp,:,ir,is,tinds]
            elseif nd == 6 && iz === nothing && ivperp === nothing
                result[:,:,global_it_start:global_it_end] = v[ivpa,:,:,ir,is,tinds]
            elseif nd == 6 && ir === nothing && ivpa === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivperp,iz,:,is,tinds]
            elseif nd == 6 && ir === nothing && ivperp === nothing
                result[:,:,global_it_start:global_it_end] = v[ivpa,:,iz,:,is,tinds]
            elseif nd == 6 && ir === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[ivpa,ivperp,:,:,is,tinds]
            elseif nd == 6 && ivpa === nothing
                result[:,global_it_start:global_it_end] = v[:,ivperp,iz,ir,is,tinds]
            elseif nd == 6 && ivperp === nothing
                result[:,global_it_start:global_it_end] = v[ivpa,:,iz,ir,is,tinds]
            elseif nd == 6 && iz === nothing
                result[:,global_it_start:global_it_end] = v[ivpa,ivperp,:,ir,is,tinds]
            elseif nd == 6 && ir === nothing
                result[:,global_it_start:global_it_end] = v[ivpa,ivperp,iz,:,is,tinds]
            elseif nd == 6
                result[global_it_start:global_it_end] = v[ivpa,ivperp,ir,iz,is,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && iz === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,:,:,global_it_start:global_it_end] = v[:,:,:,:,:,:,tinds]
            elseif nd == 7 && is === nothing && iz === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,:,global_it_start:global_it_end] = v[:,:,:,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,:,global_it_start:global_it_end] = v[:,:,:,iz,:,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && iz === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,:,global_it_start:global_it_end] = v[:,:,:,iz,:,:,tinds]
            elseif nd == 7 && is === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,ivzeta,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivzeta === nothing && ivzeta === nothing && ivz === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,ivr,:,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivzeta === nothing && ivzeta === nothing && ivr === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[ivz,:,:,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,:,ivzeta,iz,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivzeta === nothing && ivz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivr,:,iz,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivzeta === nothing && ivr === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,:,:,iz,ir,:,tinds]
            elseif nd == 7 && is === nothing && iz === nothing && ivz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivr,ivzeta,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && iz === nothing && ivr === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,:,ivzeta,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && iz === nothing && ivzeta === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,ivr,:,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && ivz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivr,ivzeta,iz,:,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && ivr === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,:,ivzeta,iz,:,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && ivzeta === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,ivr,:,iz,:,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing && iz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,:,:,:,tinds]
            elseif nd == 7 && is === nothing && ivz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivr,ivzeta,iz,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivr === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,:,ivzeta,iz,ir,:,tinds]
            elseif nd == 7 && is === nothing && ivzeta === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,ivr,:,iz,ir,:,tinds]
            elseif nd == 7 && is === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,:,ir,:,tinds]
            elseif nd == 7 && is === nothing && ir === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,iz,:,:,tinds]
            elseif nd == 7 && is === nothing
                result[:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,iz,ir,:,tinds]
            elseif nd == 7 && ir === nothing && iz === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,:,global_it_start:global_it_end] = v[:,:,:,:,:,is,tinds]
            elseif nd == 7 && iz === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,:,:,ir,is,tinds]
            elseif nd == 7 && ir === nothing && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,:,iz,:,is,tinds]
            elseif nd == 7 && ir === nothing && iz === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,:,global_it_start:global_it_end] = v[:,:,:,iz,:,is,tinds]
            elseif nd == 7 && ivzeta === nothing && ivr === nothing && ivz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,:,ivzeta,:,ir,is,tinds]
            elseif nd == 7 && ivzeta === nothing && ivzeta === nothing && ivz === nothing
                result[:,:,:,global_it_start:global_it_end] = v[:,ivr,:,:,ir,is,tinds]
            elseif nd == 7 && ivzeta === nothing && ivzeta === nothing && ivr === nothing
                result[:,:,:,global_it_start:global_it_end] = v[ivz,:,:,:,ir,is,tinds]
            elseif nd == 7 && ivr === nothing && ivz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,:,ivzeta,iz,ir,is,tinds]
            elseif nd == 7 && ivzeta === nothing && ivz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivr,:,iz,ir,is,tinds]
            elseif nd == 7 && ivzeta === nothing && ivr === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,:,:,iz,ir,is,tinds]
            elseif nd == 7 && iz === nothing && ivz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivr,ivzeta,:,ir,is,tinds]
            elseif nd == 7 && iz === nothing && ivr === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,:,ivzeta,:,ir,is,tinds]
            elseif nd == 7 && iz === nothing && ivzeta === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,ivr,:,:,ir,is,tinds]
            elseif nd == 7 && ir === nothing && ivz === nothing
                result[:,:,global_it_start:global_it_end] = v[:,ivr,ivzeta,iz,:,is,tinds]
            elseif nd == 7 && ir === nothing && ivr === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,:,ivzeta,iz,:,is,tinds]
            elseif nd == 7 && ir === nothing && ivzeta === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,ivr,:,iz,:,is,tinds]
            elseif nd == 7 && ir === nothing && iz === nothing
                result[:,:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,:,:,is,tinds]
            elseif nd == 7 && ivz === nothing
                result[:,global_it_start:global_it_end] = v[:,ivr,ivzeta,iz,ir,is,tinds]
            elseif nd == 7 && ivr === nothing
                result[:,global_it_start:global_it_end] = v[ivz,:,ivzeta,iz,ir,is,tinds]
            elseif nd == 7 && ivzeta === nothing
                result[:,global_it_start:global_it_end] = v[ivz,ivr,:,iz,ir,is,tinds]
            elseif nd == 7 && iz === nothing
                result[:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,:,ir,is,tinds]
            elseif nd == 7 && ir === nothing
                result[:,global_it_start:global_it_end] = v[ivz,ivr,ivzeta,iz,:,is,tinds]
            elseif nd == 7
                result[global_it_start:global_it_end] = v[ivz,ivr,ivzeta,iz,ir,is,tinds]
            else
                error("Unsupported combination nd=$nd, ir=$ir, iz=$iz, ivperp=$ivperp "
                      * "ivpa=$ivpa, ivzeta=$ivzeta, ivr=$ivr, ivz=$ivz.")
            end

            local_it_start = local_it_end + 1
            global_it_start = global_it_end + 1
        end
    else
        # Use existing distributed I/O loading functions
        error("parallel_io=false not supported yet")
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
        variable = Tuple(postproc_load_variable(ri, variable_name; it=tinds)
                         for ri ∈ run_info)
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
            else
                variable_prefix = plot_prefix * variable_name * "_"
            end
            if variable_name == "Er" && is_1D
                # Skip if there is no r-dimension
                continue
            end
            if input.plot_vs_z_t
                plot_vs_z_t(run_info, variable_name, is=is, data=variable, input=input,
                            outfile=variable_prefix * "vs_z_t.pdf")
            end
            if input.animate_vs_z
                animate_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                             outfile=variable_prefix * "vs_z.gif")
            end
        end
    end

    return nothing
end

function plot_vs_z_t(run_info::Tuple, var_name; is=1, data=nothing, input=nothing,
                     outfile=nothing)

    try
        if data === nothing
            data = Tuple(nothing for _ in run_info)
        end
        fig, ax, colorbar_places = get_2d_ax(length(run_info),
                                             title=get_variable_symbol(var_name))
        for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
            plot_vs_z_t(ri, var_name, is=is, data=d, input=input, ax=a, colorbar_place=cp,
                        title=ri.run_name)
        end

        if outfile !== nothing
            save(outfile, fig)
        end
        return fig
    catch e
        println("plot_vs_z_t failed for $var_name, is=$is. Error was $e")
        return nothing
    end
end

function plot_vs_z_t(run_info, var_name; is=1, data=nothing, input=nothing,
                     ax=nothing, colorbar_place=colorbar_place, title=nothing,
                     outfile=nothing)
    if data === nothing
        data = postproc_load_variable(run_info, var_name)
    end
    if input === nothing
        colormap = "reverse_deep"
    else
        colormap = input.colormap
    end
    if title === nothing
        title = get_variable_symbol(var_name)
    end

    data = select_z_t(data, input, is=is)

    fig = plot_2d(run_info.z.grid, run_info.time, data, xlabel="z", ylabel="time",
                  title=title, ax=ax, colorbar_place=colorbar_place,
                  colormap=parse_colormap(colormap))

    if outfile !== nothing && fig !== nothing
        save(outfile, fig)
    end

    return nothing
end

function animate_vs_z(run_info::Tuple, var_name; is=1, data=nothing, input=nothing,
                      outfile=nothing)

    try
        if data === nothing
            data = Tuple(nothing for _ in run_info)
        end
        # Load data if necessary
        data = Tuple(d === nothing ? postproc_load_variable(ri, var_name) : d
                     for (d,ri) ∈ zip(data, run_info))
        # Select needed dims
        data = Tuple(select_z_t(d, input, is=is) for d ∈ data)

        zcoord = Tuple(ri.z.grid for ri ∈ run_info)

        labels = Tuple(ri.run_name for ri ∈ run_info)

        fig = animate_1d(zcoord, data, xlabel="z", ylabel=get_variable_symbol(var_name),
                         labels=labels, outfile=outfile)

        return fig
    catch e
        println("$var_name, is=$is failed to animate. Error was $e")
        return nothing
    end
end

function animate_vs_z(run_info, var_name; is=1, data=nothing, input=nothing,
                      outfile=nothing)
    return animate_vs_z((run_info,), var_name; is=is, data=data, input=input,
                        outfile=outfile)
end

function get_1d_ax(n=nothing; title=nothing, kwargs...)
    if n == nothing
        fig = Figure(title=title)
        ax = Axis(fig[1,1]; kwargs...)
    else
        fig = Figure(resolution=(600*n, 400), title=title)

        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)

            plot_layout = fig[2,1] = GridLayout()
        else
            plot_layout = fig[1,1] = GridLayout()
        end

        ax = [Axis(plot_layout[1,i]; kwargs...) for i in 1:n]
    end

    return fig, ax
end

function get_2d_ax(n=nothing; title=nothing, kwargs...)
    if n == nothing
        fig = Figure(title=title)
        ax = Axis(fig[1,1]; kwargs...)
        colorbar_places = fig[1,2]
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
                 ylabel=nothing, title=nothing, kwargs...)
    if ax === nothing
        fig, ax, _ = get_1d_ax()
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

    l = lines!(ax, xcoord, data; kwargs...)

    if ax === nothing
        return fig
    else
        return l
    end
end

function plot_2d(xcoord, ycoord, data; ax=nothing, colorbar_place=nothing, xlabel=nothing,
                 ylabel=nothing, title=nothing, kwargs...)
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

function animate_1d(xcoord::Tuple, data::Tuple; xlabel=nothing, ylabel=nothing,
                    title=nothing, labels=nothing, outfile=nothing)
    n_runs = length(data)

    if labels === nothing
        labels = Tuple(nothing for _ ∈ 1:n_runs)
    end
    if outfile === nothing
        error("outfile is required for animate_1d()")
    end

    index = Observable(1)

    fig, ax = get_1d_ax(title=title, xlabel=xlabel, ylabel=ylabel)
    line_data = (@lift(@view d[:,$index]) for d ∈ data)
    for (i, (x, d, l)) ∈ enumerate(zip(xcoord, line_data, labels))
        lines!(ax, x, d, label=l)
    end
    put_legend_above(fig, ax)

    nt = minimum(size(d, 2) for d ∈ data)

    record(fig, outfile, 1:nt, framerate=5) do it
        index[] = it
    end
end

function animate_1d(xcoord, data; xlabel=nothing, ylabel=nothing, title=nothing,
                    labels=nothing, outfile=nothing)
    return animate_1d((xcoord,), (data,), xlabel=xlabel, ylabel=ylabel, title=title,
                      labels=(labels,), outfile=outfile)
end

function animate_2d(xcoord::Tuple, ycoord::Tuple, data::Tuple; xlabel=nothing,
                    ylabel=nothing, title=nothing, sub_titles=nothing, colormap=nothing,
                    outfile=nothing)
    n_runs = length(data)

    if sub_titles === nothing
        sub_titles = Tuple(nothing for _ ∈ 1:n_runs)
    end
    if outfile === nothing
        error("outfile is required for animate_2d()")
    end
    colormap = parse_colormap(colormap)

    fig, ax, colorbar_places = get_2d_ax(n_runs, title=title, xlabel=xlabel,
                                         ylabel=ylabel)
    hm = []
    for (i, (x, y, d, t, a, cp)) ∈ enumerate(zip(xcoord, ycoord, data, sub_titles, ax,
                                                 colorbar_places))
        this_hm = heatmap!(a, x, y, d[:,:,1], title=t, colormap=colormap)
        Colorbar(cp, this_hm)

        push!(hm, this_hm)
    end

    nt = minimum(size(d, 3) for d ∈ data)

    record(fig, outfile, 1:nt, framerate=5) do it
        for (h, d) ∈ zip(hm, data)
            h[3] = @view d[:,:,it]
        end
    end
end

function animate_2d(xcoord, ycoord, data; xlabel=nothing, ylabel=nothing, title=nothing,
                    sub_titles=nothing, colormap=nothing, outfile=nothing)
    return animate_2d((xcoord,), (ycoord,), (data,), xlabel=xlabel, ylabel=ylabel,
                      title=title, sub_titles=(sub_titles,), colormap=colormap,
                      outfile=outfile)
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

    if input === nothing
        it0 = size(variable, 3)
        ir0 = max(size(variable, 2) ÷ 3, 1)
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        ir0 = input.ir0
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

    if input === nothing
        it0 = size(variable, 4)
        ir0 = max(size(variable, 2) ÷ 3, 1)
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        ir0 = input.ir0
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

function select_slice_dfns(variable::AbstractArray{T,1}, dims::Symbol...; input=nothing, is=nothing) where T
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

function select_slice_dfns(variable::AbstractArray{T,2}, dims::Symbol...; input=nothing, is=nothing) where T
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

function select_slice_dfns(variable::AbstractArray{T,3}, dims::Symbol...; input=nothing, is=nothing) where T
    # Array is (z,r,t)

    if length(dims) > 3
        error("Tried to get a slice of 3d variable with dimensions $dims")
    end

    if input === nothing
        it0 = size(variable, 3)
        ir0 = max(size(variable, 2) ÷ 3, 1)
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        it0 = input.it0_dfns
        ir0 = input.ir0
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims
        slice = selectdim(slice, 3, it0)
    end
    if :r ∉ dims
        slice = selectdim(slice, r, ir0)
    end
    if :z ∉ dims
        slice = selectdim(slice, z, iz0)
    end

    return slice
end

function select_slice_dfns(variable::AbstractArray{T,4}, dims::Symbol...; input=nothing, is=1) where T
    # Array is (z,r,species,t)

    if input === nothing
        it0 = size(variable, 4)
        ir0 = max(size(variable, 1) ÷ 3, 2)
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        it0 = input.it0_dfns
        ir0 = input.ir0
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

function select_slice_dfns(variable::AbstractArray{T,6}, dims::Symbol...; input=nothing, is=1) where T
    # Array is (z,r,species,t)

    if input === nothing
        it0 = size(variable, 6)
        ir0 = max(size(variable, 4) ÷ 3, 1)
        iz0 = max(size(variable, 3) ÷ 3, 1)
        ivpa0 = max(size(variable, 2) ÷ 3, 1)
        ivperp0 = max(size(variable, 1) ÷ 3, 1)
    else
        it0 = input.it0_dfns
        ir0 = input.ir0
        iz0 = input.iz0
        ivpa0 = input.ivpa0
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

function select_slice_dfns(variable::AbstractArray{T,7}, dims::Symbol...; input=nothing, is=1) where T
    # Array is (z,r,species,t)

    if input === nothing
        it0 = size(variable, 7)
        ir0 = max(size(variable, 5) ÷ 3, 1)
        iz0 = max(size(variable, 4) ÷ 3, 1)
        ivzeta0 = max(size(variable, 3) ÷ 3, 1)
        ivr0 = max(size(variable, 2) ÷ 3, 1)
        ivz0 = max(size(variable, 1) ÷ 3, 1)
    else
        it0 = input.it0_dfns
        ir0 = input.ir0
        iz0 = input.iz0
        ivzeta0 = input.ivzeta0
        ivr0 = input.ivr0
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

end
