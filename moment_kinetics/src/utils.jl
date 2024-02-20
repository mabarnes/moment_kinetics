"""
Utility functions
"""
module utils

export get_unnormalized_parameters, print_unnormalized_parameters, to_seconds, to_minutes,
       to_hours

using ..communication
using ..constants
using ..input_structs
using ..moment_kinetics_input: mk_input
using ..reference_parameters

using Dates
using Glob
using MPI
using OrderedCollections
using TOML
using Unitful

Unitful.@unit eV "eV" "electron volt" proton_charge*Unitful.J true

function __init__()
    Unitful.register(utils)
end

"""
    get_unnormalized_parameters(input::Dict)
    get_unnormalized_parameters(input_filename::String)

Get many parameters for the simulation setup given by `input` or in the file
`input_filename`, in SI units and eV, returned as an OrderedDict.
"""
function get_unnormalized_parameters end
function get_unnormalized_parameters(input::Dict)
    io_input, evolve_moments, t_input, z, z_spectral, r, r_spectral, vpa, vpa_spectral,
        vperp, vperp_spectral, gyrophase, gyrophase_spectral, vz, vz_spectral, vr,
        vr_spectral, vzeta, vzeta_spectral, composition, species, collisions, geometry,
        drive_input, external_source_settings, num_diss_params, manufactured_solns_input =
            mk_input(input)

    reference_params = setup_reference_parameters(input)

    Nnorm = reference_params.Nref * Unitful.m^(-3)
    Tnorm = reference_params.Tref * eV
    Lnorm = reference_params.Lref * Unitful.m
    Bnorm = reference_params.Bref * Unitful.T
    cnorm = reference_params.cref * Unitful.m / Unitful.s
    timenorm = reference_params.timeref * Unitful.s

    # Assume single ion species so normalised ion mass is always 1
    mi = reference_params.mref * Unitful.kg

    parameters = OrderedDict{String,Any}()
    parameters["run_name"] = run_name

    parameters["Nnorm"] = Nnorm
    parameters["Tnorm"] = Tnorm
    parameters["Lnorm"] = Lnorm

    parameters["Lz"] = Lnorm * z_input.L

    parameters["cs0"] = cnorm

    dt = t_input.dt * timenorm
    parameters["dt"] = dt
    parameters["output time step"] = dt * t_input.nwrite
    parameters["total simulated time"] = dt * t_input.nstep

    parameters["T_e"] = Tnorm * composition.T_e
    parameters["T_wall"] = Tnorm * composition.T_wall

    parameters["CX_rate_coefficient"] = collisions.charge_exchange / Nnorm / timenorm
    parameters["ionization_rate_coefficient"] = collisions.ionization / Nnorm / timenorm
    parameters["coulomb_collision_frequency0"] =
        collisions.coulomb_collision_frequency_prefactor / timenorm

    return parameters
end
function get_unnormalized_parameters(input_filename::String, args...; kwargs...)
    return get_unnormalized_parameters(TOML.parsefile(input_filename), args...;
                                       kwargs...)
end

"""
    print_unnormalized_parameters(input)

Print many parameters for the simulation setup given by `input` (a Dict of parameters or
a String giving a filename), in SI units and eV.
"""
function print_unnormalized_parameters(args...; kwargs...)

    parameters = get_unnormalized_parameters(args...; kwargs...)

    println("Dimensional parameters for '$(parameters["run_name"])'")

    for (k,v) ∈ parameters
        println("$k = $v")
    end

    return nothing
end

# Utility functions for dates, adapted from
# https://discourse.julialang.org/t/convert-time-interval-to-seconds/3806/4

"""
    to_seconds(x::T) where {T<:TimePeriod}

Convert a time period `x` to seconds
"""
to_seconds(x::T) where {T<:TimePeriod} = x/convert(T, Second(1))

"""
    to_minutes(x::T) where {T<:TimePeriod}

Convert a time period `x` to seconds
"""
to_minutes(x::T) where {T<:TimePeriod} = x/convert(T, Minute(1))

"""
    to_hours(x::T) where {T<:TimePeriod}

Convert a time period `x` to seconds
"""
to_hours(x::T) where {T<:TimePeriod} = x/convert(T, Hour(1))

# Utility functions used for restarting

"""
Append a number to the filename, to get a new, non-existing filename to backup the file
to.
"""
function get_backup_filename(filename)
    if !isfile(filename)
        error("Requested to restart from $filename, but this file does not exist")
    end
    counter = 1
    temp, extension = splitext(filename)
    extension = extension[2:end]
    temp, iblock_or_type = splitext(temp)
    iblock_or_type = iblock_or_type[2:end]
    iblock = nothing
    basename = nothing
    type = nothing
    if iblock_or_type == "dfns"
        iblock = nothing
        type = iblock_or_type
        basename = temp
        parallel_io = true
    else
        # Filename had an iblock, so we are not using parallel I/O, but actually want to
        # use the iblock for this block, not necessarily for the exact file that was
        # passed.
        iblock = iblock_index[]
        basename, type = splitext(temp)
        type = type[2:end]
        parallel_io = false
    end
    if type != "dfns"
        error("Must pass the '.dfns.h5' output file for restarting. Got $filename.")
    end
    backup_dfns_filename = ""
    if parallel_io
        # Using parallel I/O
        while true
            backup_dfns_filename = "$(basename)_$(counter).$(type).$(extension)"
            if !isfile(backup_dfns_filename)
                break
            end
            counter += 1
        end
        # Create dfns_filename here even though it is the filename passed in, as
        # parallel_io=false branch needs to get the right `iblock` for this block.
        dfns_filename = "$(basename).dfns.$(extension)"
        moments_filename = "$(basename).moments.$(extension)"
        backup_moments_filename = "$(basename)_$(counter).moments.$(extension)"
    else
        while true
            backup_dfns_filename = "$(basename)_$(counter).$(type).$(iblock).$(extension)"
            if !isfile(backup_dfns_filename)
                break
            end
            counter += 1
        end
        # Create dfns_filename here even though it is almost the filename passed in, in
        # order to get the right `iblock` for this block.
        dfns_filename = "$(basename).dfns.$(iblock).$(extension)"
        moments_filename = "$(basename).moments.$(iblock).$(extension)"
        backup_moments_filename = "$(basename)_$(counter).moments.$(iblock).$(extension)"
    end
    backup_dfns_filename == "" && error("Failed to find a name for backup file.")
    backup_prefix_iblock = ("$(basename)_$(counter)", iblock)
    original_prefix_iblock = (basename, iblock)
    return dfns_filename, backup_dfns_filename, parallel_io, moments_filename,
           backup_moments_filename, backup_prefix_iblock, original_prefix_iblock
end

"""
    get_default_restart_filename(io_input, prefix; error_if_no_file_found=true)

Get the default name for the file to restart from, using the input from `io_input`.

`prefix` gives the type of file to open, e.g. "moments", "dfns", or "initial_electron".

If no matching file is found, raise an error unless `error_if_no_file_found=false` is
passed, in which case no error is raised and instead the function returns `nothing`.
"""
function get_default_restart_filename(io_input, prefix; error_if_no_file_found=true)
    binary_format = io_input.binary_format
    if binary_format === hdf5
        ext = "h5"
    elseif binary_format === netcdf
        ext = "cdf"
    else
        error("Unrecognized binary_format '$binary_format'")
    end
    restart_filename_pattern = joinpath(io_input.output_dir, io_input.run_name * ".$prefix*." * ext)
    restart_filename_glob = glob(restart_filename_pattern)
    if length(restart_filename_glob) == 0
        if error_if_no_file_found
            error("No '$prefix' output file to restart from found matching the pattern "
                  * "$restart_filename_pattern")
        end
        restart_filename = nothing
    else
        restart_filename = restart_filename_glob[1]
    end
    return restart_filename
end

"""
    get_prefix_iblock_and_move_existing_file(restart_filename, output_dir)

Move `restart_filename` to a backup location (if it is in `output_dir`), returning a
prefix and block-index (which might be `nothing`) which can be used to open the file for
reloading variables.
"""
function get_prefix_iblock_and_move_existing_file(restart_filename, output_dir)
    # Move the output file being restarted from to make sure it doesn't get
    # overwritten.
    dfns_filename, backup_dfns_filename, parallel_io, moments_filename,
    backup_moments_filename, backup_prefix_iblock, original_prefix_iblock =
        get_backup_filename(restart_filename)

    # Ensure every process got the filenames and checked files exist before moving
    # files
    MPI.Barrier(comm_world)

    if abspath(output_dir) == abspath(dirname(dfns_filename))
        # Only move the file if it is in our current run directory. Otherwise we are
        # restarting from another run, and will not be overwriting the file.
        if (parallel_io && global_rank[] == 0) || (!parallel_io && block_rank[] == 0)
            mv(dfns_filename, backup_dfns_filename)
            mv(moments_filename, backup_moments_filename)
        end
    else
        # Reload from dfns_filename without moving the file
        backup_prefix_iblock = original_prefix_iblock
    end

    # Ensure files have been moved before any process tries to read from them
    MPI.Barrier(comm_world)

    return backup_prefix_iblock
end

end #utils
