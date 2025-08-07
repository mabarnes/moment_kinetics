"""
Utility functions
"""
module utils

export get_unnormalized_parameters, print_unnormalized_parameters, to_seconds, to_minutes,
       to_hours, recursive_merge, merge_dict_with_kwargs!

using ..communication
using ..constants
using ..input_structs
using ..looping
using ..moment_kinetics_input: mk_input
using ..reference_parameters

# Import moment_kinetics so we can refer to it in docstrings
import ..moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

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
    get_unnormalized_parameters(input::AbstractDict)
    get_unnormalized_parameters(input_filename::String)

Get many parameters for the simulation setup given by `input` or in the file
`input_filename`, in SI units and eV, returned as an OrderedDict.
"""
function get_unnormalized_parameters end
function get_unnormalized_parameters(input::OptionsDict, warn_unexpected::Bool=false)
    io_input, evolve_moments, t_input, z, z_spectral, r, r_spectral, vpa, vpa_spectral,
        vperp, vperp_spectral, gyrophase, gyrophase_spectral, vz, vz_spectral, vr,
        vr_spectral, vzeta, vzeta_spectral, composition, species, collisions, geometry,
        drive_input, num_diss_params, manufactured_solns_input =
            mk_input(input)

    reference_params = setup_reference_parameters(input, warn_unexpected)

    Nnorm = reference_params.Nref * Unitful.m^(-3)
    Tnorm = reference_params.Tref * eV
    Lnorm = reference_params.Lref * Unitful.m
    Bnorm = reference_params.Bref * Unitful.T
    cnorm = reference_params.cref * Unitful.m / Unitful.s
    timenorm = reference_params.timeref * Unitful.s

    parameters = OrderedDict{String,Any}()
    parameters["run_name"] = io_input.run_name

    # Assume single ion species so normalised ion mass is always 1
    m_i = reference_params.mref * Unitful.kg
    parameters["m_i"] = m_i
    m_e = electron_mass * Unitful.kg
    parameters["m_e"] = m_e

    parameters["Nnorm"] = Nnorm
    parameters["Tnorm"] = Tnorm
    parameters["Lnorm"] = Lnorm
    parameters["timenorm"] = timenorm

    parameters["Lz"] = Lnorm * z.L

    parameters["cs0"] = sqrt(2.0) * cnorm
    parameters["vthi0"] = sqrt(2.0) * cnorm
    parameters["vthe0"] = sqrt(2.0 / composition.me_over_mi) * cnorm

    dt = t_input["dt"] * timenorm
    parameters["dt"] = dt
    parameters["output time step"] = dt * t_input["nwrite"]
    parameters["total simulated time"] = dt * t_input["nstep"]

    parameters["T_e"] = Tnorm * composition.T_e
    parameters["T_wall"] = Tnorm * composition.T_wall

    parameters["CX_rate_coefficient"] = collisions.reactions.charge_exchange_frequency / Nnorm / timenorm
    parameters["ionization_rate_coefficient"] = collisions.reactions.ionization_frequency / Nnorm / timenorm
    # Dimensionless thermal speeds at T=Tnorm - need to divide by vth0^3 to get collision
    # frequency at reference parameters because of way nuii0 is defined in code.
    vthi0 = sqrt(2.0)
    vthe0 = sqrt(2.0 / composition.me_over_mi)
    parameters["coulomb_collision_frequency_ii0"] =
        get_reference_collision_frequency_ii(reference_params) / vthi0^3 / timenorm
    parameters["coulomb_collision_frequency_ee0"] =
        get_reference_collision_frequency_ee(reference_params) / vthe0^3 / timenorm
    parameters["coulomb_collision_frequency_ei0"] =
        get_reference_collision_frequency_ei(reference_params) / vthe0^3 / timenorm
    parameters["coulomb_collision_frequency_ie0"] =
        parameters["coulomb_collision_frequency_ei0"] * composition.me_over_mi
    parameters["krook_collision_frequency_ii0"] =
        collisions.krook.nuii0 / vthi0^3 / timenorm
    parameters["krook_collision_frequency_ee0"] =
        collisions.krook.nuee0 / vthe0^3 / timenorm
    parameters["krook_collision_frequency_ei0"] =
        collisions.krook.nuei0 / vthe0^3 / timenorm

    # Include some useful derived quantities
    pcharge = proton_charge * Unitful.C
    parameters["Omega_i0"] = pcharge * Bnorm / m_i
    parameters["Omega_e0"] = pcharge * Bnorm / m_e
    parameters["rho_i0"] = parameters["vthi0"] / parameters["Omega_i0"]
    parameters["rho_e0"] = parameters["vthe0"] / parameters["Omega_e0"]

    return parameters
end
function get_unnormalized_parameters(input_filename::String, args...; kwargs...)
    return get_unnormalized_parameters(TOML.parsefile(input_filename), args...;
                                       kwargs...)
end

"""
    print_unnormalized_parameters(input)

Print many parameters for the simulation setup given by `input` (an AbstractDict of
parameters or a String giving a filename), in SI units and eV.
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
    if iblock_or_type ∈ ("dfns", "initial_electron")
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
    if type ∉ ("dfns", "initial_electron")
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
        dfns_filename = "$(basename).$(type).$(extension)"
        if type == "dfns"
            moments_filename = "$(basename).moments.$(extension)"
        else
            moments_filename = nothing
        end
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
        dfns_filename = "$(basename).$(type).$(iblock).$(extension)"
        if type == "dfns"
            moments_filename = "$(basename).moments.$(iblock).$(extension)"
        else
            moments_filename = nothing
        end
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
    if isabspath(restart_filename_pattern)
        # Special handling for absolute paths, as these give an error when `glob()` is
        # called normally
        restart_filename_glob = glob(basename(restart_filename_pattern),
                                     dirname(restart_filename_pattern))
    else
        restart_filename_glob = glob(restart_filename_pattern)
    end
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
            if moments_filename !== nothing
                mv(moments_filename, backup_moments_filename)
            end
        end
    else
        # Reload from dfns_filename without moving the file
        backup_prefix_iblock = original_prefix_iblock
    end

    # Ensure files have been moved before any process tries to read from them
    MPI.Barrier(comm_world)

    return backup_prefix_iblock, dfns_filename, backup_dfns_filename
end

"""
    enum_from_string(enum_type, name)

Get an the value of `enum_type`, whose name is given by the String (or Symbol) `name`.

Returns `nothing` if the name is not found.
"""
function enum_from_string(enum_type, name)
    name = Symbol(name)
    for e ∈ instances(enum_type)
        if name == Symbol(e)
            return e
        end
    end
    return nothing
end

"""
    recursive_merge(a, b)

Merge two AbstractDicts `a` and `b`. Any elements that are AbstractDicts are also merged
(rather than just replacing with the entry in `b`).
"""
function recursive_merge end
function recursive_merge(a::AbstractDict, b::AbstractDict)
    result = deepcopy(a)
    a_keys = collect(keys(a))
    for (k,v) ∈ pairs(b)
        if k ∉ a_keys
            result[k] = v
        elseif isa(result[k], AbstractDict) && isa(v, AbstractDict)
            result[k] = recursive_merge(result[k], v)
        elseif isa(result[k], AbstractDict) || isa(v, AbstractDict)
            error("Cannot merge a Dict with a non-Dict, got $(result[k]) and $v")
        else
            result[k] = v
        end
    end
    return result
end

"""
Dict merge function for named keyword arguments for case when input AbstractDict is a
mixed AbstractDict of AbstractDicts and non-AbstractDict float/int/string entries, and
the keyword arguments are also a mix of AbstractDicts and non-AbstractDicts
"""
function merge_dict_with_kwargs!(dict_base; args...)
    for (k,v) in args
        k = String(k)
        if k in keys(dict_base) && isa(v, AbstractDict)
            v = recursive_merge(dict_base[k], v)
        end
        dict_base[k] = v
    end
    return nothing
end

# Utility functions for timestepping

"""
    get_minimum_CFL_r(speed, r)

Calculate the minimum (over a shared-memory block) of the CFL factor 'speed/(grid
spacing)' (with no prefactor) corresponding to advection speed `speed` for advection in
the r direction.

Reduces the result over the shared-memory block (handling distributed parallelism is left
to the calling site). The result is only to be used on rank-0 of the shared-memory block.
"""
function get_minimum_CFL_r(speed::AbstractArray{T,4} where T, r)
    min_CFL = Inf

    dr = r.cell_width
    nr = r.n
    @loop_z_vperp_vpa iz ivperp ivpa begin
        for ir ∈ 1:nr
            min_CFL = min(min_CFL, abs(dr[ir] / speed[ir,ivpa,ivperp,iz]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end

"""
    get_minimum_CFL_z(speed, z)

Calculate the minimum (over a shared-memory block) of the CFL factor 'speed/(grid
spacing)' (with no prefactor) corresponding to advection speed `speed` for advection in
the z direction.

Reduces the result over the shared-memory block (handling distributed parallelism is left
to the calling site). The result is only to be used on rank-0 of the shared-memory block.
"""
function get_minimum_CFL_z(speed::AbstractArray{T,4} where T, z)
    min_CFL = Inf

    dz = z.cell_width
    nz = z.n
    @loop_r_vperp_vpa ir ivperp ivpa begin
        for iz ∈ 1:nz
            min_CFL = min(min_CFL, abs(dz[iz] / speed[iz,ivpa,ivperp,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end
function get_minimum_CFL_z(speed::AbstractArray{T,4} where T, z, ir)
    min_CFL = Inf

    dz = z.cell_width
    nz = z.n
    @loop_vperp_vpa ivperp ivpa begin
        for iz ∈ 1:nz
            min_CFL = min(min_CFL, abs(dz[iz] / speed[iz,ivpa,ivperp,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end

"""
    get_minimum_CFL_vperp(speed, vperp)

Calculate the minimum (over a shared-memory block) of the CFL factor 'speed/(grid
spacing)' (with no prefactor) corresponding to advection speed `speed` for advection in
the vperp direction.

Reduces the result over the shared-memory block (handling distributed parallelism is left
to the calling site). The result is only to be used on rank-0 of the shared-memory block.
"""
function get_minimum_CFL_vperp(speed::AbstractArray{T,4} where T, vperp)
    min_CFL = Inf

    dvperp = vperp.cell_width
    nvperp = vperp.n
    @loop_r_z_vpa ir iz ivpa begin
        for ivperp ∈ 1:nvperp
            min_CFL = min(min_CFL, abs(dvperp[ivperp] / speed[ivperp,ivpa,iz,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end

"""
    get_minimum_CFL_vpa(speed, vpa)

Calculate the minimum (over a shared-memory block) of the CFL factor 'speed/(grid
spacing)' (with no prefactor) corresponding to advection speed `speed` for advection in
the vpa direction.

Reduces the result over the shared-memory block (handling distributed parallelism is left
to the calling site). The result is only to be used on rank-0 of the shared-memory block.
"""
function get_minimum_CFL_vpa(speed::AbstractArray{T,4} where T, vpa)
    min_CFL = Inf

    dvpa = vpa.cell_width
    nvpa = vpa.n
    @loop_r_z_vperp ir iz ivperp begin
        for ivpa ∈ 1:nvpa
            min_CFL = min(min_CFL, abs(dvpa[ivpa] / speed[ivpa,ivperp,iz,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end
function get_minimum_CFL_vpa(speed::AbstractArray{T,4} where T, vpa, ir)
    min_CFL = Inf

    dvpa = vpa.cell_width
    nvpa = vpa.n
    @loop_z_vperp iz ivperp begin
        for ivpa ∈ 1:nvpa
            min_CFL = min(min_CFL, abs(dvpa[ivpa] / speed[ivpa,ivperp,iz,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end

"""
    get_minimum_CFL_neutral_z(speed, z)

Calculate the minimum (over a shared-memory block) of the CFL factor 'speed/(grid
spacing)' (with no prefactor) corresponding to advection speed `speed` for advection of
neutrals in the z direction.

Reduces the result over the shared-memory block (handling distributed parallelism is left
to the calling site). The result is only to be used on rank-0 of the shared-memory block.
"""
function get_minimum_CFL_neutral_z(speed::AbstractArray{T,5} where T, z)
    min_CFL = Inf

    dz = z.cell_width
    nz = z.n
    @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
        for iz ∈ 1:nz
            min_CFL = min(min_CFL, abs(dz[iz] / speed[iz,ivz,ivr,ivzeta,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end

"""
    get_minimum_CFL_neutral_vz(speed, vz)

Calculate the minimum (over a shared-memory block) of the CFL factor 'speed/(grid
spacing)' (with no prefactor) corresponding to advection speed `speed` for advection of
neutrals in the vz direction.

Reduces the result over the shared-memory block (handling distributed parallelism is left
to the calling site). The result is only to be used on rank-0 of the shared-memory block.
"""
function get_minimum_CFL_neutral_vz(speed::AbstractArray{T,5} where T, vz)
    min_CFL = Inf

    dvz = vz.cell_width
    nvz = vz.n
    @loop_r_z_vzeta_vr ir iz ivzeta ivr begin
        for ivz ∈ 1:nvz
            min_CFL = min(min_CFL, abs(dvz[ivz] / speed[ivz,ivr,ivzeta,iz,ir]))
        end
    end

    if comm_block[] !== MPI.COMM_NULL
        min_CFL = MPI.Reduce(min_CFL, min, comm_block[]; root=0)
    end

    return min_CFL
end

"""
    get_CFL!(CFL, speed, coord)

Calculate the CFL factor 'speed/(grid spacing)' (with no prefactor) corresponding to
advection speed `speed` for advection. Note that moment_kinetics is set up so that
dimension in which advection happens is the first dimension of `speed` - `coord` is the
coordinate corresponding to this dimension.

The result is written in `CFL`. This function is only intended to be used in
post-processing.
"""
function get_CFL end

function get_CFL!(CFL::AbstractArray{T,4}, speed::AbstractArray{T,4}, coord) where T

    nmain, n2, n3, n4 = size(speed)

    for i4 ∈ 1:n4, i3 ∈ 1:n3, i2 ∈ 1:n2, imain ∈ 1:nmain
        CFL[imain,i2,i3,i4] = abs(coord.cell_width[imain] / speed[imain,i2,i3,i4])
    end

    return CFL
end

function get_CFL!(CFL::AbstractArray{T,5}, speed::AbstractArray{T,5}, coord) where T

    nmain, n2, n3, n4, n5 = size(speed)

    for i5 ∈ 1:n5, i4 ∈ 1:n4, i3 ∈ 1:n3, i2 ∈ 1:n2, imain ∈ 1:nmain
        CFL[imain,i2,i3,i4,i5] = abs(coord.cell_width[imain] / speed[imain,i2,i3,i4,i5])
    end

    return CFL
end

function get_CFL!(CFL::AbstractArray{T,6}, speed::AbstractArray{T,6}, coord) where T

    nmain, n2, n3, n4, n5, n6 = size(speed)

    for i6 ∈ 1:n6, i5 ∈ 1:n5, i4 ∈ 1:n4, i3 ∈ 1:n3, i2 ∈ 1:n2, imain ∈ 1:nmain
        CFL[imain,i2,i3,i4,i5,i6] = abs(coord.cell_width[imain] / speed[imain,i2,i3,i4,i5,i6])
    end

    return CFL
end

end #utils
