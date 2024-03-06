"""
Utility functions
"""
module utils

export get_unnormalized_parameters, print_unnormalized_parameters, to_seconds, to_minutes,
       to_hours

using ..communication
using ..constants
using ..looping
using ..moment_kinetics_input: mk_input
using ..reference_parameters

using Dates
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
    io_input, evolve_moments, t_params, z, z_spectral, r, r_spectral, vpa, vpa_spectral,
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

    dt = t_params.dt * timenorm
    parameters["dt"] = dt
    parameters["output time step"] = dt * t_params.nwrite
    parameters["total simulated time"] = dt * t_params.nstep

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

# Utility functions for timestepping

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
