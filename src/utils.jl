"""
Utility functions
"""
module utils

export get_unnormalized_parameters, print_unnormalized_parameters, to_seconds, to_minutes,
       to_hours

using ..moment_kinetics_input: mk_input

using Dates
using OrderedCollections
using TOML
using Unitful

Unitful.@unit eV "eV" "electron volt" 1.602176634e-19*Unitful.J true

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
        drive_input, external_source_settings, num_diss_params, manufactured_solns_input,
        reference_parameters = mk_input(input)

    Nnorm = reference_parameters.Nref * Unitful.m^(-3)
    Tnorm = reference_parameters.Tref * eV
    Lnorm = reference_parameters.Lref * Unitful.m
    Bnorm = reference_parameters.Bref * Unitful.T
    cnorm = reference_parameters.cref * Unitful.m / Unitful.s
    timenorm = reference_parameters.timeref * Unitful.s

    # Assume single ion species so normalised ion mass is always 1
    mi = reference_parameters.mnorm * Unitful.kg

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

end #utils
