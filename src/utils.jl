"""
Utility functions
"""
module utils

using ..moment_kinetics_input: mk_input

using DataStructures: OrderedDict
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
function get_unnormalized_parameters(input::Dict; Nnorm::Number, Tnorm::Number,
                                     Lnorm::Number, Bnorm::Number)
    io_input, evolve_moments, t_input, z_input, r_input, vpa_input, vperp_input,
    gyrophase_input, vz_input, vr_input, vzeta_input, composition, species, collisions,
    geometry, drive_input, num_diss_params = mk_input(input)

    if !(Nnorm isa Unitful.AbstractQuantity)
        Nnorm *= Unitful.m^(-3)
    end
    if !(Tnorm isa Unitful.AbstractQuantity)
        Tnorm *= eV
    end
    if !(Lnorm isa Unitful.AbstractQuantity)
        Lnorm *= Unitful.m
    end
    if !(Bnorm isa Unitful.AbstractQuantity)
        Bnorm *= Unitful.T
    end

    # Assume ions are Deuterium
    mi = 3.3435837724e-27*Unitful.kg

    # Cold-ion sound speed with constant input electron temperature used for
    # normalization
    cs = Unitful.upreferred(sqrt(2.0*Tnorm/mi))

    # Time normalization
    timenorm = Unitful.upreferred(Lnorm / cs)

    parameters = OrderedDict{String,Any}()
    parameters["run_name"] = io_input.run_name

    parameters["Nnorm"] = Nnorm
    parameters["Tnorm"] = Tnorm
    parameters["Lnorm"] = Lnorm

    parameters["Lz"] = Lnorm * z_input.L

    parameters["cs0"] = cs

    dt = t_input.dt * timenorm
    parameters["dt"] = dt
    parameters["moments output time step"] = dt * t_input.nwrite_moments
    parameters["dfns output time step"] = dt * t_input.nwrite_dfns
    parameters["total simulated time"] = dt * t_input.nstep

    parameters["T_e"] = Tnorm * composition.T_e
    parameters["T_wall"] = Tnorm * composition.T_wall

    parameters["CX_rate_coefficient"] = collisions.charge_exchange / Nnorm / timenorm
    parameters["ionization_rate_coefficient"] = collisions.ionization / Nnorm / timenorm

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

end #utils
