"""
Reference parameters

Reference parameters are not needed or used by the main part of the code, but define the
physical units of the simulation, and are needed for a few specific steps during setup
(e.g. calculation of normalised collision frequency).
"""
module reference_parameters

export setup_reference_parameters

using ..constants
using ..input_structs

"""
"""
function setup_reference_parameters(input_dict)
    # Get reference parameters for normalizations
    reference_parameter_section = copy(set_defaults_and_check_section!(
        input_dict, "reference_params";
        Bref=1.0,
        Lref=10.0,
        Nref=1.0e19,
        Tref=100.0,
        mref=deuteron_mass,
       ))
    reference_parameter_section["cref"] = sqrt(2.0 * proton_charge * reference_parameter_section["Tref"] / (reference_parameter_section["mref"]))
    reference_parameter_section["timeref"] = reference_parameter_section["Lref"] / reference_parameter_section["cref"]
    reference_parameter_section["Omegaref"] = proton_charge * reference_parameter_section["Bref"] / reference_parameter_section["mref"]

    reference_params = Dict_to_NamedTuple(reference_parameter_section)

    return reference_params
end

end
