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

    Nref_per_cm3 = reference_parameter_section["Nref"] * 1.0e-6
    Tref = reference_parameter_section["Tref"]

    # Coulomb logarithm at reference parameters for same-species, singly-charged ion-ion
    # collisions, using NRL formulary. Formula given for n in units of cm^-3 and T in
    # units of eV.
    reference_parameter_section["logLambda_ii"] = 23.0 - log(sqrt(2.0*Nref_per_cm3) / Tref^1.5)

    # Coulomb logarithm at reference parameters for electron-electron collisions, using
    # NRL formulary. Formula given for n in units of cm^-3 and T in units of eV.
    reference_parameter_section["logLambda_ee"] = 23.5 - log(sqrt(Nref_per_cm3) / Tref^1.25) - sqrt(1.0e-5 + (log(Tref) -2.0)^2 / 16.0)

    # Coulomb logarithm at reference parameters for electron-ion collisions with
    # singly-charged ions, using NRL formulary. Formula given for n in units of cm^-3 and
    # T in units of eV.
    # Note: assume reference temperature is the same for ions and electrons, so ignore
    # case in NRL formulary where Te < Ti*me/mi.
    if Tref < 10.0
        reference_parameter_section["logLambda_ei"] = 23.0 - log(sqrt(Nref_per_cm3) / Tref^1.5)
    else
        reference_parameter_section["logLambda_ei"] = 24.0 - log(sqrt(Nref_per_cm3) / Tref)
    end

    reference_params = Dict_to_NamedTuple(reference_parameter_section)

    return reference_params
end

end
