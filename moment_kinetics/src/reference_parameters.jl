"""
Reference parameters

Reference parameters are not needed or used by the main part of the code, but define the
physical units of the simulation, and are needed for a few specific steps during setup
(e.g. calculation of normalised collision frequency).
"""
module reference_parameters

export setup_reference_parameters
export get_reference_collision_frequency_ii, get_reference_collision_frequency_ee,
       get_reference_collision_frequency_ei

using ..constants
using ..input_structs

"""
"""
function setup_reference_parameters(input_dict, warn_unexpected::Bool)
    # Get reference parameters for normalizations
    reference_parameter_section = copy(set_defaults_and_check_section!(
        input_dict, "reference_params", warn_unexpected;
        Bref=1.0,
        Lref=10.0,
        Nref=1.0e19,
        Tref=100.0,
        mref=deuteron_mass,
       ))
    reference_parameter_section["cref"] = sqrt(proton_charge * reference_parameter_section["Tref"] / (reference_parameter_section["mref"]))
    reference_parameter_section["timeref"] = reference_parameter_section["Lref"] / reference_parameter_section["cref"]
    reference_parameter_section["Omegaref"] = proton_charge * reference_parameter_section["Bref"] / reference_parameter_section["mref"]

    Nref_per_cm3 = reference_parameter_section["Nref"] * 1.0e-6
    Tref = reference_parameter_section["Tref"]

    reference_parameter_section["me"] = electron_mass

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

"""
Calculate normalized ion-ion collision frequency for Coulomb collisions using reference
density, mass and speed.
Note this is not the thermal collision frequency evaluated with reference parameters, but
rather the quantity that when multiplied by \$\\hat{n}s/v_{Ts}^3\$ gives the
de-dimensionalised thermal collision frequency - the difference is a factor of
\$2^{3/2}\$.

Currently valid only for hydrogenic ions (Z=1)
"""
function get_reference_collision_frequency_ii(reference_params)
    Nref = reference_params.Nref
    Tref = reference_params.Tref
    mref = reference_params.mref
    timeref = reference_params.timeref
    cref = reference_params.cref
    logLambda_ii = reference_params.logLambda_ii

    # Collision frequency, using \hat{\nu} from Appendix, p. 277 of Helander "Collisional
    # Transport in Magnetized Plasmas" (2002).
    nu_ii0_per_s = Nref * proton_charge^4 * logLambda_ii  /
                   (4.0 * π * epsilon0^2 * mref^2 * cref^3) # s^-1
    nu_ii0 = nu_ii0_per_s * timeref

    return nu_ii0
end

"""
Calculate normalized electron-electron collision frequency for Coulomb collisions using
reference density, mass and speed.
Note this is not the thermal collision frequency evaluated with reference parameters, but
rather the quantity that when multiplied by \$\\hat{n}s/v_{Ts}^3\$ gives the
de-dimensionalised thermal collision frequency - the difference is a factor of
\$2^{3/2}\$.
"""
function get_reference_collision_frequency_ee(reference_params)
    Nref = reference_params.Nref
    Tref = reference_params.Tref
    me = reference_params.me
    timeref = reference_params.timeref
    cref = reference_params.cref
    logLambda_ee = reference_params.logLambda_ee

    # Collision frequency, using \hat{\nu} from Appendix, p. 277 of Helander "Collisional
    # Transport in Magnetized Plasmas" (2002).
    # Note the electron thermal speed used in the code is normalised to cref, so we use
    # cref in these two formulas rather than a reference electron thermal speed, so that
    # when multiplied by the normalised electron thermal speed we get the correct
    # normalised collision frequency.
    nu_ee0_per_s = Nref * proton_charge^4 * logLambda_ee  /
                   (4.0 * π * epsilon0^2 * me^2 * cref^3) # s^-1
    nu_ee0 = nu_ee0_per_s * timeref

    return nu_ee0
end

"""
Calculate normalized electron-ion collision frequency for Coulomb collisions using
reference density, mass and speed.
Note this is not the thermal collision frequency evaluated with reference parameters, but
rather the quantity that when multiplied by \$\\hat{n}s/v_{Ts}^3\$ gives the
de-dimensionalised thermal collision frequency - the difference is a factor of
\$2^{3/2}\$.

Currently valid only for hydrogenic ions (Z=1)
"""
function get_reference_collision_frequency_ei(reference_params)
    Nref = reference_params.Nref
    Tref = reference_params.Tref
    me = reference_params.me
    timeref = reference_params.timeref
    cref = reference_params.cref
    logLambda_ei = reference_params.logLambda_ei

    # Collision frequency, using \hat{\nu} from Appendix, p. 277 of Helander "Collisional
    # Transport in Magnetized Plasmas" (2002).
    # Note the electron thermal speed used in the code is normalised to cref, so we use
    # cref in these two formulas rather than a reference electron thermal speed, so that
    # when multiplied by the normalised electron thermal speed we get the correct
    # normalised collision frequency.
    nu_ei0_per_s = Nref * proton_charge^4 * logLambda_ei  /
                   (4.0 * π * epsilon0^2 * me^2 * cref^3) # s^-1
    nu_ei0 = nu_ei0_per_s * timeref

    return nu_ei0
end

end
