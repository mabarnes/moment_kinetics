"""
"""
module krook_collisions

export setup_krook_collisions_input, krook_collisions!, electron_krook_collisions!,
       get_electron_krook_collisions_term

using ..looping
using ..debugging
using ..input_structs: krook_collisions_input, set_defaults_and_check_section!
using ..jacobian_matrices
using ..moment_kinetics_structs
using ..timer_utils 
using ..type_definitions
using ..collision_frequencies
using ..reference_parameters
using OrderedCollections: OrderedDict

"""
Function for reading Krook collision operator input parameters. 
Structure the namelist as follows.

[krook_collisions]
use_krook = true
nuii0 = 1.0
frequency_option = "manual"
"""
function setup_krook_collisions_input(toml_input::AbstractDict, warn_unexpected::Bool)
    reference_params = setup_reference_parameters(toml_input, warn_unexpected)
    # get reference collision frequency
    nuii_krook_default = get_reference_collision_frequency_ii(reference_params)
    nuee_krook_default = get_reference_collision_frequency_ee(reference_params)
    nuei_krook_default = get_reference_collision_frequency_ei(reference_params)
    # read the input toml and specify a sensible default    
    input_section = set_defaults_and_check_section!(
        toml_input, "krook_collisions", warn_unexpected;
        # begin default inputs (as kwargs)
        use_krook = false,
        nuii0 = -1.0,
        nuee0 = -1.0,
        nuei0 = -1.0,
        frequency_option = "reference_parameters")
       
    # ensure that the collision frequency is consistent with the input option
    frequency_option = input_section["frequency_option"]
    if frequency_option == "reference_parameters"
        input_section["nuii0"] = nuii_krook_default
        input_section["nuee0"] = nuee_krook_default
        input_section["nuei0"] = nuei_krook_default
    elseif frequency_option == "collisionality_scan"
        input_section["nuii0"] *= nuii_krook_default
        input_section["nuee0"] *= nuee_krook_default
        input_section["nuei0"] *= nuei_krook_default
    elseif frequency_option == "manual" 
        # use the frequency from the input file
        # do nothing
    else
        error("Invalid option [krook_collisions] "
              * "frequency_option=$(frequency_option) passed")
    end
    # finally, ensure prefactor < 0 if use_krook is false
    # so that prefactor > 0 is the only check required in the rest of the code
    if !input_section["use_krook"]
        input_section["nuii0"] = -1.0
        input_section["nuee0"] = -1.0
        input_section["nuei0"] = -1.0
    end
    input = OrderedDict(Symbol(k)=>v for (k,v) in input_section)
    return krook_collisions_input(; input...)
end


"""
Add collision operator

Currently Krook collisions
"""
@timeit global_timer krook_collisions!(
                         pdf_out, fvec_in, moments, composition, collisions, vperp, vpa,
                         dt) = begin
    @begin_s_r_z_region()

    if vperp.n == 1
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        Maxwellian_prefactor = 1.0 / π^1.5
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).
    if moments.evolve_p && moments.evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                # For 1V need to use parallel temperature for Maxwellian in Krook
                # operator, and for consistency with old 1D1V results also calculate
                # collision frequency using parallel temperature.
                Krook_vth = sqrt(3.0) * vth
                adjust_1V = 1.0 / sqrt(3.0)
            else
                Krook_vth = vth
                adjust_1V = 1.0
            end
            nu_ii = get_collision_frequency_ii(collisions, n, Krook_vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - Maxwellian_prefactor * adjust_1V * exp(-(vpa.grid[ivpa]*adjust_1V)^2
                                                              - (vperp.grid[ivperp]*adjust_1V)^2))
            end
        end
    elseif moments.evolve_p
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                # For 1V need to use parallel temperature for Maxwellian in Krook
                # operator, and for consistency with old 1D1V results also calculate
                # collision frequency using parallel temperature.
                Krook_vth = sqrt(3.0) * vth
                adjust_1V = 1.0 / sqrt(3.0)
            else
                Krook_vth = vth
                adjust_1V = 1.0
            end
            nu_ii = get_collision_frequency_ii(collisions, n, Krook_vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - Maxwellian_prefactor * adjust_1V * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])*adjust_1V)^2
                                                              - (vperp.grid[ivperp])^2)*adjust_1V)
            end
        end
    elseif moments.evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                # For 1V need to use parallel temperature for Maxwellian in Krook
                # operator, and for consistency with old 1D1V results also calculate
                # collision frequency using parallel temperature.
                Krook_vth = sqrt(3.0) * vth
                vth_prefactor = 1.0 / Krook_vth
            else
                Krook_vth = vth
                vth_prefactor = 1.0 / Krook_vth^3
            end
            nu_ii = get_collision_frequency_ii(collisions, n, Krook_vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - vth_prefactor * Maxwellian_prefactor * exp(-(vpa.grid[ivpa]/Krook_vth)^2
                                                                  - (vperp.grid[ivperp]/Krook_vth)^2))
            end
        end
    elseif moments.evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                # For 1V need to use parallel temperature for Maxwellian in Krook
                # operator, and for consistency with old 1D1V results also calculate
                # collision frequency using parallel temperature.
                Krook_vth = sqrt(3.0) * vth
                vth_prefactor = 1.0 / Krook_vth
            else
                Krook_vth = vth
                vth_prefactor = 1.0 / Krook_vth^3
            end
            nu_ii = get_collision_frequency_ii(collisions, n, Krook_vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                 - vth_prefactor * Maxwellian_prefactor
                 * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/Krook_vth)^2
                           - (vperp.grid[ivperp]/Krook_vth)^2))
            end
        end
    else
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                # For 1V need to use parallel temperature for Maxwellian in Krook
                # operator.
                Krook_vth = sqrt(3.0) * vth
                vth_prefactor = 1.0 / Krook_vth
            else
                Krook_vth = vth
                vth_prefactor = 1.0 / Krook_vth^3
            end
            nu_ii = get_collision_frequency_ii(collisions, n, Krook_vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - n * vth_prefactor * Maxwellian_prefactor
                     * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/Krook_vth)^2
                           - (vperp.grid[ivperp]/Krook_vth)^2))
            end
        end
    end

    return nothing
end

"""
Add Krook collision operator for electrons

Note that this function operates on a single point in `r`, so `pdf_out`, `pdf_in`,
`dens_in`, `upar_in`, `upar_ion_in`, and `vth_in` should have no r-dimension.
"""
@timeit global_timer electron_krook_collisions!(
                         pdf_out, pdf_in, dens_in, upar_in, upar_ion_in, vth_in,
                         collisions, vperp, vpa, dt) = begin
    @begin_anyzv_z_region()

    if vperp.n == 1
        Maxwellian_prefactor = 1.0 / sqrt(π)
    else
        Maxwellian_prefactor = 1.0 / π^1.5
    end

    # For now, electrons are always fully moment-kinetic
    evolve_density = true
    evolve_upar = true
    evolve_p = true

    if vperp.n > 1 && (evolve_density || evolve_upar || evolve_p)
        error("Krook collisions not implemented for 2V moment-kinetic cases yet")
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).
    if evolve_p && evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            if vperp.n == 1
                # For 1V need to use parallel temperature for Maxwellian in Krook
                # operator, and for consistency with old 1D1V results also calculate
                # collision frequency using parallel temperature.
                Krook_vth = sqrt(3.0) * vth
                adjust_1V = 1.0 / sqrt(3.0)
            else
                Krook_vth = vth
                adjust_1V = 1.0
            end
            nu_ee = get_collision_frequency_ee(collisions, n, Krook_vth)
            nu_ei = get_collision_frequency_ei(collisions, n, Krook_vth)

            # e-i collisions push electrons towards a Maxwellian drifting at the ion
            # parallel flow, so need a corresponding normalised parallel velocity
            # coordinate.
            # For now, assume there is only one ion species rather than bothering to
            # calculate an average ion flow speed, or sum over ion species here.
            @. vpa.scratch = vpa.grid + (upar_ion_in[iz,1] - upar_in[iz]) / vth

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - Maxwellian_prefactor * adjust_1V *
                               exp(-(vpa.grid[ivpa]*adjust_1V)^2 - (vperp.grid[ivperp]*adjust_1V)^2))
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - Maxwellian_prefactor * adjust_1V *
                                 exp(-(vpa.scratch[ivpa]*adjust_1V)^2 - (vperp.grid[ivperp]*adjust_1V)^2))
                   )
            end
        end
    elseif evolve_p
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - Maxwellian_prefactor * exp(-((vpa.grid[ivpa] - upar_in[iz])/vth)^2
                                                          - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - Maxwellian_prefactor * exp(-((vpa.grid[ivpa] - upar_ion_in[iz,1])/vth)^2
                                                            - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    elseif evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            # e-i collisions push electrons towards a Maxwellian drifting at the ion
            # parallel flow, so need a corresponding normalised parallel velocity
            # coordinate.
            # For now, assume there is only one ion species rather than bothering to
            # calculate an average ion flow speed, or sum over ion species here.
            @. vpa.scratch = vpa.grid + (upar_ion_in[iz,1] - upar_in[iz])

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - 1.0 / vth * Maxwellian_prefactor * exp(-(vpa.grid[ivpa] / vth)^2
                                                                      - (vperp.grid[ivperp] / vth)^2))
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - 1.0 / vth * Maxwellian_prefactor * exp(-(vpa.scratch[ivpa] / vth)^2
                                                                        - (vperp.grid[ivperp] / vth)^2))
                   )
            end
        end
    elseif evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - 1.0 / vth * Maxwellian_prefactor
                             * exp(-((vpa.grid[ivpa] - upar_in[iz]) / vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ei * (pdf_in[ivpa,ivperp,iz]
                               - 1.0 / vth * Maxwellian_prefactor
                               * exp(-((vpa.grid[ivpa] - upar_ion_in[iz,1]) / vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    else
        @loop_z iz begin
            n = dens_in[iz]
            vth = vth_in[iz]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz] -= dt * (
                    nu_ee * (pdf_in[ivpa,ivperp,iz]
                             - n * vth_prefactor * Maxwellian_prefactor
                             * exp(-((vpa.grid[ivpa] - upar_in[iz])/vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ee * (pdf_in[ivpa,ivperp,iz]
                               - n * vth_prefactor * Maxwellian_prefactor
                               * exp(-((vpa.grid[ivpa] - upar_ion_in[iz,1])/vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    end

    return nothing
end

function get_electron_krook_collisions_term(sub_terms::ElectronSubTerms)

    nuee0 = sub_terms.nuee0
    nuei0 = sub_terms.nuei0
    collisions = sub_terms.collisions

    if collisions === nothing || nuee0 ≤ 0.0 && nuei0 ≤ 0.0
        return NullTerm()
    end

    # Terms from `electron_krook_collisions!()`
    #   nu_ee * (pdf_in[ivpa,ivperp,iz]
    #            - Maxwellian_prefactor * krook_adjust_1V *
    #              exp(-(wpa*krook_adjust_1V)^2 - (wperp*krook_adjust_1V)^2))
    #   + nu_ei * (pdf_in[ivpa,ivperp,iz]
    #              - Maxwellian_prefactor * krook_adjust_1V *
    #                exp(-((wpa + (upar_ion - upar)/vth)*krook_adjust_1V)^2 - (wperp*krook_adjust_1V)^2))

    krook_adjust_vth_1V = sub_terms.krook_adjust_vth_1V
    krook_adjust_1V = sub_terms.krook_adjust_1V
    Maxwellian_prefactor = sub_terms.Maxwellian_prefactor

    n = sub_terms.n
    u = sub_terms.u
    u_ion = sub_terms.u_ion
    p = sub_terms.p
    vth = sub_terms.vth
    nu_ee = get_collision_frequency_ee(collisions, n, vth * krook_adjust_vth_1V)
    nu_ei = get_collision_frequency_ei(collisions, n, vth * krook_adjust_vth_1V)
    wperp = sub_terms.wperp
    wpa = sub_terms.wpa
    f = sub_terms.f

    term = (
            nu_ee * (f
                     - Maxwellian_prefactor * krook_adjust_1V * exp(-krook_adjust_1V^2 * (wpa^2 + wperp^2))
                    )
            + nu_ei * (f
                       - Maxwellian_prefactor * krook_adjust_1V
                       * exp(-krook_adjust_1V^2 * ((wpa + (u_ion - u) * vth^(-1))^2
                                                   + wperp^2))
                      )
           )

    return term
end

end # krook_collisions
