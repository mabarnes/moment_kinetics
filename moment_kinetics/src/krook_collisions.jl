"""
"""
module krook_collisions

export setup_krook_collisions!, get_collision_frequency_ii, get_collision_frequency_ee,
       get_collision_frequency_ei, krook_collisions!, electron_krook_collisions!

using ..constants
using ..looping

"""
Calculate normalized collision frequency at reference parameters for Coulomb collisions.

Currently valid only for hydrogenic ions (Z=1)
"""
function setup_krook_collisions!(collisions, reference_params, scan_input)
    Nref = reference_params.Nref
    Tref = reference_params.Tref
    mref = reference_params.mref
    timeref = reference_params.timeref
    cref = reference_params.cref
    logLambda_ii = reference_params.logLambda_ii
    logLambda_ee = reference_params.logLambda_ee
    logLambda_ei = reference_params.logLambda_ei

    # Collision frequencies, using \hat{\nu} from Appendix, p. 277 of Helander
    # "Collisional Transport in Magnetized Plasmas" (2002).
    nu_ii0_per_s = Nref * proton_charge^4 * logLambda_ii /
                   (4.0 * π * epsilon0^2 * mref^2 * cref^3) # s^-1
    nu_ii0 = nu_ii0_per_s * timeref

    # Note the electron thermal speed used in the code is normalised to cref, so we use
    # cref in these two formulas rather than a reference electron thermal speed, so that
    # when multiplied by the normalised electron thermal speed we get the correct
    # normalised collision frequency.
    nu_ee0_per_s = Nref * proton_charge^4 * logLambda_ee /
                   (4.0 * π * epsilon0^2 * electron_mass^2 * cref^3) # s^-1
    nu_ee0 = nu_ee0_per_s * timeref

    nu_ei0_per_s = Nref * proton_charge^4 * logLambda_ei /
                   (4.0 * π * epsilon0^2 * electron_mass^2 * cref^3) # s^-1
    nu_ei0 = nu_ei0_per_s * timeref

    collisions.krook_collisions_option = get(scan_input, "krook_collisions_option", "none")
    if collisions.krook_collisions_option == "reference_parameters"
        collisions.krook_collision_frequency_prefactor_ii = nu_ii0
    elseif collisions.krook_collisions_option == "manual" # get the frequency from the input file
        collisions.krook_collision_frequency_prefactor_ii = get(scan_input, "nuii_krook", nu_ii0)
    elseif collisions.krook_collisions_option == "none"
        # By default, no krook collisions included
        collisions.krook_collision_frequency_prefactor_ii = -1.0
    else
        error("Invalid option "
              * "krook_collisions_option=$(collisions.krook_collisions_option) passed")
    end

    if collisions.krook_collisions_option == "reference_parameters"
        collisions.krook_collision_frequency_prefactor_ee = nu_ee0
    elseif collisions.krook_collisions_option == "manual" # get the frequency from the input file
        # If the "manual" option is used, the collision frequency is not multiplied by
        # vthe^(-3), so need to correct it to be evaluated with the electron thermal speed
        # at Tref for the default value.
        vthe_ref = sqrt(2.0 * Tref / electron_mass)
        collisions.krook_collision_frequency_prefactor_ee =
            get(scan_input, "nuee_krook", nu_ee0 * (cref/vthe_ref)^3)
    elseif collisions.krook_collisions_option == "none"
        # By default, no krook collisions included
        collisions.krook_collision_frequency_prefactor_ee = -1.0
    else
        error("Invalid option "
              * "krook_collisions_option=$(collisions.krook_collisions_option) passed")
    end

    if collisions.krook_collisions_option == "reference_parameters"
        collisions.krook_collision_frequency_prefactor_ei = nu_ei0
    elseif collisions.krook_collisions_option == "manual" # get the frequency from the input file
        # If the "manual" option is used, the collision frequency is not multiplied by
        # vthe^(-3), so need to correct it to be evaluated with the electron thermal speed
        # at Tref for the default value.
        vthe_ref = sqrt(2.0 * Tref / electron_mass)
        collisions.krook_collision_frequency_prefactor_ei =
            get(scan_input, "nuei_krook", nu_ei0 * (cref/vthe_ref)^3)
    elseif collisions.krook_collisions_option == "none"
        # By default, no krook collisions included
        collisions.krook_collision_frequency_prefactor_ei = -1.0
    else
        error("Invalid option "
              * "krook_collisions_option=$(collisions.krook_collisions_option) passed")
    end

    return nothing
end

"""
    get_collision_frequency_ii(collisions, n, vth)

Calculate the ion-ion collision frequency, depending on the settings/parameters in
`collisions`, for the given density `n` and thermal speed `vth`.

`n` and `vth` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency_ii(collisions, n, vth)
    if collisions.krook_collisions_option == "reference_parameters"
        return @. collisions.krook_collision_frequency_prefactor_ii * n * vth^(-3)
    elseif collisions.krook_collisions_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. collisions.krook_collision_frequency_prefactor_ii + 0.0 * n
    elseif collisions.krook_collisions_option == "none"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. 0.0 * n
    else
        error("Unrecognised option "
              * "krook_collisions_option=$(collisions.krook_collisions_option)")
    end
end

"""
    get_collision_frequency_ee(collisions, n, vthe)

Calculate the electron-electron collision frequency, depending on the settings/parameters
in `collisions`, for the given density `n` and electron thermal speed `vthe`.

`n` and `vthe` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency_ee(collisions, n, vthe)
    if collisions.krook_collisions_option == "reference_parameters"
        return @. collisions.krook_collision_frequency_prefactor_ee * n * vthe^(-3)
    elseif collisions.krook_collisions_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. collisions.krook_collision_frequency_prefactor_ee + 0.0 * n
    elseif collisions.krook_collisions_option == "none"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. 0.0 * n
    else
        error("Unrecognised option "
              * "krook_collisions_option=$(collisions.krook_collisions_option)")
    end
end

"""
    get_collision_frequency_ei(collisions, n, vthe)

Calculate the electron-electron collision frequency, depending on the settings/parameters
in `collisions`, for the given density `n` and electron thermal speed `vthe`.

`n` and `vthe` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency_ei(collisions, n, vthe)
    if collisions.krook_collisions_option == "reference_parameters"
        return @. collisions.krook_collision_frequency_prefactor_ei * n * vthe^(-3)
    elseif collisions.krook_collisions_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. collisions.krook_collision_frequency_prefactor_ei + 0.0 * n
    elseif collisions.krook_collisions_option == "none"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. 0.0 * n
    else
        error("Unrecognised option "
              * "krook_collisions_option=$(collisions.krook_collisions_option)")
    end
end

"""
Add collision operator

Currently Krook collisions
"""
function krook_collisions!(pdf_out, fvec_in, moments, composition, collisions, vperp, vpa, dt)
    begin_s_r_z_region()

    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Krook collisions not implemented for 2V moment-kinetic cases yet")
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).
    if moments.evolve_ppar && moments.evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2))
            end
        end
    elseif moments.evolve_ppar
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/vth)^2
                           - (vperp.grid[ivperp]/vth)^2))
            end
        end
    elseif moments.evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - 1.0 / vth * exp(-(vpa.grid[ivpa] / vth)^2
                                       - (vperp.grid[ivperp] / vth)^2))
            end
        end
    elseif moments.evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                 - 1.0 / vth
                 * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is]) / vth)^2
                           - (vperp.grid[ivperp]/vth)^2))
            end
        end
    else
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ii = get_collision_frequency_ii(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] -= dt * nu_ii *
                    (fvec_in.pdf[ivpa,ivperp,iz,ir,is]
                     - n * vth_prefactor
                     * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/vth)^2
                           - (vperp.grid[ivperp]/vth)^2))
            end
        end
    end

    return nothing
end

"""
Add Krook collision operator for electrons
"""
function electron_krook_collisions!(pdf_out, fvec_in, moments, composition, collisions,
                                    vperp, vpa, dt)
    begin_r_z_region()

    # For now, electrons are always fully moment-kinetic
    evolve_density = true
    evolve_upar = true
    evolve_ppar = true

    if vperp.n > 1 && (evolve_density || evolve_upar || evolve_ppar)
        error("Krook collisions not implemented for 2V moment-kinetic cases yet")
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).
    if evolve_ppar && evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_r_z ir iz begin
            n = fvec_in.electron_density[iz,ir]
            vth = moments.electron.vth[iz,ir]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            # e-i collisions push electrons towards a Maxwellian drifting at the ion
            # parallel flow, so need a corresponding normalised parallel velocity
            # coordinate.
            # For now, assume there is only one ion species rather than bothering to
            # calculate an average ion flow speed, or sum over ion species here.
            @. vpa.scratch = vpa.grid + (fvec_in.upar[iz,ir,1] - fvec_in.electron_upar[iz,ir]) / vth

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir] -= dt * (
                    nu_ee * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                             - exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2))
                    + nu_ei * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                               - exp(-vpa.scratch[ivpa]^2 - vperp.grid[ivperp]^2))
                   )
            end
        end
    elseif evolve_ppar
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_r_z ir iz begin
            n = fvec_in.electron_density[iz,ir]
            vth = moments.electron.vth[iz,ir]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir] -= dt * (
                    nu_ee * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                             - exp(-((vpa.grid[ivpa] - fvec_in.electron_upar[iz,ir])/vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ei * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                               - exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,1])/vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    elseif evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_r_z ir iz begin
            n = fvec_in.electron_density[iz,ir]
            vth = moments.electron.vth[iz,ir]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)

            # e-i collisions push electrons towards a Maxwellian drifting at the ion
            # parallel flow, so need a corresponding normalised parallel velocity
            # coordinate.
            # For now, assume there is only one ion species rather than bothering to
            # calculate an average ion flow speed, or sum over ion species here.
            @. vpa.scratch = vpa.grid + (fvec_in.upar[iz,ir,1] - fvec_in.electron_upar[iz,ir])

            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir] -= dt * (
                    nu_ee * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                             - 1.0 / vth * exp(-(vpa.grid[ivpa] / vth)^2
                                               - (vperp.grid[ivperp] / vth)^2))
                    + nu_ei * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                               - 1.0 / vth * exp(-(vpa.scratch[ivpa] / vth)^2
                                                 - (vperp.grid[ivperp] / vth)^2))
                   )
            end
        end
    elseif evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_r_z ir iz begin
            n = fvec_in.electron_density[iz,ir]
            vth = moments.electron.vth[iz,ir]
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir] -= dt * (
                    nu_ee * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                             - 1.0 / vth
                             * exp(-((vpa.grid[ivpa] - fvec_in.electron_upar[iz,ir]) / vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ei * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                               - 1.0 / vth
                               * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,1]) / vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    else
        @loop_r_z ir iz begin
            n = fvec_in.electron_density[iz,ir]
            vth = moments.electron.vth[iz,ir]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ee = get_collision_frequency_ee(collisions, n, vth)
            nu_ei = get_collision_frequency_ei(collisions, n, vth)
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir] -= dt * (
                    nu_ee * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                             - n * vth_prefactor
                             * exp(-((vpa.grid[ivpa] - fvec_in.electron_upar[iz,ir])/vth)^2
                                   - (vperp.grid[ivperp]/vth)^2))
                    # e-i collisions push electrons towards a Maxwellian drifting at the ion
                    # parallel flow, so need a corresponding normalised parallel velocity
                    # coordinate.
                    # For now, assume there is only one ion species rather than bothering to
                    # calculate an average ion flow speed, or sum over ion species here.
                    + nu_ee * (fvec_in.electron_pdf[ivpa,ivperp,iz,ir]
                               - n * vth_prefactor
                               * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,1])/vth)^2
                                     - (vperp.grid[ivperp]/vth)^2))
                   )
            end
        end
    end

    return nothing
end

end # krook_collisions
