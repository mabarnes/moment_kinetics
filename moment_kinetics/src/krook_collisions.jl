"""
"""
module krook_collisions

export setup_krook_collisions, get_collision_frequency, krook_collisions!

using ..constants: epsilon0, proton_charge
using ..looping

"""
Calculate normalized collision frequency at reference parameters for Coulomb collisions.

Currently valid only for hydrogenic ions (Z=1)
"""
function setup_krook_collisions(reference_params)
    Nref = reference_params.Nref
    Tref = reference_params.Tref
    mref = reference_params.mref
    timeref = reference_params.timeref
    cref = reference_params.cref

    Nref_per_cm3 = Nref * 1.0e-6

    # Coulomb logarithm at reference parameters for same-species ion-ion collisions, using
    # NRL formulary. Formula given for n in units of cm^-3 and T in units of eV.
    logLambda_ii = 23.0 - log(sqrt(2.0*Nref_per_cm3) / Tref^1.5)

    # Collision frequency, using \hat{\nu} from Appendix, p. 277 of Helander "Collisional
    # Transport in Magnetized Plasmas" (2002).
    nu_ii0_per_s = Nref * proton_charge^4 * logLambda_ii  /
                   (4.0 * π * epsilon0^2 * mref^2 * cref^3) # s^-1
    nu_ii0 = nu_ii0_per_s * timeref

    return nu_ii0
end

"""
    get_collision_frequency(collisions, n, vth)

Calculate the collision frequency, depending on the settings/parameters in `collisions`,
for the given density `n` and thermal speed `vth`.

`n` and `vth` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency(collisions, n, vth)
    if collisions.krook_collisions_option == "reference_parameters"
        return @. collisions.krook_collision_frequency_prefactor * n * vth^(-3)
    elseif collisions.krook_collisions_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. collisions.krook_collision_frequency_prefactor + 0.0 * n
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
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
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
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
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
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
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
            vth = moments.charged.vth[iz,ir,is]
            nu_ii = get_collision_frequency(collisions, n, vth)
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
            vth = moments.charged.vth[iz,ir,is]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ii = get_collision_frequency(collisions, n, vth)
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

end # krook_collisions
