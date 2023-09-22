"""
"""
module krook_collisions

using ..constants: epsilon0, proton_charge
using ..looping

"""
Calculate normalized collision frequency at reference parameters for Coulomb collisions.

Currently valid only for hydrogenic ions (Z=1)
"""
function setup_krook_collisions(reference_parameters)
    Nref = reference_parameters.Nref
    Tref = reference_parameters.Tref
    mref = reference_parameters.mref
    timeref = reference_parameters.timeref

    Nref_per_cm3 = Nref * 1.0e-6

    # Coulomb logarithm at reference parameters for same-species ion-ion collisions, using
    # NRL formulary. Formula given for n in units of cm^-3 and T in units of eV.
    logLambda_ii = 23.0 - log(sqrt(2.0*Nref_per_cm3) / Tref^1.5)

    # Collision frequency, using \hat{\nu} from Appendix, p. 277 of Helander "Collisional
    # Transport in Magnetized Plasmas" (2002).
    T0_Joules = Tref * proton_charge # Tref in Joules
    mi_kg = mref # mi in kg
    vth_i0 = sqrt(2.0 * T0_Joules / mi_kg) # vth_i at reference parameters in m.s^-1
    nu_ii0_per_s = Nref * proton_charge^4 * logLambda_ii  /
                   (4.0 * π * epsilon0^2 * mi_kg^2 * vth_i0^3) # s^-1
    nu_ii0 = nu_ii0_per_s * timeref

    return nu_ii0
end

"""
Add collision operator

Currently Krook collisions
"""
function krook_collisions!(pdf_out, fvec_in, moments, composition, collisions, vperp, vpa, dt)
    begin_s_r_z_region()

    if vperp.n > 1
        error("Krook collisions not implemented for 2V case yet")
    end

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).

    if moments.evolve_ppar && moments.evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_s_r_z is ir iz begin
            n = fvec_in.density[iz,ir,is]
            T = (moments.charged.vth[iz,ir,is])^2
            nu_ii = collisions.krook_collision_frequency_prefactor * n * T^(-1.5)
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
            T = vth^2
            nu_ii = collisions.krook_collision_frequency_prefactor * n * T^(-1.5)
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
            T = vth^2
            nu_ii = collisions.krook_collision_frequency_prefactor * n * T^(-1.5)
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
            T = vth^2
            nu_ii = collisions.krook_collision_frequency_prefactor * n * T^(-1.5)
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
            T = vth^2
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            nu_ii = collisions.krook_collision_frequency_prefactor * n * T^(-1.5)
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
