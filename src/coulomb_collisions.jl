"""
"""
module coulomb_collisions

using ..constants: epsilon0, proton_charge
using ..looping

"""
Calculate normalized collision frequency at reference parameters for Coulomb collisions.

Currently valid only for hydrogenic ions (Z=1)
"""
function setup_coulomb_collisions(composition)
    Nnorm = composition.Nnorm
    Tnorm = composition.Tnorm
    mnorm = composition.mnorm
    timenorm = composition.timenorm

    Nnorm_per_cm3 = Nnorm * 1.0e-6

    # Coulomb logarithm at reference parameters for same-species ion-ion collisions, using
    # NRL formulary. Formula given for n in units of cm^-3 and T in units of eV.
    logLambda_ii = 23.0 - log(sqrt(2.0*Nnorm_per_cm3) / Tnorm^1.5)

    # Collision frequency, using \hat{\nu} from Appendix, p. 277 of Helander "Collisional
    # Transport in Magnetized Plasmas" (2002).
    T0_Joules = Tnorm * proton_charge # Tnorm in Joules
    mi_kg = composition.mi * mnorm # mi in kg
    vth_i0 = sqrt(2.0 * T0_Joules / mi_kg) # vth_i at reference parameters in m.s^-1
    nu_ii0_per_s = Nnorm * proton_charge^4 * logLambda_ii  /
                   (4.0 * π * epsilon0^2 * mi_kg^2 * vth_i0^3) # s^-1
    nu_ii0 = nu_ii0_per_s * timenorm

    return nu_ii0
end

"""
Add collision operator

Currently Krook collisions
"""
function coulomb_collisions!(pdf_out, fvec_in, moments, composition, collisions, vpa, dt)
    begin_s_r_z_region()

    # Note: do not need 1/sqrt(pi) for the 'Maxwellian' term because the pdf is already
    # normalized by sqrt(pi) (see velocity_moments.integrate_over_vspace).

    if moments.evolve_ppar && moments.evolve_upar
        # Compared to evolve_upar version, grid is already normalized by vth, and multiply
        # through by vth, remembering pdf is already multiplied by vth
        @loop_s_r_z is ir iz begin
            if is ∈ composition.ion_species_range
                n = fvec_in.density[iz,ir,is]
                T = fvec_in.ppar[iz,ir,is] / n
                nu_ii = collisions.coulomb_collision_frequency_prefactor * n * T^(-1.5)
                @loop_vpa ivpa begin
                    pdf_out[ivpa,iz,ir,is] -= dt * nu_ii *
                        (fvec_in.pdf[ivpa,iz,ir,is]
                         - exp(-vpa.grid[ivpa]^2))
                end
            end
        end
    elseif moments.evolve_ppar
        # Compared to full-f collision operater, multiply through by vth, remembering pdf
        # is already multiplied by vth, and grid is already normalized by vth
        @loop_s_r_z is ir iz begin
            if is ∈ composition.ion_species_range
                n = fvec_in.density[iz,ir,is]
                T = fvec_in.ppar[iz,ir,is] / n
                vth = moments.vth[iz,ir,is]
                nu_ii = collisions.coulomb_collision_frequency_prefactor * n * T^(-1.5)
                @loop_vpa ivpa begin
                    pdf_out[ivpa,iz,ir,is] -= dt * nu_ii *
                        (fvec_in.pdf[ivpa,iz,ir,is]
                         - exp(-(vpa.grid[ivpa] - fvec_in.upar[iz,ir,is]/vth)^2))
                end
            end
        end
    elseif moments.evolve_upar
        # Compared to evolve_density version, grid is already shifted by upar
        @loop_s_r_z is ir iz begin
            if is ∈ composition.ion_species_range
                n = fvec_in.density[iz,ir,is]
                T = fvec_in.ppar[iz,ir,is] / n
                vth = moments.vth[iz,ir,is]
                nu_ii = collisions.coulomb_collision_frequency_prefactor * n * T^(-1.5)
                @loop_vpa ivpa begin
                    pdf_out[ivpa,iz,ir,is] -= dt * nu_ii *
                        (fvec_in.pdf[ivpa,iz,ir,is]
                         - 1.0 / vth * exp(-(vpa.grid[ivpa] / vth)^2))
                end
            end
        end
    elseif moments.evolve_density
        # Compared to full-f collision operater, divide through by density, remembering
        # that pdf is already normalized by density
        @loop_s_r_z is ir iz begin
            if is ∈ composition.ion_species_range
                n = fvec_in.density[iz,ir,is]
                T = fvec_in.ppar[iz,ir,is] / n
                vth = moments.vth[iz,ir,is]
                nu_ii = collisions.coulomb_collision_frequency_prefactor * n * T^(-1.5)
                @loop_vpa ivpa begin
                    pdf_out[ivpa,iz,ir,is] -= dt * nu_ii *
                        (fvec_in.pdf[ivpa,iz,ir,is]
                         - 1.0 / vth
                           * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is]) / vth)^2))
                end
            end
        end
    else
        @loop_s_r_z is ir iz begin
            if is ∈ composition.ion_species_range
                n = fvec_in.density[iz,ir,is]
                T = fvec_in.ppar[iz,ir,is] / n
                vth = moments.vth[iz,ir,is]
                nu_ii = collisions.coulomb_collision_frequency_prefactor * n * T^(-1.5)
                @loop_vpa ivpa begin
                    pdf_out[ivpa,iz,ir,is] -= dt * nu_ii *
                        (fvec_in.pdf[ivpa,iz,ir,is]
                         - n / vth
                           * exp(-((vpa.grid[ivpa] - fvec_in.upar[iz,ir,is])/vth)^2))
                end
            end
        end
    end

    return nothing
end

end # coulomb_collisions
