"""
"""
module energy_equation

export energy_equation!
export neutral_energy_equation!

using ..calculus: derivative!
using ..looping

"""
evolve the parallel pressure by solving the energy equation
"""
function energy_equation!(ppar, fvec, moments, collisions, dt, spectral, composition,
                          ion_source_settings, num_diss_params)

    begin_s_r_z_region()

    @loop_s_r_z is ir iz begin
        ppar[iz,ir,is] += dt*(-fvec.upar[iz,ir,is]*moments.ion.dppar_dz_upwind[iz,ir,is]
                              - moments.ion.dqpar_dz[iz,ir,is]
                              - 3.0*fvec.ppar[iz,ir,is]*moments.ion.dupar_dz[iz,ir,is])
    end


    if ion_source_settings.active
        source_amplitude = moments.ion.external_source_pressure_amplitude
        @loop_s_r_z is ir iz begin
            ppar[iz,ir,is] += dt * source_amplitude[iz,ir]
        end
    end

    diffusion_coefficient = num_diss_params.ion.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_s_r_z is ir iz begin
            ppar[iz,ir,is] += dt*diffusion_coefficient*moments.ion.d2ppar_dz2[iz,ir,is]
        end
    end

    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        if abs(collisions.charge_exchange) > 0.0
            @loop_s_r_z is ir iz begin
                ppar[iz,ir,is] -=
                    dt*collisions.charge_exchange*(
                        fvec.density_neutral[iz,ir,is]*fvec.ppar[iz,ir,is] -
                        fvec.density[iz,ir,is]*fvec.pz_neutral[iz,ir,is] -
                        fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is] *
                            (fvec.upar[iz,ir,is] - fvec.uz_neutral[iz,ir,is])^2)
            end
        end
        if abs(collisions.ionization) > 0.0
            @loop_s_r_z is ir iz begin
                ppar[iz,ir,is] +=
                    dt*collisions.ionization*fvec.density[iz,ir,is] * (
                        fvec.pz_neutral[iz,ir,is] +
                        fvec.density_neutral[iz,ir,is] *
                            (fvec.upar[iz,ir,is]-fvec.uz_neutral[iz,ir,is])^2)
            end
        end
    end
end

"""
evolve the neutral parallel pressure by solving the energy equation
"""
function neutral_energy_equation!(pz, fvec, moments, collisions, dt, spectral,
                                  composition, neutral_source_settings, num_diss_params)

    begin_sn_r_z_region()

    @loop_sn_r_z isn ir iz begin
        pz[iz,ir,isn] += dt*(-fvec.uz_neutral[iz,ir,isn]*moments.neutral.dpz_dz_upwind[iz,ir,isn]
                             - moments.neutral.dqz_dz[iz,ir,isn]
                             - 3.0*fvec.pz_neutral[iz,ir,isn]*moments.neutral.duz_dz[iz,ir,isn])
    end

    if neutral_source_settings.active
        source_amplitude = moments.neutral.external_source_pressure_amplitude
        @loop_s_r_z isn ir iz begin
            pz[iz,ir,isn] += dt * source_amplitude[iz,ir]
        end
    end

    diffusion_coefficient = num_diss_params.neutral.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_sn_r_z isn ir iz begin
            pz[iz,ir,isn] += dt*diffusion_coefficient*moments.neutral.d2pz_dz2[iz,ir,isn]
        end
    end

    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        if abs(collisions.charge_exchange) > 0.0
            @loop_sn_r_z isn ir iz begin
                pz[iz,ir,isn] -=
                    dt*collisions.charge_exchange*(
                        fvec.density[iz,ir,isn]*fvec.pz_neutral[iz,ir,isn] -
                        fvec.density_neutral[iz,ir,isn]*fvec.ppar[iz,ir,isn] -
                        fvec.density_neutral[iz,ir,isn]*fvec.density[iz,ir,isn] *
                            (fvec.uz_neutral[iz,ir,isn] - fvec.upar[iz,ir,isn])^2)
            end
        end
        if abs(collisions.ionization) > 0.0
            @loop_sn_r_z isn ir iz begin
                pz[iz,ir,isn] -=
                    dt*collisions.ionization*fvec.density[iz,ir,isn]*fvec.pz_neutral[iz,ir,isn]
            end
        end
    end
end

end
