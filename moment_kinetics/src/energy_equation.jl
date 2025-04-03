"""
"""
module energy_equation

export energy_equation!
export neutral_energy_equation!

using ..calculus: derivative!
using ..looping
using ..timer_utils

"""
evolve the parallel pressure by solving the energy equation
"""
@timeit global_timer energy_equation!(
                         ppar_out, fvec, moments, collisions, dt, spectral, composition,
                         ion_source_settings, num_diss_params) = begin

    @begin_s_r_z_region()

    dppar_dt = moments.ion.dppar_dt

    @loop_s_r_z is ir iz begin
        dppar_dt[iz,ir,is] = (-fvec.upar[iz,ir,is]*moments.ion.dppar_dz_upwind[iz,ir,is]
                              - moments.ion.dqpar_dz[iz,ir,is]
                              - 3.0*fvec.ppar[iz,ir,is]*moments.ion.dupar_dz[iz,ir,is])
    end


    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active
            @views source_amplitude = moments.ion.external_source_pressure_amplitude[:, :, index]
            @loop_s_r_z is ir iz begin
                dppar_dt[iz,ir,is] += source_amplitude[iz,ir]
            end
        end
    end

    diffusion_coefficient = num_diss_params.ion.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_s_r_z is ir iz begin
            dppar_dt[iz,ir,is] += diffusion_coefficient*moments.ion.d2ppar_dz2[iz,ir,is]
        end
    end

    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0
            @loop_s_r_z is ir iz begin
                dppar_dt[iz,ir,is] -=
                    charge_exchange*(
                        fvec.density_neutral[iz,ir,is]*fvec.ppar[iz,ir,is] -
                        fvec.density[iz,ir,is]*fvec.pz_neutral[iz,ir,is] -
                        fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is] *
                            (fvec.upar[iz,ir,is] - fvec.uz_neutral[iz,ir,is])^2)
            end
        end
        if abs(ionization) > 0.0
            @loop_s_r_z is ir iz begin
                dppar_dt[iz,ir,is] +=
                    ionization*fvec.density[iz,ir,is] * (
                        fvec.pz_neutral[iz,ir,is] +
                        fvec.density_neutral[iz,ir,is] *
                            (fvec.upar[iz,ir,is]-fvec.uz_neutral[iz,ir,is])^2)
            end
        end
    end

    @loop_s_r_z is ir iz begin
        ppar_out[iz,ir,is] += dt * dppar_dt[iz,ir,is]
    end

    return nothing
end

"""
evolve the neutral parallel pressure by solving the energy equation
"""
@timeit global_timer neutral_energy_equation!(
                         pz_out, fvec, moments, collisions, dt, spectral, composition,
                         neutral_source_settings, num_diss_params) = begin

    @begin_sn_r_z_region()

    dpz_dt = moments.neutral.dpz_dt

    @loop_sn_r_z isn ir iz begin
        dpz_dt[iz,ir,isn] = (-fvec.uz_neutral[iz,ir,isn]*moments.neutral.dpz_dz_upwind[iz,ir,isn]
                             - moments.neutral.dqz_dz[iz,ir,isn]
                             - 3.0*fvec.pz_neutral[iz,ir,isn]*moments.neutral.duz_dz[iz,ir,isn])
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_amplitude = moments.neutral.external_source_pressure_amplitude[:, :, index]
            @loop_s_r_z isn ir iz begin
                dpz_dt[iz,ir,isn] += source_amplitude[iz,ir]
            end
        end
    end

    diffusion_coefficient = num_diss_params.neutral.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_sn_r_z isn ir iz begin
            dpz_dt[iz,ir,isn] += diffusion_coefficient*moments.neutral.d2pz_dz2[iz,ir,isn]
        end
    end

    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0
            @loop_sn_r_z isn ir iz begin
                dpz_dt[iz,ir,isn] -=
                    charge_exchange*(
                        fvec.density[iz,ir,isn]*fvec.pz_neutral[iz,ir,isn] -
                        fvec.density_neutral[iz,ir,isn]*fvec.ppar[iz,ir,isn] -
                        fvec.density_neutral[iz,ir,isn]*fvec.density[iz,ir,isn] *
                            (fvec.uz_neutral[iz,ir,isn] - fvec.upar[iz,ir,isn])^2)
            end
        end
        if abs(ionization) > 0.0
            @loop_sn_r_z isn ir iz begin
                dpz_dt[iz,ir,isn] -=
                    ionization*fvec.density[iz,ir,isn]*fvec.pz_neutral[iz,ir,isn]
            end
        end
    end

    @loop_sn_r_z isn ir iz begin
        pz_out[iz,ir,isn] += dt * dpz_dt[iz,ir,isn]
    end

    return nothing
end

end
