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
                         p_out, fvec, moments, fields, collisions, dt, spectral,
                         composition, geometry, ion_source_settings,
                         num_diss_params) = begin

    @begin_s_r_z_region()

    upar = fvec.upar
    p = fvec.p
    ppar = moments.ion.ppar
    dupar_dz = moments.ion.dupar_dz
    dp_dr_upwind = moments.ion.dp_dr_upwind
    dp_dz_upwind = moments.ion.dp_dz_upwind
    dqpar_dz = moments.ion.dqpar_dz
    dp_dt = moments.ion.dp_dt
    vEr = fields.vEr
    vEz = fields.vEz
    bz = geometry.bzed

    @loop_s_r_z is ir iz begin
        dp_dt[iz,ir,is] = -(vEr[iz,ir] * dp_dr_upwind[iz,ir,is]
                            + (vEz[iz,ir] + bz[iz,ir] * upar[iz,ir,is]) * dp_dz_upwind[iz,ir,is]
                            + bz[iz,ir] * p[iz,ir,is] * dupar_dz[iz,ir,is]
                            + 2.0/3.0 * bz[iz,ir] * dqpar_dz[iz,ir,is]
                            + 2.0/3.0 * bz[iz,ir] * ppar[iz,ir,is]*moments.ion.dupar_dz[iz,ir,is])
    end


    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active
            @views source_amplitude = moments.ion.external_source_pressure_amplitude[:, :, index]
            @loop_s_r_z is ir iz begin
                dp_dt[iz,ir,is] += source_amplitude[iz,ir]
            end
        end
    end

    diffusion_coefficient = num_diss_params.ion.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_s_r_z is ir iz begin
            dp_dt[iz,ir,is] += diffusion_coefficient*moments.ion.d2ppar_dz2[iz,ir,is]
        end
    end

    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0
            @loop_s_r_z is ir iz begin
                dp_dt[iz,ir,is] -=
                    charge_exchange*(
                        fvec.density_neutral[iz,ir,is]*fvec.p[iz,ir,is] -
                        fvec.density[iz,ir,is]*fvec.p_neutral[iz,ir,is] -
                        1.0/3.0 * fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is] *
                            (fvec.upar[iz,ir,is] - fvec.uz_neutral[iz,ir,is])^2)
            end
        end
        if abs(ionization) > 0.0
            @loop_s_r_z is ir iz begin
                dp_dt[iz,ir,is] +=
                    ionization*fvec.density[iz,ir,is] * (
                        fvec.p_neutral[iz,ir,is] +
                        1.0/3.0 * fvec.density_neutral[iz,ir,is] *
                            (fvec.upar[iz,ir,is]-fvec.uz_neutral[iz,ir,is])^2)
            end
        end
    end

    @loop_s_r_z is ir iz begin
        p_out[iz,ir,is] += dt * dp_dt[iz,ir,is]
    end

    return nothing
end

"""
evolve the neutral parallel pressure by solving the energy equation
"""
@timeit global_timer neutral_energy_equation!(
                         p_out, fvec, moments, collisions, dt, spectral, composition,
                         neutral_source_settings, num_diss_params) = begin

    @begin_sn_r_z_region()

    dp_dt = moments.neutral.dp_dt

    @loop_sn_r_z isn ir iz begin
        dp_dt[iz,ir,isn] = (-fvec.uz_neutral[iz,ir,isn]*moments.neutral.dp_dz_upwind[iz,ir,isn]
                            -fvec.p_neutral[iz,ir,isn]*moments.neutral.duz_dz[iz,ir,isn]
                            - 2.0/3.0*moments.neutral.dqz_dz[iz,ir,isn]
                            - 2.0/3.0*moments.neutral.pz[iz,ir,isn]*moments.neutral.duz_dz[iz,ir,isn])
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_amplitude = moments.neutral.external_source_pressure_amplitude[:, :, index]
            @loop_s_r_z isn ir iz begin
                dp_dt[iz,ir,isn] += source_amplitude[iz,ir]
            end
        end
    end

    diffusion_coefficient = num_diss_params.neutral.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_sn_r_z isn ir iz begin
            dp_dt[iz,ir,isn] += diffusion_coefficient*moments.neutral.d2pz_dz2[iz,ir,isn]
        end
    end

    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0
            @loop_sn_r_z isn ir iz begin
                dp_dt[iz,ir,isn] -=
                    charge_exchange*(
                        fvec.density[iz,ir,isn]*fvec.p_neutral[iz,ir,isn] -
                        fvec.density_neutral[iz,ir,isn]*fvec.p[iz,ir,isn] -
                        1.0/3.0 * fvec.density_neutral[iz,ir,isn]*fvec.density[iz,ir,isn] *
                            (fvec.uz_neutral[iz,ir,isn] - fvec.upar[iz,ir,isn])^2)
            end
        end
        if abs(ionization) > 0.0
            @loop_sn_r_z isn ir iz begin
                dp_dt[iz,ir,isn] -=
                    ionization*fvec.density[iz,ir,isn]*fvec.p_neutral[iz,ir,isn]
            end
        end
    end

    @loop_sn_r_z isn ir iz begin
        p_out[iz,ir,isn] += dt * dp_dt[iz,ir,isn]
    end

    return nothing
end

end
