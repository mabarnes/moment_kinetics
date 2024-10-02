"""
"""
module neutral_vz_advection

export neutral_advection_vz!
export update_speed_neutral_vz!

using ..advection: advance_f_local!
using ..communication
using ..looping
using ..timer_utils

"""
"""
@timeit global_timer neutral_advection_vz!(
                         f_out, fvec_in, fields, moments, advect, vz, vr, vzeta, z, r, dt,
                         vz_spectral, composition, collisions, neutral_source_settings) =
begin

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.
    if !moments.evolve_upar
        return nothing
    end

    begin_sn_r_z_vzeta_vr_region()

    # calculate the advection speed corresponding to current f
    update_speed_neutral_vz!(advect, fields, fvec_in, moments, vz, vr, vzeta, z, r,
                             composition, collisions, neutral_source_settings)
    @loop_sn isn begin
        @loop_r_z_vzeta_vr ir iz ivzeta ivr begin
            @views advance_f_local!(f_out[:,ivr,ivzeta,iz,ir,isn], fvec_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn],
                                    advect[isn], ivr, ivzeta, iz, ir, vz, dt, vz_spectral)
        end
    end
end

"""
calculate the advection speed in the vz-direction at each grid point
"""
function update_speed_neutral_vz!(advect, fields, fvec, moments, vz, vr, vzeta, z, r,
                                  composition, collisions, neutral_source_settings)
    @boundscheck r.n == size(advect[1].speed,5) || throw(BoundsError(advect[1].speed))
    @boundscheck z.n == size(advect[1].speed,4) || throw(BoundsError(advect[1].speed))
    @boundscheck vzeta.n == size(advect[1].speed,3) || throw(BoundsError(advect[1].speed))
    @boundscheck vr.n == size(advect[1].speed,2) || throw(BoundsError(advect[1].speed))
    @boundscheck composition.n_neutral_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vz.n == size(advect[1].speed,1) || throw(BoundsError(advect[1].speed))
    if vz.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default_neutral!(advect, fields, fvec, moments, vz, z, r,
                                      composition, collisions, neutral_source_settings)
    elseif vz.advection.option == "constant"
        begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant
            @loop_sn isn begin
                update_speed_constant_neutral!(advect[isn], vz, 1:vr.n, 1:vzeta.n, 1:z.n, 1:r.n)
            end
        end
    elseif vpa.advection.option == "linear"
        begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant ⋅ (vpa + L_vpa/2)
            @loop_sn isn begin
                update_speed_linear_neutral!(advect[isn], vz, 1:vr.n, 1:vzeta.n, 1:z.n, 1:r.n)
            end
        end
    end
    return nothing
end

"""
"""
function update_speed_default_neutral!(advect, fields, fvec, moments, vz, z, r,
                                       composition, collisions, neutral_source_settings)
    if moments.evolve_ppar && moments.evolve_upar
        update_speed_n_u_p_evolution_neutral!(advect, fvec, moments, vz, z, r,
                                              composition, collisions,
                                              neutral_source_settings)
    elseif moments.evolve_ppar
        update_speed_n_p_evolution_neutral!(advect, fields, fvec, moments, vz, z, r,
                                            composition, collisions,
                                            neutral_source_settings)
    elseif moments.evolve_upar
        update_speed_n_u_evolution_neutral!(advect, fvec, moments, vz, z, r, composition,
                                            collisions, neutral_source_settings)
    end
end

"""
update the advection speed (for the neutral distribution function) in the z-velocity
coordinate for the case where density, flow and pressure are evolved independently from
the pdf; in this case, the parallel velocity coordinate is the normalized peculiar
velocity wpahat = (vpa - upar)/vth
"""
function update_speed_n_u_p_evolution_neutral!(advect, fvec, moments, vz, z, r,
                                               composition, collisions,
                                               neutral_source_settings)
    @loop_sn isn begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • (wpar/2*ppar)*dqpar/dz
            # • -wpar^2 * d(vth)/dz term
            @loop_z_vzeta_vr iz ivzeta ivr begin
                @views @. advect[isn].speed[:,ivr,ivzeta,iz,ir] =
                    moments.neutral.dpz_dz[iz,ir,isn]/(fvec.density_neutral[iz,ir,isn]*moments.neutral.vth[iz,ir,isn]) +
                    0.5*vz.grid*moments.neutral.dqz_dz[iz,ir,isn]/fvec.pz_neutral[iz,ir,isn] -
                    vz.grid^2*moments.neutral.dvth_dz[iz,ir,isn]
            end
        end
        # add in contributions from charge exchange and ionization collisions
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0 || abs(ionization) > 0.0
            @loop_r_z_vzeta_vr ir iz ivzeta ivr begin
                @views @. advect[isn].speed[:,ivr,ivzeta,iz,ir] +=
                    charge_exchange *
                    (0.5*vz.grid/fvec.pz_neutral[iz,ir,isn]
                     * (fvec.density[iz,ir,isn]*fvec.pz_neutral[iz,ir,isn]
                        - fvec.density_neutral[iz,ir,isn]*fvec.ppar[iz,ir,isn]
                        - fvec.density_neutral[iz,ir,isn]*fvec.density[iz,ir,isn]
                          * (fvec.uz_neutral[iz,ir,isn]-fvec.upar[iz,ir,isn])^2)
                     - fvec.density[iz,ir,isn]
                       * (fvec.upar[iz,ir,isn]-fvec.uz_neutral[iz,ir,isn])
                       / moments.neutral.vth[iz,ir,isn])
            end
        end
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_density_amplitude = moments.neutral.external_source_density_amplitude[:, :, index]
            @views source_momentum_amplitude = moments.neutral.external_source_momentum_amplitude[:, :, index]
            @views source_pressure_amplitude = moments.neutral.external_source_pressure_amplitude[:, :, index]
            density = fvec.density_neutral
            uz = fvec.uz_neutral
            pz = fvec.pz_neutral
            vth = moments.neutral.vth
            vz_grid = vz.grid
            @loop_s_r_z is ir iz begin
                term1 = source_density_amplitude[iz,ir] * uz[iz,ir,is]/(density[iz,ir,is]*vth[iz,ir,is])
                term2_over_vpa =
                    -0.5 * (source_pressure_amplitude[iz,ir] +
                            2.0 * uz[iz,ir,is] * source_momentum_amplitude[iz,ir]) /
                        pz[iz,ir,is] +
                    0.5 * source_density_amplitude[iz,ir] / density[iz,ir,is]
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    advect[is].speed[ivz,ivr,ivzeta,iz,ir] += term1 +
                                                            vz_grid[ivz] * term2_over_vpa
                end
            end
        end
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized velocity
vpahat = vpa/vth
"""
function update_speed_n_p_evolution_neutral!(advect, fields, fvec, moments, vz, z, r,
                                             composition, collisions,
                                             neutral_source_settings)
    @loop_sn isn begin
        # include contributions common to both ion and neutral species
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • (vpahat/2*ppar)*dqpar/dz
            # • vpahat*(upar/vth-vpahat) * d(vth)/dz term
            # • vpahat*d(upar)/dz
            @loop_z_vzeta_vr iz ivzeta ivr begin
                @views @. advect[isn].speed[:,ivr,ivzeta,iz,ir] =
                    0.5*vz.grid*moments.neutral.dqz_dz[iz,ir,isn]/fvec.pz_neutral[iz,ir,isn] +
                    vz.grid*moments.neutral.dvth_dz[iz] * (fvec.uz_neutral[iz,ir,isn]/moments.neutral.vth[iz,ir,isn] - vz.grid) +
                    vz.grid*moments.neutral.duz_dz[iz,ir,isn]
            end
        end
        charge_exchange = collisions.reactions.charge_exchange_frequency
        if abs(charge_exchange) > 0.0
            # add in contributions from charge exchange and ionization collisions
            error("suspect the charge exchange and ionization contributions here may be "
                  * "wrong because (upar[is]-upar[isp])^2 type terms were missed in the "
                  * "energy equation when it was substituted in to derive them.")
            @loop_r_z_vzeta_vr ir iz ivzeta ivr begin
                @views @. advect[is].speed[:,ivr,ivzeta,iz,ir] += charge_exchange *
                        0.5*vz.grid*fvec.density_neutral[iz,ir,is] * (1.0-fvec.ppar[iz,ir,is]/fvec.pz_neutral[iz,ir,is])
            end
        end
    end
    if any(x -> x.active, neutral_source_settings)
        error("External source not implemented for evolving n and ppar case")
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_n_u_evolution_neutral!(advect, fvec, moments, vz, z, r, composition,
                                             collisions, neutral_source_settings)
    @loop_sn isn begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • -wpar*dupar/dz
            @loop_z_vzeta_vr iz ivzeta ivr begin
                @views @. advect[isn].speed[:,ivr,ivzeta,iz,ir] =
                    moments.neutral.dpz_dz[iz,ir,isn]/fvec.density_neutral[iz,ir,isn] -
                    vz.grid*moments.neutral.duz_dz[iz,ir,isn]
            end
        end

        # if neutrals present compute contribution to parallel acceleration due to charge exchange
        # and/or ionization collisions betweens ions and neutrals

        charge_exchange = collisions.reactions.charge_exchange_frequency
        if abs(charge_exchange) > 0.0
            # include contribution to neutral acceleration due to collisional friction with ions
            @loop_r_z_vzeta_vr ir iz ivzeta ivr begin
                @views @. advect[isn].speed[:,ivr,ivzeta,iz,ir] -= charge_exchange*fvec.density[iz,ir,isn]*(fvec.upar[iz,ir,isn]-fvec.uz_neutral[iz,ir,isn])
            end
        end
    end
    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_density_amplitude = moments.neutral.external_source_density_amplitude[:, :, index]
            density = fvec.density_neutral
            uz = fvec.uz_neutral
            vth = moments.neutral.vth
            @loop_sn_r_z isn ir iz begin
                term = source_density_amplitude[iz,ir] * uz[iz,ir,isn] / density[iz,ir,isn]
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    advect[isn].speed[ivz,ivr,ivzeta,iz,ir] += term
                end
            end
        end
    end
end

"""
update the advection speed dvpa/dt = constant
"""
function update_speed_constant_neutral!(advect, vz, vr_range, vzeta_range, z_range, r_range)
    #@inbounds @fastmath begin
    for ir ∈ r_range
        for iz ∈ z_range
            for ivzeta ∈ vzeta_range, ivr ∈ vr_range
                @views advect.speed[:,ivr,ivzeta,iz,ir] .= vz.advection.constant_speed
            end
        end
    end
    #end
end

"""
update the advection speed dvpa/dt = const*(vpa + L/2)
"""
function update_speed_linear_neutral(advect, vz, vr_range, vzeta_range, z_range, r_range)
    @inbounds @fastmath begin
        for ir ∈ r_range
            for iz ∈ z_range
                for ivzeta ∈ vzeta_range, ivr ∈ vr_range
                    @views @. advect.speed[:,ivr,ivzeta,iz,ir] = vz.advection.constant_speed*(vz.grid+0.5*vpa.L)
                end
            end
        end
    end
end

end
