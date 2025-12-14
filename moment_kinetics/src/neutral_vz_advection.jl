"""
"""
module neutral_vz_advection

export neutral_advection_vz!
export update_speed_neutral_vz!

using ..advection: advance_f_local!
using ..communication
using ..debugging
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

    @begin_sn_r_z_vzeta_vr_region()

    # calculate the advection speed corresponding to current f
    update_speed_neutral_vz!(advect, fields, fvec_in, moments, vz, vr, vzeta, z, r,
                             composition, collisions, neutral_source_settings)
    @loop_sn isn begin
        @loop_r_z_vzeta_vr ir iz ivzeta ivr begin
            @views advance_f_local!(f_out[:,ivr,ivzeta,iz,ir,isn],
                                    fvec_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn],
                                    advect[:,ivr,ivzeta,iz,ir,isn], vz, dt, vz_spectral)
        end
    end
end

"""
calculate the advection speed in the vz-direction at each grid point
"""
function update_speed_neutral_vz!(advect, fields, fvec, moments, vz, vr, vzeta, z, r,
                                  composition, collisions, neutral_source_settings)
    @debug_consistency_checks r.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks composition.n_neutral_species == size(advect,6) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,1) || throw(BoundsError(advect))

    # dvpa/dt = Ze/m ⋅ E_parallel
    if moments.evolve_p && moments.evolve_upar
        update_speed_n_u_p_evolution_neutral!(advect, fvec, moments, vz, z, r,
                                              composition, collisions,
                                              neutral_source_settings)
    elseif moments.evolve_p
        update_speed_n_p_evolution_neutral!(advect, fields, fvec, moments, vz, z, r,
                                            composition, collisions,
                                            neutral_source_settings)
    elseif moments.evolve_upar
        update_speed_n_u_evolution_neutral!(advect, fvec, moments, vz, z, r, composition,
                                            collisions, neutral_source_settings)
    end

    return nothing
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
    uz = fvec.uz_neutral
    vth = moments.neutral.vth
    duz_dz = moments.neutral.duz_dz
    dvth_dz = moments.neutral.dvth_dz
    duz_dt = moments.neutral.duz_dt
    dvth_dt = moments.neutral.dvth_dt
    wz = vz.grid
    @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
        @. advect[:,ivr,ivzeta,iz,ir,isn] =
            (
             - (duz_dt[iz,ir,isn] + (vth[iz,ir,isn] * wz + uz[iz,ir,isn]) * duz_dz[iz,ir,isn])
             - wz * (dvth_dt[iz,ir,isn] + (vth[iz,ir,isn] * wz + uz[iz,ir,isn]) * dvth_dz[iz,ir,isn])
            ) / vth[iz,ir,isn]
    end

    return nothing
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
    vth = moments.neutral.vth
    dvth_dz = moments.neutral.dvth_dz
    dvth_dt = moments.neutral.dvth_dt
    wz = vz.grid
    @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
        @. advect[:,ivr,ivzeta,iz,ir,isn] =
            (
             - wz * (dvth_dt[iz,ir,isn] + vth[iz,ir,isn] * wz * dvth_dz[iz,ir,isn])
            ) / vth[iz,ir,isn]
    end

    return nothing
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_n_u_evolution_neutral!(advect, fvec, moments, vz, z, r, composition,
                                             collisions, neutral_source_settings)
    uz = fvec.uz_neutral
    duz_dz = moments.neutral.duz_dz
    duz_dt = moments.neutral.duz_dt
    wz = vz.grid
    @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
        @. advect[:,ivr,ivzeta,iz,ir,isn] =
             - (duz_dt[iz,ir,isn] + (wz + uz[iz,ir,isn]) * duz_dz[iz,ir,isn])
    end

    return nothing
end

end
