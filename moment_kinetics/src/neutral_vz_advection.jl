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
                         vz_spectral, composition, collisions,
                         neutral_source_settings) = begin
    return neutral_advection_vz!(f_out, fvec_in, fields, moments, advect, vz, vr, vzeta,
                                 z, r, dt, vz_spectral, composition, collisions,
                                 neutral_source_settings, Val(moments.evolve_density),
                                 Val(moments.evolve_upar), Val(moments.evolve_p))
end
function neutral_advection_vz!(f_out, fvec_in, fields, moments, advect, vz, vr, vzeta, z,
                               r, dt, vz_spectral, composition, collisions,
                               neutral_source_settings, evolve_density::Val,
                               evolve_upar::Val, evolve_p::Val)

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.
    if evolve_upar === Val(false) && evolve_p === Val(false)
        return nothing
    end

    @begin_sn_r_z_vzeta_vr_region()

    speed_args = get_speed_vz_inner_args(advect, fvec_in, moments, vz, evolve_density,
                                         evolve_upar, evolve_p)
    f_in = fvec_in.pdf_neutral
    @loop_sn_r isn ir begin
        speed_args_snr = get_speed_vz_inner_views_snr(isn, ir, speed_args...)
        this_f_out = @view f_out[:,:,:,:,ir,isn]
        this_f_in = @view f_in[:,:,:,:,ir,isn]
        @loop_z iz begin
            speed_args_z = get_speed_vz_inner_views_z(iz, speed_args_snr...)
            this_iz_f_out = @view this_f_out[:,:,:,iz]
            this_iz_f_in = @view this_f_in[:,:,:,iz]
            @loop_vzeta_vr ivzeta ivr begin
                speed_args_vzetavr = get_speed_vz_inner_views_vzetavr(ivzeta, ivr, speed_args_z...)
                # calculate the advection speed corresponding to current f
                update_speed_vz_inner!(speed_args_vzetavr...)
                @views advance_f_local!(this_iz_f_out[:,ivr,ivzeta],
                                        this_iz_f_in[:,ivr,ivzeta],
                                        first(speed_args_vzetavr), vz, dt, vz_spectral)
            end
        end
    end
end

"""
calculate the advection speed in the vz-direction at each grid point
"""
function update_speed_neutral_vz!(advect, fields, fvec, moments, vz, vr, vzeta, z, r,
                                  composition, collisions, neutral_source_settings)
    return update_speed_neutral_vz!(advect, fields, fvec, moments, vz, vr, vzeta, z, r,
                                    composition, collisions, neutral_source_settings,
                                    Val(moments.evolve_density), Val(moments.evolve_upar),
                                    Val(moments.evolve_p))
end
function update_speed_neutral_vz!(advect, fields, fvec, moments, vz, vr, vzeta, z, r,
                                  composition, collisions, neutral_source_settings,
                                  evolve_density::Val, evolve_upar::Val, evolve_p::Val)
    @debug_consistency_checks r.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks composition.n_neutral_species == size(advect,6) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,1) || throw(BoundsError(advect))

    if !(evolve_p === Val(true) || evolve_upar === Val(true))
        # No vz advection, so nothing to do.
        return nothing
    end

    speed_args = get_speed_vz_inner_args(advect, fvec, moments, vz, evolve_density,
                                         evolve_upar, evolve_p)
    @loop_sn_r isn ir begin
        speed_args_snr = get_speed_vz_inner_views_snr(isn, ir, speed_args...)
        @loop_z iz begin
            speed_args_z = get_speed_vz_inner_views_z(iz, speed_args_snr...)
            @loop_vzeta_vr ivzeta ivr begin
                update_speed_vz_inner!(get_speed_vz_inner_views_vzetavr(ivzeta, ivr, speed_args_z...)...)
            end
        end
    end

    return nothing
end

@inline function get_speed_vz_inner_args(advect, fvec, moments, vz, evolve_density::Val,
                                         evolve_upar::Val, evolve_p::Val)
    if evolve_p === Val(true) && evolve_upar === Val(true)
        return advect, fvec.uz_neutral, moments.neutral.duz_dt, moments.neutral.duz_dz,
               moments.neutral.vth, moments.neutral.dvth_dt, moments.neutral.dvth_dz,
               vz.grid, evolve_density, evolve_upar, evolve_p
    elseif evolve_p === Val(true)
        return advect, moments.neutral.vth, moments.neutral.dvth_dt,
               moments.neutral.dvth_dz, vz.grid, evolve_density, evolve_upar, evolve_p
    elseif evolve_upar === Val(true)
        return advect, fvec.uz_neutral, moments.neutral.duz_dt, moments.neutral.duz_dz,
               vz.grid, evolve_density, evolve_upar, evolve_p
    elseif evolve_density === Val(true)
        error("Evolving only density unsupported in this function at the moment.")
    else
        error("No evolving moments unsupported in this function at the moment.")
    end
end

@inline function get_speed_vz_inner_views_snr(isn, ir, advect, uz, duz_dt, duz_dz, vth,
                                              dvth_dt, dvth_dz, wz,
                                              evolve_density::Val{true},
                                              evolve_upar::Val{true}, evolve_p::Val{true})
    return @views advect[:,:,:,:,ir,isn], uz[:,ir,isn], duz_dt[:,ir,isn],
                  duz_dz[:,ir,isn], vth[:,ir,isn], dvth_dt[:,ir,isn], dvth_dz[:,ir,isn],
                  wz, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vz_inner_views_z(iz, advect, uz, duz_dt, duz_dz, vth, dvth_dt,
                                            dvth_dz, wz, evolve_density::Val{true},
                                            evolve_upar::Val{true}, evolve_p::Val{true})
    return @views advect[:,:,:,iz], uz[iz], duz_dt[iz],
                  duz_dz[iz], vth[iz], dvth_dt[iz], dvth_dz[iz],
                  wz, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vz_inner_views_vzetavr(ivzeta, ivr, advect, uz, duz_dt, duz_dz,
                                                  vth, dvth_dt, dvth_dz, wz,
                                                  evolve_density::Val{true},
                                                  evolve_upar::Val{true},
                                                  evolve_p::Val{true})
    return @views advect[:,ivr,ivzeta], uz, duz_dt, duz_dz, vth, dvth_dt, dvth_dz, wz,
                  evolve_density, evolve_upar, evolve_p
end

"""
update the advection speed (for the neutral distribution function) in the z-velocity
coordinate for the case where density, flow and pressure are evolved independently from
the pdf; in this case, the parallel velocity coordinate is the normalized peculiar
velocity wpahat = (vpa - upar)/vth
"""
function update_speed_vz_inner!(advect, uz, duz_dt, duz_dz, vth, dvth_dt, dvth_dz, wz,
                                evolve_density::Val{true}, evolve_upar::Val{true},
                                evolve_p::Val{true})
    @. advect =
           (
            - (duz_dt + (vth * wz + uz) * duz_dz)
            - wz * (dvth_dt + (vth * wz + uz) * dvth_dz)
           ) / vth

    return nothing
end

@inline function get_speed_vz_inner_views_snr(isn, ir, advect, vth, dvth_dt, dvth_dz, wz,
                                              evolve_density::Val{true},
                                              evolve_upar::Val{false},
                                              evolve_p::Val{true})
    return @views advect[:,:,:,:,ir,isn], vth[:,ir,isn], dvth_dt[:,ir,isn],
                  dvth_dz[:,ir,isn], wz, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vz_inner_views_z(iz, advect, vth, dvth_dt, dvth_dz, wz,
                                            evolve_density::Val{true},
                                            evolve_upar::Val{false}, evolve_p::Val{true})
    return @views advect[:,:,:,iz], vth[iz], dvth_dt[iz], dvth_dz[iz], wz, evolve_density,
                  evolve_upar, evolve_p
end

@inline function get_speed_vz_inner_views_vzetavr(ivzeta, ivr, advect, vth, dvth_dt,
                                                  dvth_dz, wz, evolve_density::Val{true},
                                                  evolve_upar::Val{false},
                                                  evolve_p::Val{true})
    return @views advect[:,ivr,ivzeta], vth, dvth_dt, dvth_dz, wz, evolve_density,
                  evolve_upar, evolve_p
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized velocity
vpahat = vpa/vth
"""
function update_speed_vz_inner!(advect, vth, dvth_dt, dvth_dz, wz,
                                evolve_density::Val{true}, evolve_upar::Val{false},
                                evolve_p::Val{true})
    @. advect = - wz * (dvth_dt + vth * wz * dvth_dz) / vth

    return nothing
end

@inline function get_speed_vz_inner_views_snr(isn, ir, advect, uz, duz_dt, duz_dz, wz,
                                              evolve_density::Val{true},
                                              evolve_upar::Val{true}, evolve_p::Val{false})
    return @views advect[:,:,:,:,ir,isn], uz[:,ir,isn], duz_dt[:,ir,isn],
                  duz_dz[:,ir,isn], wz, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vz_inner_views_z(iz, advect, uz, duz_dt, duz_dz, wz,
                                            evolve_density::Val{true},
                                            evolve_upar::Val{true}, evolve_p::Val{false})
    return @views advect[:,:,:,iz], uz[iz], duz_dt[iz], duz_dz[iz], wz, evolve_density,
                  evolve_upar, evolve_p
end

@inline function get_speed_vz_inner_views_vzetavr(ivzeta, ivr, advect, uz, duz_dt, duz_dz,
                                                  wz, evolve_density::Val{true},
                                                  evolve_upar::Val{true},
                                                  evolve_p::Val{false})
    return @views advect[:,ivr,ivzeta], uz, duz_dt, duz_dz, wz, evolve_density,
                  evolve_upar, evolve_p
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_vz_inner!(advect, uz, duz_dt, duz_dz, wz, evolve_density::Val{true},
                                evolve_upar::Val{true}, evolve_p::Val{false})
    @. advect = - (duz_dt + (wz + uz) * duz_dz)

    return nothing
end

end
