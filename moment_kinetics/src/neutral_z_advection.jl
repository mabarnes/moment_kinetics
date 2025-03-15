"""
"""
module neutral_z_advection

export neutral_advection_z!
export update_speed_neutral_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..looping
using ..timer_utils
using ..derivatives: derivative_z!

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer neutral_advection_z!(
                         f_out, fvec_in, moments, advect, r, z, vzeta, vr, vz, dt, t,
                         spectral, composition, scratch_dummy) = begin

    @begin_sn_r_vzeta_vr_vz_region()

    @loop_sn isn begin
        # get the updated speed along the z direction using the current f
        @views update_speed_neutral_z!(advect[isn], fvec_in.uz_neutral[:,:,isn],
                                       moments.neutral.vth[:,:,isn], moments.evolve_upar,
                                       moments.evolve_ppar, vz, vr, vzeta, z, r, t)
        # update adv_fac
        @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in z.scratch
            @views unnormalize_pdf!(
                scratch_dummy.buffer_vzvrvzetazrsn_2[ivz,ivr,ivzeta,:,ir,isn],
                fvec_in.pdf_neutral[ivz,ivr,ivzeta,:,ir,isn],
                fvec_in.density_neutral[:,ir,isn], moments.neutral.vth[:,ir,isn],
                moments.evolve_density, moments.evolve_ppar)
            @views adjust_advection_speed!(advect[isn].speed[:,ivz,ivr,ivzeta,ir],
                                           fvec_in.density_neutral[:,ir,isn],
                                           moments.neutral.vth[:,ir,isn],
                                           moments.evolve_density, moments.evolve_ppar)
            @views @. advect[isn].adv_fac[:,ivz,ivr,ivzeta,ir] = -dt*advect[isn].speed[:,ivz,ivr,ivzeta,ir]
        end
    end
    #calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vzvrvzetazrsn_1,
                  scratch_dummy.buffer_vzvrvzetazrsn_2, advect,
                  scratch_dummy.buffer_vzvrvzetarsn_1, scratch_dummy.buffer_vzvrvzetarsn_2,
                  scratch_dummy.buffer_vzvrvzetarsn_3, scratch_dummy.buffer_vzvrvzetarsn_4,
                  scratch_dummy.buffer_vzvrvzetarsn_5, scratch_dummy.buffer_vzvrvzetarsn_6,
                  spectral, z)

    # advance z-advection equation
    @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
        @. @views z.scratch = scratch_dummy.buffer_vzvrvzetazrsn_1[ivz,ivr,ivzeta,:,ir,isn]
        @views advance_f_df_precomputed!(f_out[ivz,ivr,ivzeta,:,ir,isn], z.scratch,
                                         advect[isn], ivz, ivr, ivzeta, ir, z, dt)
    end
end

"""
"""
function adjust_advection_speed!(speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        for i in eachindex(speed)
            speed[i] *= vth[i]/dens[i]
        end
    elseif evolve_density
        for i in eachindex(speed)
            speed[i] /= dens[i]
        end
    end
    return nothing
end

"""
"""
function unnormalize_pdf!(unnorm, norm, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. unnorm = norm * dens/vth
    elseif evolve_density
        @. unnorm = norm * dens
    else
        @. unnorm = norm
    end
    return nothing
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_neutral_z!(advect, uz, vth, evolve_upar, evolve_ppar, vz, vr, vzeta,
                                 z, r, t)
    @boundscheck r.n == size(advect.speed,5) || throw(BoundsError(advect))
    @boundscheck vzeta.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vr.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vz.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        @inbounds begin
            @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
                @. advect.speed[:,ivz,ivr,ivzeta,ir] = vz.grid[ivz]
            end
            if evolve_ppar
                @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
                    @views @. advect.speed[:,ivz,ivr,ivzeta,ir] *= vth[:,ir]
                end
            end
            if evolve_upar
                @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
                    @views @. advect.speed[:,ivz,ivr,ivzeta,ir] += uz[:,ir]
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
                @. advect.speed[:,ivz,ivr,ivzeta,ir] = z.advection.constant_speed
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
                @views @. advect.speed[:,ivz,ivr,ivzeta,ir] = z.advection.constant_speed*(z.grid+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
                @views @. advect.speed[:,ivz,ivr,ivzeta,ir] = z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    return nothing
end

end
