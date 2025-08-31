"""
"""
module neutral_z_advection

export neutral_advection_z!
export update_speed_neutral_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
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
                                       moments.evolve_p, vz, vr, vzeta, z, r, t)
        # update adv_fac
        @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
            @views @. advect[isn].adv_fac[:,ivz,ivr,ivzeta,ir] = -dt*advect[isn].speed[:,ivz,ivr,ivzeta,ir]
        end
    end
    #calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vzvrvzetazrsn_1, fvec_in.pdf_neutral, advect,
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
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_neutral_z!(advect, uz, vth, evolve_upar, evolve_p, vz, vr, vzeta,
                                 z, r, t)
    @debug_consistency_checks r.n == size(advect.speed,5) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect.speed,4) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect.speed,3) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect.speed,2) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect.speed,1) || throw(BoundsError(speed))

    @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
        @. advect.speed[:,ivz,ivr,ivzeta,ir] = vz.grid[ivz]
    end
    if evolve_p
        @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
            @views @. advect.speed[:,ivz,ivr,ivzeta,ir] *= vth[:,ir]
        end
    end
    if evolve_upar
        @loop_r_vzeta_vr_vz ir ivzeta ivr ivz begin
            @views @. advect.speed[:,ivz,ivr,ivzeta,ir] += uz[:,ir]
        end
    end

    return nothing
end

end
