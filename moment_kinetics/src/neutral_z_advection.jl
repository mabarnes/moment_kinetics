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

    # get the updated speed along the z direction using the current f
    @views update_speed_neutral_z!(advect, fvec_in.uz_neutral, moments.neutral.vth,
                                   moments.evolve_upar, moments.evolve_p, vz, vr, vzeta,
                                   z, r, t)
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
                                         advect[:,ivz,ivr,ivzeta,ir,isn], z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_neutral_z!(advect, uz, vth, evolve_upar, evolve_p, vz, vr, vzeta,
                                 z, r, t)
    @debug_consistency_checks r.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,1) || throw(BoundsError(advect))

    @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
        @. advect[:,ivz,ivr,ivzeta,ir,isn] = vz.grid[ivz]
    end
    if evolve_p
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            @views @. advect[:,ivz,ivr,ivzeta,ir,isn] *= vth[:,ir,isn]
        end
    end
    if evolve_upar
        @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
            @views @. advect[:,ivz,ivr,ivzeta,ir,isn] += uz[:,ir,isn]
        end
    end

    return nothing
end

end
