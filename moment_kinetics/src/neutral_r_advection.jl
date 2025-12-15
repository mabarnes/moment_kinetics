"""
"""
module neutral_r_advection

export neutral_advection_r!
export update_speed_neutral_r!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..derivatives: derivative_r!

"""
do a single stage time advance in r (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer neutral_advection_r!(
                         f_out, fvec_in, advect, r, z, vzeta, vr, vz, dt, r_spectral,
                         composition, geometry, scratch_dummy) = begin
    
    @begin_sn_z_vzeta_vr_vz_region()
    
    # get the updated speed along the r direction using the current f
    @views update_speed_neutral_r!(advect, r, z, vzeta, vr, vz)
    # calculate the upwind derivative along r
    derivative_r!(scratch_dummy.buffer_vzvrvzetazrsn_1, fvec_in.pdf_neutral, advect,
					scratch_dummy.buffer_vzvrvzetazsn_1, scratch_dummy.buffer_vzvrvzetazsn_2,
					scratch_dummy.buffer_vzvrvzetazsn_3,scratch_dummy.buffer_vzvrvzetazsn_4,
					scratch_dummy.buffer_vzvrvzetazsn_5,scratch_dummy.buffer_vzvrvzetazsn_6,
					r_spectral,r)

    # advance r-advection equation
    @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
        @. @views r.scratch = scratch_dummy.buffer_vzvrvzetazrsn_1[ivz,ivr,ivzeta,iz,:,isn]
        @views advance_f_df_precomputed!(f_out[ivz,ivr,ivzeta,iz,:,isn], r.scratch,
                                         advect[:,ivz,ivr,ivzeta,iz,isn], r, dt)
    end
end

"""
calculate the advection speed in the r-direction at each grid point
"""
function update_speed_neutral_r!(advect, r, z, vzeta, vr, vz)
    @debug_consistency_checks z.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks r.n == size(advect,1) || throw(BoundsError(advect))
    if r.n > 1
        @inbounds begin
            @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
                @views advect[:,ivz,ivr,ivzeta,iz,isn] .= vr.grid[ivr]
            end
        end
    else
        # no advection if no length in r 
        @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
            advect[:,ivz,ivr,ivzeta,iz,isn] .= 0.0
        end
    end
    return nothing
end

end
