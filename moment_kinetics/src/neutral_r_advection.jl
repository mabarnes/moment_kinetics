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
using ..type_definitions
using ..derivatives: derivative_r!

"""
do a single stage time advance in r (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer neutral_advection_r!(
                         f_out, fvec_in, moments, advect, r, z, vzeta, vr, vz, dt,
                         r_spectral, composition, geometry, scratch_dummy) = begin

    if moments.evolve_p === Val(true)
        error("neutral_advection_r!() does not support moment-kinetic simulations with "
              * "separate pressure yet.")
    end
    
    @begin_sn_z_vzeta_vr_vz_region()
    
    # get the updated speed along the r direction using the current f
    @views update_speed_neutral_r!(advect, r, z, vzeta, vr, vz, moments.evolve_p)

    # calculate the upwind derivative along r
    df_dr = scratch_dummy.buffer_vzvrvzetazrsn_1
    derivative_r!(df_dr, fvec_in.pdf_neutral, advect, scratch_dummy.buffer_vzvrvzetazsn_1,
                  scratch_dummy.buffer_vzvrvzetazsn_2,
                  scratch_dummy.buffer_vzvrvzetazsn_3,scratch_dummy.buffer_vzvrvzetazsn_4,
                  scratch_dummy.buffer_vzvrvzetazsn_5,scratch_dummy.buffer_vzvrvzetazsn_6,
                  r_spectral,r)

    # advance r-advection equation
    @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
        @views advance_f_df_precomputed!(f_out[ivz,ivr,ivzeta,iz,:,isn],
                                         df_dr[ivz,ivr,ivzeta,iz,:,isn],
                                         advect[:,ivz,ivr,ivzeta,iz,isn], r, dt)
    end
end

"""
calculate the advection speed in the r-direction at each grid point
"""
function update_speed_neutral_r!(advect, r, z, vzeta, vr, vz, evolve_p::Bool)
    return update_speed_neutral_r!(advect, r, z, vzeta, vr, vz, Val(evolve_p))
end
function update_speed_neutral_r!(advect, r, z, vzeta, vr, vz,
                                 evolve_p::Val)
    @debug_consistency_checks z.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks r.n == size(advect,1) || throw(BoundsError(advect))
    if r.n > 1
        vr_grid = vr.grid
        @loop_sn_z isn iz begin
            speed_args_snz = get_speed_neutral_r_inner_views_sz(isn, iz, advect, vr_grid,
                                                                evolve_p)
            @loop_vzeta ivzeta begin
                speed_args_vzeta = get_speed_neutral_r_inner_views_vzeta(ivzeta, speed_args_snz...)
                @loop_vr ivr begin
                    speed_args_vr = get_speed_neutral_r_inner_views_vr(ivr, speed_args_vzeta...)
                    @loop_vz ivz begin
                        update_speed_neutral_r_inner!(get_speed_neutral_r_inner_views_vz(ivz, speed_args_vr...)...)
                    end
                end
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

@inline function get_speed_neutral_r_inner_views_sz(isn, iz, advect, vr, evolve_p::Val)
    return @views advect[:,:,:,:,iz,isn], vr, evolve_p
end

@inline function get_speed_neutral_r_inner_views_vzeta(ivzeta, advect, vr, evolve_p::Val)
    return @views advect[:,:,:,ivzeta], vr, evolve_p
end

@inline function get_speed_neutral_r_inner_views_vr(ivr, advect, vr, evolve_p::Val)
    return @views advect[:,:,ivr], vr[ivr], evolve_p
end

@inline function get_speed_neutral_r_inner_views_vz(ivz, advect, vr, evolve_p::Val)
    return @views advect[:,ivz], vr, evolve_p
end

function update_speed_neutral_r_inner!(advect, vr, evolve_p::Val)
    if evolve_p === Val(true)
        error("update_speed_neutral_r_inner!() does not support moment-kinetic "
              * "simulations with separate pressure yet.")
    else
        @. advect = vr
    end
end

end
