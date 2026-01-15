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
    
    # get the updated speed along the r direction using the current f
    @views update_speed_neutral_r!(advect, r, z, vzeta, vr, vz, moments.evolve_p)

    @begin_sn_z_vzeta_vr_vz_region()

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
                                         advect[ivz,ivr,ivzeta,iz,:,isn], r, dt)
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
    @debug_consistency_checks r.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,1) || throw(BoundsError(advect))

    @begin_sn_r_z_vzeta_vr_region()

    if r.n > 1
        vr_grid = vr.grid
        @loop_sn_r isn ir begin
            speed_args_snr = get_speed_neutral_r_inner_views_sr(isn, ir, advect, vr_grid,
                                                                evolve_p)
            @loop_z iz begin
                speed_args_z = get_speed_neutral_r_inner_views_z(iz, speed_args_snr...)
                @loop_vzeta_vr ivzeta ivr begin
                    update_speed_neutral_r_inner!(get_speed_neutral_r_inner_views_vzetavr(ivzeta, ivr, speed_args_z...)...)
                end
            end
        end
    else
        # no advection if no length in r 
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            advect[:,ivr,ivzeta,iz,ir,isn] .= 0.0
        end
    end
    return nothing
end

@inline function get_speed_neutral_r_inner_views_sr(isn, ir, advect, vr, evolve_p::Val)
    return @views advect[:,:,:,:,ir,isn], vr, evolve_p
end

@inline function get_speed_neutral_r_inner_views_z(iz, advect, vr, evolve_p::Val)
    return @views advect[:,:,:,iz], vr, evolve_p
end

@inline function get_speed_neutral_r_inner_views_vzetavr(ivzeta, ivr, advect, vr, evolve_p::Val)
    return @views advect[:,ivr,ivzeta], vr[ivr], evolve_p
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
