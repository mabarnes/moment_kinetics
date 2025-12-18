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
    df_dz = scratch_dummy.buffer_vzvrvzetazrsn_1
    derivative_z!(df_dz, fvec_in.pdf_neutral, advect, scratch_dummy.buffer_vzvrvzetarsn_1,
                  scratch_dummy.buffer_vzvrvzetarsn_2,
                  scratch_dummy.buffer_vzvrvzetarsn_3,
                  scratch_dummy.buffer_vzvrvzetarsn_4,
                  scratch_dummy.buffer_vzvrvzetarsn_5,
                  scratch_dummy.buffer_vzvrvzetarsn_6, spectral, z)

    # advance z-advection equation
    @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
        @views advance_f_df_precomputed!(f_out[ivz,ivr,ivzeta,:,ir,isn],
                                         df_dz[ivz,ivr,ivzeta,:,ir,isn],
                                         advect[:,ivz,ivr,ivzeta,ir,isn], z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_neutral_z!(advect, uz, vth, evolve_upar::Bool, evolve_p::Bool, vz,
                                 vr, vzeta, z, r, t)
    return update_speed_neutral_z!(advect, uz, vth, Val(evolve_upar), Val(evolve_p), vz,
                                   vr, vzeta, z, r, t)
end
function update_speed_neutral_z!(advect, uz, vth, evolve_upar::Val, evolve_p::Val, vz, vr,
                                 vzeta, z, r, t)
    @debug_consistency_checks r.n == size(advect,5) || throw(BoundsError(advect))
    @debug_consistency_checks vzeta.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vr.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vz.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,1) || throw(BoundsError(advect))

    vz_grid = vz.grid
    @loop_sn_r isn ir begin
        speed_args_snr = get_speed_neutral_z_inner_views_sr(isn, ir, advect, uz, vth,
                                                            vz_grid, evolve_upar,
                                                            evolve_p)
        @loop_vzeta_vr ivzeta ivr begin
            speed_args_vzetavr = get_speed_neutral_z_inner_views_vzetavr(ivzeta, ivr, speed_args_snr...)
            @loop_vz ivz begin
                update_speed_neutral_z_inner!(get_speed_neutral_z_inner_views_vz(ivz, speed_args_vzetavr...)...)
            end
        end
    end

    return nothing
end

@inline function get_speed_neutral_z_inner_views_sr(isn, ir, advect, uz, vth, vz,
                                                    evolve_upar::Val, evolve_p::Val)
    return @views advect[:,:,:,:,ir,isn], uz[:,ir,isn], vth[:,ir,isn], vz, evolve_upar,
                  evolve_p
end

@inline function get_speed_neutral_z_inner_views_vzetavr(ivzeta, ivr, advect, uz, vth, vz,
                                                         evolve_upar::Val, evolve_p::Val)
    return @views advect[:,:,ivr,ivzeta], uz, vth, vz, evolve_upar, evolve_p
end

@inline function get_speed_neutral_z_inner_views_vz(ivz, advect, uz, vth, vz,
                                                    evolve_upar::Val, evolve_p::Val)
    return @views advect[:,ivz], uz, vth, vz[ivz], evolve_upar, evolve_p
end

function update_speed_neutral_z_inner!(advect, uz, vth, vz, evolve_upar::Val,
                                       evolve_p::Val)
    if evolve_p === Val(true)
        @. advect = (vth * vz + uz)
    elseif evolve_upar === Val(true)
        @. advect = (vz + uz)
    else
        @. advect = vz
    end

    return nothing
end

end
