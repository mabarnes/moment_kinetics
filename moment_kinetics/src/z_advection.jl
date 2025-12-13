"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..derivatives: derivative_z!

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer z_advection!(
                         f_out, fvec_in, moments, fields, advect, z, vpa, vperp, r, dt, t,
                         spectral, composition, geometry, scratch_dummy) = begin

    # get the updated speed along the z direction using the current f
    update_speed_z!(advect, fvec_in.upar, moments.ion.vth, moments.evolve_upar,
                    moments.evolve_p, vpa, vperp, z, r, geometry)

    @begin_s_r_vperp_vpa_region()

    #calculate the upwind derivative
    df_dz = scratch_dummy.buffer_vpavperpzrs_1
    derivative_z!(df_dz, fvec_in.pdf, advect, scratch_dummy.buffer_vpavperprs_1,
                  scratch_dummy.buffer_vpavperprs_2, scratch_dummy.buffer_vpavperprs_3,
                  scratch_dummy.buffer_vpavperprs_4, scratch_dummy.buffer_vpavperprs_5,
                  scratch_dummy.buffer_vpavperprs_6, spectral, z)

    # advance z-advection equation
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,:,ir,is],
                                         df_dz[ivpa,ivperp,:,ir,is],
                                         advect[ivpa,ivperp,:,ir,is], z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, upar, vth, evolve_upar::Bool, evolve_p::Bool, vpa, vperp,
                         z, r, geometry)
    return update_speed_z!(advect, upar, vth, Val(evolve_upar), Val(evolve_p), vpa, vperp,
                           z, r, geometry)
end
function update_speed_z!(advect, upar, vth, evolve_upar::Val, evolve_p::Val, vpa, vperp,
                         z, r, geometry)
    @debug_consistency_checks r.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vperp.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks vpa.n == size(advect,1) || throw(BoundsError(advect))

    @begin_s_r_z_vperp_region()

    bzed = geometry.bzed
    vpa_grid = vpa.grid
    @loop_s_r is ir begin
        speed_args_sr = get_speed_z_inner_views_sr(is, ir, advect, upar, vth, vpa_grid,
                                                   bzed, evolve_upar, evolve_p)
        @loop_z iz begin
            speed_args_z = get_speed_z_inner_views_z(iz, speed_args_sr...)
            @loop_vperp ivperp begin
                update_speed_z_inner!(get_speed_z_inner_views_vperp(ivperp, speed_args_z...)...)
            end
        end
    end
    return nothing
end

@inline function get_speed_z_inner_views_sr(is, ir, advect, upar, vth, vpa, bzed,
                                            evolve_upar::Val, evolve_p::Val)
    return @views advect[:,:,:,ir,is], upar[:,ir,is], vth[:,ir,is], vpa, bzed[:,ir],
                  evolve_upar, evolve_p
end

@inline function get_speed_z_inner_views_z(iz, advect, upar, vth, vpa, bzed,
                                           evolve_upar::Val, evolve_p::Val)
    return @views advect[:,:,iz], upar[iz], vth[iz], vpa, bzed[iz], evolve_upar, evolve_p
end

@inline function get_speed_z_inner_views_vperp(ivperp, advect, upar, vth, vpa, bzed,
                                               evolve_upar::Val, evolve_p::Val)
    return @views advect[:,ivperp], upar, vth, vpa, bzed, evolve_upar, evolve_p
end

function update_speed_z_inner!(advect, upar, vth, vpa, bzed, evolve_upar::Val,
                               evolve_p::Val)
    if evolve_p === Val(true)
        @. advect = (vth * vpa + upar) * bzed
    elseif evolve_upar === Val(true)
        @. advect = (vpa + upar) * bzed
    else
        # vpa bzed
        @. advect = vpa * bzed
    end

    return nothing
end

function get_ion_z_advection_term_evolve_nup(sub_terms::IonSubTerms)
    vth = sub_terms.vth
    wpa = sub_terms.wpa
    upar = sub_terms.upar
    bzed = sub_terms.bzed
    df_dz = sub_terms.df_dz

    term = (vth * wpa + upar) * bzed * df_dz

    return term
end

end
