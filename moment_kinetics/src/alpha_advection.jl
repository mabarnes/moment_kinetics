"""
"""
module alpha_advection

export alpha_advection!
export update_speed_alpha!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..derivatives: derivative_z!

"""
Do a single stage time advance (potentially as part of a multi-stage RK scheme).
"""
@timeit global_timer alpha_advection!(
                         f_out, fvec_in, moments, fields, advect, z, vpa, vperp, r, dt, t,
                         spectral, composition, geometry, scratch_dummy) = begin

    @begin_s_r_vperp_vpa_region()

    # Get the updated alpha-advection speed projected along the z direction using the
    # current f.
    @views update_speed_alpha!(advect, moments.evolve_upar, moments.evolve_p, fields, vpa,
                               vperp, z, r, geometry)

    #calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, fvec_in.pdf, advect,
                  scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
                  scratch_dummy.buffer_vpavperprs_3, scratch_dummy.buffer_vpavperprs_4,
                  scratch_dummy.buffer_vpavperprs_5, scratch_dummy.buffer_vpavperprs_6,
                  spectral, z)

    # advance z-advection equation
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @. @views z.scratch = scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,:,ir,is], z.scratch,
                                         advect[:,ivpa,ivperp,ir,is], z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_alpha!(advect, evolve_upar::Bool, evolve_p::Bool, fields, vpa,
                             vperp, z, r, geometry)
    return update_speed_alpha!(advect, Val(evolve_upar), Val(evolve_p), fields, vpa,
                               vperp, z, r, geometry)
end
function update_speed_alpha!(advect, evolve_upar::Val, evolve_p::Val, fields, vpa, vperp,
                             z, r, geometry)
    @debug_consistency_checks r.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vperp.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vpa.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,1) || throw(BoundsError(advect))

    speed_args = get_speed_alpha_inner_args(advect, fields, geometry, z, vpa, vperp,
                                            evolve_upar, evolve_p)
    @loop_s_r is ir begin
        speed_args_sr = get_speed_alpha_inner_views_sr(is, ir, speed_args...)
        @loop_vperp ivperp begin
            speed_args_vperp = get_speed_alpha_inner_views_vperp(ivperp, speed_args_sr...)
            @loop_vpa ivpa begin
                @views update_speed_alpha_inner!(get_speed_alpha_inner_views_vpa(ivpa, speed_args_vperp...)...)
            end
        end
    end

    return nothing
end

@inline function get_speed_alpha_inner_args(advect, fields, geometry, z, vpa, vperp,
                                            evolve_upar, evolve_p)
    return advect, fields.vEz, geometry.rhostar, z.scratch, geometry.bzeta,
           geometry.jacobian, geometry.Bmag, geometry.curvature_drift_z,
           geometry.grad_B_drift_z, fields.gEr, vpa.grid, vperp.grid, evolve_upar,
           evolve_p
end

@inline function get_speed_alpha_inner_views_sr(is, ir, advect, vEz, rhostar, geofac,
                                                bzeta, jacobian, Bmag, curvature_drift_z,
                                                grad_B_drift_z, gEr, vpa, vperp,
                                                evolve_upar, evolve_p)
    @views @. geofac = bzeta[:,ir] * jacobian[:,ir] / Bmag[:,ir]
    return @views advect[:,:,:,ir,is], vEz[:,ir], rhostar, geofac,
                  curvature_drift_z[:,ir], grad_B_drift_z[:,ir], gEr[:,:,ir,is], vpa,
                  vperp, evolve_upar, evolve_p
end

@inline function get_speed_alpha_inner_views_vperp(ivperp, advect, vEz, rhostar, geofac,
                                                   curvature_drift_z, grad_B_drift_z, gEr,
                                                   vpa, vperp, evolve_upar, evolve_p)
    return @views advect[:,:,ivperp], vEz, rhostar, geofac, curvature_drift_z,
                  grad_B_drift_z, gEr[ivperp,:], vpa, vperp[ivperp], evolve_upar, evolve_p
end

@inline function get_speed_alpha_inner_views_vpa(ivpa, advect, vEz, rhostar, geofac,
                                                 curvature_drift_z, grad_B_drift_z, gEr,
                                                 vpa, vperp, evolve_upar, evolve_p)
    return @views advect[:,ivpa], vEz, rhostar, geofac, curvature_drift_z, grad_B_drift_z,
                  gEr, vpa[ivpa], vperp, evolve_upar, evolve_p
end

function update_speed_alpha_inner!(advect, vEz, rhostar, geofac, curvature_drift_z,
                                   grad_B_drift_z, gEr, vpa, vperp, evolve_upar::Val,
                                   evolve_p::Val)
    if evolve_upar === Val(true) || evolve_p === Val(true)
        # Magnetic drifts not supported here yet
        # ExB drift
        @. advect = vEz
    else
        # ExB drift
        @. advect = -rhostar * geofac * gEr
        # magnetic curvature drift
        @. advect += rhostar * vpa^2 * curvature_drift_z
        # magnetic grad B drift
        @. advect += 0.5 * rhostar * vperp^2 * grad_B_drift_z
    end
    return nothing
end

end
