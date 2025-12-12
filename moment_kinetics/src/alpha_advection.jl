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

    @loop_s is begin
        # Get the updated alpha-advection speed projected along the z direction using the
        # current f.
        @views update_speed_alpha!(advect[is], moments.evolve_upar, moments.evolve_p,
                                   fields, vpa, vperp, z, r, geometry, is)
        # update adv_fac
        this_adv_fac = advect[is].adv_fac
        this_speed = advect[is].speed
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @views @. this_adv_fac[:,ivpa,ivperp,ir] = -dt*this_speed[:,ivpa,ivperp,ir]
        end
    end

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
                                         advect[is], ivpa, ivperp, ir, z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_alpha!(advect, evolve_upar, evolve_p, fields, vpa, vperp, z, r,
                             geometry, is)
    @debug_consistency_checks r.n == size(advect.speed,4) || throw(BoundsError(advect))
    @debug_consistency_checks vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @debug_consistency_checks vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect.speed,1) || throw(BoundsError(speed))

    Bmag = geometry.Bmag
    bzeta = geometry.bzeta
    jacobian = geometry.jacobian
    rhostar = geometry.rhostar
    curvature_drift_z = geometry.curvature_drift_z
    grad_B_drift_z = geometry.grad_B_drift_z
    if evolve_p || evolve_upar
        vEz = fields.vEz
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @. @views advect.speed[:,ivpa,ivperp,ir] = vEz[:,ir]
        end
    else
        @loop_r_vperp_vpa ir ivperp ivpa begin
            # ExB drift
            @. @views advect.speed[:,ivpa,ivperp,ir] = -rhostar*bzeta[:,ir]*jacobian[:,ir]/Bmag[:,ir]*fields.gEr[ivperp,:,ir,is]
            # magnetic curvature drift
            @. @views advect.speed[:,ivpa,ivperp,ir] += rhostar*(vpa.grid[ivpa]^2)*curvature_drift_z[:,ir]
            # magnetic grad B drift
            @. @views advect.speed[:,ivpa,ivperp,ir] += 0.5*rhostar*(vperp.grid[ivperp]^2)*grad_B_drift_z[:,ir]
        end
    end

    return nothing
end

end
