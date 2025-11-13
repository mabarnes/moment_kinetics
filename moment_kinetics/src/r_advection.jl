"""
"""
module r_advection

export r_advection!
export r_advection_1D_ITG!
export update_speed_r!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..derivatives: derivative_r!

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer r_advection!(
                         f_out, fvec_in, moments, fields, advect, r, z, vperp, vpa, dt,
                         r_spectral, composition, geometry, scratch_dummy) = begin

    @begin_s_z_vperp_vpa_region()

    @loop_s is begin
        # get the updated speed along the r direction using the current f
        @views update_speed_r!(advect[is], fields, moments.evolve_density,
                               moments.evolve_upar, moments.evolve_p, vpa, vperp, z, r,
                               geometry, is)
        # update adv_fac
        this_adv_fac = advect[is].adv_fac
        this_speed = advect[is].speed
        @loop_z_vperp_vpa iz ivperp ivpa begin
            @views @. this_adv_fac[:,ivpa,ivperp,iz] = -dt * this_speed[:,ivpa,ivperp,iz]
        end
    end
    # calculate the upwind derivative along r
    df_dr = scratch_dummy.buffer_vpavperpzrs_1
    derivative_r!(df_dr, fvec_in.pdf, advect, scratch_dummy.buffer_vpavperpzs_1,
                  scratch_dummy.buffer_vpavperpzs_2,
                  scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
                  scratch_dummy.buffer_vpavperpzs_5,scratch_dummy.buffer_vpavperpzs_6,
                  r_spectral,r)

    # advance r-advection equation
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,iz,:,is],
          df_dr[ivpa,ivperp,iz,:,is], advect[is], ivpa, ivperp, iz, r, dt)
    end
end

"""
1D false r advection, intended to provoke ITG in 1D. Despite there being no real r 
advection, we insist that a radial temperature gradient exists, and add the ExB drift
term to the DKE anyway to bring in temperature gradient drive. Since f is not known 
along r, and hence its temperature dependence is not known, it is assumed that f 
is roughly a Maxwellian with an R dependent temperature only. This means we can control
L_T through this term.
"""
function r_advection_1D_ITG!(
                         f_out, fvec_in, moments, fields, advect, r, z, vperp, vpa, dt,
                         r_spectral, composition, geometry, scratch_dummy)

    @begin_s_z_vperp_vpa_region()
    # only needs to be one of the terms because it's constant in z and r
    B_factor = geometry.Bzeta[1,1] / geometry.Bmag[1,1]^2
    L_T = composition.ion[1].L_T
    #println("B_factor = $B_factor")
    #println("L_T = $L_T")

    f = fvec_in.pdf
    Ez = fields.gEz
    p = moments.ion.p
    n = moments.ion.dens

    # Expression is B_zeta/B^2 * Ez * f/T * dT/dr * (m(vperp^2 + vpa^2)/2T - 3/2)
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        f_out[ivpa,ivperp,iz,1,is] += B_factor * Ez[ivperp,iz,1,is] * f[ivpa,ivperp,iz,1,is]/L_T *
                                        (1/2 * (vperp.grid[ivperp]^2 + vpa.grid[ivpa]^2)/
                                                (p[iz,1,is]/n[iz,1,is]) - 3/2) * dt
    end
end

"""
calculate the advection speed in the r-direction at each grid point
"""
function update_speed_r!(advect, fields, evolve_density, evolve_upar, evolve_p, vpa,
                         vperp, z, r, geometry, is)
    @debug_consistency_checks z.n == size(advect.speed,4) || throw(BoundsError(advect))
    @debug_consistency_checks vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @debug_consistency_checks vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @debug_consistency_checks r.n == size(advect.speed,1) || throw(BoundsError(speed))

    if r.n > 1
        if evolve_density || evolve_upar || evolve_p
            # Magnetic drifts not supported here yet
            vEr = fields.vEr
            speed = advect.speed
            @loop_z_vperp_vpa iz ivperp ivpa begin
                # ExB drift
                @views @. speed[:,ivpa,ivperp,iz] = vEr[iz,:]
            end
        else
            Bmag = geometry.Bmag
            rhostar = geometry.rhostar
            bzeta = geometry.bzeta
            jacobian = geometry.jacobian
            geofac = r.scratch
            curvature_drift_r = geometry.curvature_drift_r
            grad_B_drift_r = geometry.grad_B_drift_r
            gEz = fields.gEz
            speed = advect.speed
            @loop_z_vperp_vpa iz ivperp ivpa begin
                # ExB drift
                @views @. geofac = bzeta[iz,:]*jacobian[iz,:]/Bmag[iz,:]
                @views @. speed[:,ivpa,ivperp,iz] = rhostar*geofac*gEz[ivperp,iz,:,is]
                # magnetic curvature drift
                @. @views speed[:,ivpa,ivperp,iz] += rhostar*(vpa.grid[ivpa]^2)*curvature_drift_r[iz,:]
                # magnetic grad B drift
                @. @views speed[:,ivpa,ivperp,iz] += 0.5*rhostar*(vperp.grid[ivperp]^2)*grad_B_drift_r[iz,:]
            end
        end
    else
        # no advection if no length in r
        @loop_z_vperp_vpa iz ivperp ivpa begin
            advect.speed[:,ivpa,ivperp,iz] .= 0.0
        end
    end
    return nothing
end

end
