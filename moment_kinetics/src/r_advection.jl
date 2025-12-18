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

    # get the updated speed along the r direction using the current f
    update_speed_r!(advect, fields, moments.evolve_density, moments.evolve_upar,
                    moments.evolve_p, vpa, vperp, z, r, geometry)

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
                                         df_dr[ivpa,ivperp,iz,:,is],
                                         advect[:,ivpa,ivperp,iz,is], r, dt)
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
function update_speed_r!(advect, fields, evolve_density::Bool, evolve_upar::Bool,
                         evolve_p::Bool, vpa, vperp, z, r, geometry)
    return update_speed_r!(advect, fields, Val(evolve_density), Val(evolve_upar),
                           Val(evolve_p), vpa, vperp, z, r, geometry)
end
function update_speed_r!(advect, fields, evolve_density::Val, evolve_upar::Val,
                         evolve_p::Val, vpa, vperp, z, r, geometry)
    @debug_consistency_checks z.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vperp.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vpa.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks r.n == size(advect,1) || throw(BoundsError(advect))

    if r.n > 1
        speed_args = get_speed_r_inner_args(advect, fields, geometry, r, vpa, vperp,
                                            evolve_density, evolve_upar, evolve_p)
        @loop_s_z is iz begin
            speed_args_sz = get_speed_r_inner_views_sz(is, iz, speed_args...)
            @loop_vperp ivperp begin
                speed_args_vperp = get_speed_r_inner_views_vperp(ivperp, speed_args_sz...)
                @loop_vpa ivpa begin
                    @views update_speed_r_inner!(get_speed_r_inner_views_vpa(ivpa, speed_args_vperp...)...)
                end
            end
        end
    else
        # no advection if no length in r
        @loop_s_z_vperp_vpa is iz ivperp ivpa begin
            advect[:,ivpa,ivperp,iz,is] .= 0.0
        end
    end
    return nothing
end

@inline function get_speed_r_inner_args(advect, fields, geometry, r, vpa, vperp,
                                        evolve_density, evolve_upar, evolve_p)
    return advect, fields.vEr, geometry.rhostar, r.scratch, geometry.bzeta,
           geometry.jacobian, geometry.Bmag, geometry.curvature_drift_r,
           geometry.grad_B_drift_r, fields.gEz, vpa.grid, vperp.grid, evolve_density,
           evolve_upar, evolve_p
end

@inline function get_speed_r_inner_views_sz(is, iz, advect, vEr, rhostar, geofac, bzeta,
                                            jacobian, Bmag, curvature_drift_r,
                                            grad_B_drift_r, gEz, vpa, vperp,
                                            evolve_density, evolve_upar, evolve_p)
    @views @. geofac = bzeta[iz,:] * jacobian[iz,:] / Bmag[iz,:]
    return @views advect[:,:,:,iz,is], vEr[iz,:], rhostar, geofac,
                  curvature_drift_r[iz,:], grad_B_drift_r[iz,:], gEz[:,iz,:,is], vpa,
                  vperp, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_r_inner_views_vperp(ivperp, advect, vEr, rhostar, geofac,
                                               curvature_drift_r, grad_B_drift_r, gEz,
                                               vpa, vperp, evolve_density, evolve_upar,
                                               evolve_p)
    return @views advect[:,:,ivperp], vEr, rhostar, geofac, curvature_drift_r,
                  grad_B_drift_r, gEz[ivperp,:], vpa, vperp[ivperp], evolve_density,
                  evolve_upar, evolve_p
end

@inline function get_speed_r_inner_views_vpa(ivpa, advect, vEr, rhostar, geofac,
                                             curvature_drift_r, grad_B_drift_r, gEz, vpa,
                                             vperp, evolve_density, evolve_upar, evolve_p)
    return @views advect[:,ivpa], vEr, rhostar, geofac, curvature_drift_r, grad_B_drift_r,
                  gEz, vpa[ivpa], vperp, evolve_density, evolve_upar, evolve_p
end

function update_speed_r_inner!(advect, vEr, rhostar, geofac, curvature_drift_r,
                               grad_B_drift_r, gEz, vpa, vperp, evolve_density::Val,
                               evolve_upar::Val, evolve_p::Val)
    if evolve_density === Val(true) || evolve_upar === Val(true) || evolve_p === Val(true)
        # Magnetic drifts not supported here yet
        # ExB drift
        @. advect = vEr
    else
        # ExB drift
        @. advect = rhostar * geofac * gEz
        # magnetic curvature drift
        @. advect += rhostar * vpa^2 * curvature_drift_r
        # magnetic grad B drift
        @. advect += 0.5 * rhostar * vperp^2 * grad_B_drift_r
    end
    return nothing
end

end
