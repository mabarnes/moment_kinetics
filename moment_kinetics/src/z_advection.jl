"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..looping
using ..timer_utils
using ..derivatives: derivative_z!

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer z_advection!(
                         f_out, fvec_in, moments, fields, advect, z, vpa, vperp, r, dt, t,
                         spectral, composition, geometry, scratch_dummy) = begin

    @begin_s_r_vperp_vpa_region()

    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fvec_in.upar[:,:,is],
                               moments.ion.vth[:,:,is], moments.evolve_upar,
                               moments.evolve_ppar, fields, vpa, vperp, z, r, t, geometry, is)
        # update adv_fac
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @views adjust_advection_speed!(advect[is].speed[:,ivpa,ivperp,ir],
                                           fvec_in.density[:,ir,is],
                                           moments.ion.vth[:,ir,is],
                                           moments.evolve_density, moments.evolve_ppar)
            @views @. advect[is].adv_fac[:,ivpa,ivperp,ir] = -dt*advect[is].speed[:,ivpa,ivperp,ir]
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in z.scratch
            @views unnormalize_pdf!(
                scratch_dummy.buffer_vpavperpzrs_2[ivpa,ivperp,:,ir,is],
                fvec_in.pdf[ivpa,ivperp,:,ir,is], fvec_in.density[:,ir,is],
                moments.ion.vth[:,ir,is], moments.evolve_density, moments.evolve_ppar)
        end
    end
    #calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, scratch_dummy.buffer_vpavperpzrs_2,
                  advect, scratch_dummy.buffer_vpavperprs_1,
                  scratch_dummy.buffer_vpavperprs_2, scratch_dummy.buffer_vpavperprs_3,
                  scratch_dummy.buffer_vpavperprs_4, scratch_dummy.buffer_vpavperprs_5,
                  scratch_dummy.buffer_vpavperprs_6, spectral, z)

    # advance z-advection equation
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @. @views z.scratch = scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,:,ir,is], z.scratch,
                                         advect[is], ivpa, ivperp, ir, z, dt)
    end
end

"""
"""
function adjust_advection_speed!(speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        for i in eachindex(speed)
            speed[i] *= vth[i]/dens[i]
        end
    elseif evolve_density
        for i in eachindex(speed)
            speed[i] /= dens[i]
        end
    end
    return nothing
end

"""
"""
function unnormalize_pdf!(unnorm, norm, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. unnorm = norm * dens/vth
    elseif evolve_density
        @. unnorm = norm * dens
    else
        @. unnorm = norm
    end
    return nothing
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, upar, vth, evolve_upar, evolve_ppar, fields, vpa, vperp,
                         z, r, t, geometry, is)
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        # bzed = B_z/B only used for z.advection.option == "default"
        bzed = geometry.bzed
        Bmag = geometry.Bmag
        bzeta = geometry.bzeta
        jacobian = geometry.jacobian
        rhostar = geometry.rhostar
        ExBfac = -0.5*rhostar
        geofac = z.scratch
        cvdriftz = geometry.cvdriftz
        gbdriftz = geometry.gbdriftz
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                # vpa bzed
                @. @views advect.speed[:,ivpa,ivperp,ir] = vpa.grid[ivpa]*bzed[:,ir]
                # ExB drift
                @. @views advect.speed[:,ivpa,ivperp,ir] += ExBfac*bzeta[:,ir]*jacobian[:,ir]/Bmag[:,ir]*fields.gEr[ivperp,:,ir,is]
                # magnetic curvature drift
                @. @views advect.speed[:,ivpa,ivperp,ir] += rhostar*(vpa.grid[ivpa]^2)*cvdriftz[:,ir]
                # magnetic grad B drift
                @. @views advect.speed[:,ivpa,ivperp,ir] += 0.5*rhostar*(vperp.grid[ivperp]^2)*gbdriftz[:,ir]
            end
            if evolve_ppar
                @loop_r_vperp_vpa ir ivperp ivpa begin
                    @. @views advect.speed[:,ivpa,ivperp,ir] *= vth[:,ir]
                end
            end
            if evolve_upar
                @loop_r_vperp_vpa ir ivperp ivpa begin
                    @. @views advect.speed[:,ivpa,ivperp,ir] += upar[:,ir]
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @views advect.speed[:,ivpa,ivperp,ir] .= z.advection.constant_speed
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @. @views advect.speed[:,ivpa,ivperp,ir] = z.advection.constant_speed*(z.grid[i]+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @. @views advect.speed[:,ivpa,ivperp,ir] = z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    return nothing
end

end
