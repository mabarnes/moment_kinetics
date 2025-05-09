"""
"""
module r_advection

export r_advection!
export update_speed_r!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
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
        @views update_speed_r!(advect[is], fvec_in.upar[:,:,is],
                               moments.ion.vth[:,:,is], fields, moments.evolve_upar,
                               moments.evolve_p, vpa, vperp, z, r, geometry, is)
        # advance r-advection equation
        @loop_z_vpa iz ivpa begin
        end
        # update adv_fac
        @loop_z_vperp_vpa iz ivperp ivpa begin
            @views adjust_advection_speed!(advect[is].speed[:,ivpa,ivperp,iz],
                                           fvec_in.density[iz,:,is],
                                           moments.ion.vth[iz,:,is],
                                           moments.evolve_density, moments.evolve_p)
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in r.scratch
            @views unnormalize_pdf!(
                       scratch_dummy.buffer_vpavperpzrs_2[ivpa,ivperp,iz,:,is],
                       fvec_in.pdf[ivpa,ivperp,iz,:,is], fvec_in.density[iz,:,is],
                       moments.ion.vth[iz,:,is], moments.evolve_density,
                       moments.evolve_p)
            advect[is].adv_fac[:,ivpa,ivperp,iz] .= -dt.*advect[is].speed[:,ivpa,ivperp,iz]
        end
    end
    # calculate the upwind derivative along r
    derivative_r!(scratch_dummy.buffer_vpavperpzrs_1, scratch_dummy.buffer_vpavperpzrs_2,
        advect, scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
        scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
        scratch_dummy.buffer_vpavperpzs_5,scratch_dummy.buffer_vpavperpzs_6, r_spectral,r)

    # advance r-advection equation
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        @. r.scratch = scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,iz,:,is]
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,iz,:,is],
          r.scratch, advect[is], ivpa, ivperp, iz, r, dt)
    end
end

"""
"""
function adjust_advection_speed!(speed, dens, vth, evolve_density, evolve_p)
    if evolve_p
        @. speed *= vth/dens
    elseif evolve_density
        @. speed /= dens
    end
    return nothing
end

"""
"""
function unnormalize_pdf!(unnorm, norm, dens, vth, evolve_density, evolve_p)
    if evolve_p
        @. unnorm = norm * dens/vth
    elseif evolve_density
        @. unnorm = norm * dens
    else
        @. unnorm = norm
    end
    return nothing
end

"""
calculate the advection speed in the r-direction at each grid point
"""
function update_speed_r!(advect, upar, vth, fields, evolve_upar, evolve_p, vpa, vperp,
                         z, r, geometry, is)
    @boundscheck z.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck r.n == size(advect.speed,1) || throw(BoundsError(speed))
    if evolve_upar || evolve_p
        error("r_advection is not compatible with evolve_upar or evolve_p")
    end
    if r.advection.option == "default" && r.n > 1
        Bmag = geometry.Bmag
        rhostar = geometry.rhostar
        bzeta = geometry.bzeta
        jacobian = geometry.jacobian
        geofac = r.scratch
        cvdriftr = geometry.cvdriftr
        gbdriftr = geometry.gbdriftr
        @inbounds begin
            @loop_z_vperp_vpa iz ivperp ivpa begin
                # ExB drift
                @. geofac = bzeta[iz,:]*jacobian[iz,:]/Bmag[iz,:]
                @views @. advect.speed[:,ivpa,ivperp,iz] = rhostar*geofac*fields.gEz[ivperp,iz,:,is]
                # magnetic curvature drift
                @. @views advect.speed[:,ivpa,ivperp,iz] += rhostar*(vpa.grid[ivpa]^2)*cvdriftr[iz,:]
                # magnetic grad B drift
                @. @views advect.speed[:,ivpa,ivperp,iz] += 0.5*rhostar*(vperp.grid[ivperp]^2)*gbdriftr[iz,:]
            end
        end
    elseif r.advection.option == "default" && r.n == 1
        # no advection if no length in r
        @loop_z_vperp_vpa iz ivperp ivpa begin
            advect.speed[:,ivpa,ivperp,iz] .= 0.
        end
    elseif r.advection.option == "constant"
        @inbounds begin
            @loop_z_vperp_vpa iz ivperp ivpa begin
                @views advect.speed[:,ivpa,ivperp,iz] .= r.advection.constant_speed
            end
        end
    end
    return nothing
end

end
