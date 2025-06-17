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
        # update adv_fac
        this_adv_fac = advect[is].adv_fac
        this_speed = advect[is].speed
        @loop_z_vperp_vpa iz ivperp ivpa begin
            @views @. this_adv_fac[:,ivpa,ivperp,iz] = -dt * this_speed[:,ivpa,ivperp,iz]
        end
    end
    # calculate the upwind derivative along r
    derivative_r!(scratch_dummy.buffer_vpavperpzrs_1, fvec_in.pdf, advect,
                  scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
                  scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
                  scratch_dummy.buffer_vpavperpzs_5,scratch_dummy.buffer_vpavperpzs_6,
                  r_spectral,r)

    # advance r-advection equation
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,iz,:,is],
          scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,iz,:,is], advect[is], ivpa,
          ivperp, iz, r, dt)
    end
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
        geofac = r.scratch
        @inbounds begin
            @loop_z iz begin
                @views @. geofac = bzeta[iz,:]*jacobian[iz,:]/Bmag[iz,:]
                @loop_vperp_vpa ivperp ivpa begin
                    # ExB drift
                    @views @. advect.speed[:,ivpa,ivperp,iz] = rhostar*geofac*fields.gEz[ivperp,iz,:,is]
                    # magnetic curvature drift
                    @. @views advect.speed[:,ivpa,ivperp,iz] += rhostar*(vpa.grid[ivpa]^2)*cvdriftr[iz,:]
                    # magnetic grad B drift
                    @. @views advect.speed[:,ivpa,ivperp,iz] += 0.5*rhostar*(vperp.grid[ivperp]^2)*gbdriftr[iz,:]
                end
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
