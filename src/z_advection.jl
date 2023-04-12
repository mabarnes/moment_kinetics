"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..looping
using ..derivatives: derivative_z!

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function z_advection!(f_out, fvec_in, fields, advect, z, vpa, mu, r, dt, t, z_spectral, composition, geometry, scratch_dummy)
    
    begin_s_r_mu_vpa_region()
    
    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fields, vpa, mu, z, r, t, geometry)
        # update adv_fac
        @loop_r_mu_vpa ir imu ivpa begin
            @views @. advect[is].adv_fac[:,ivpa,imu,ir] = -dt*advect[is].speed[:,ivpa,imu,ir]
        end
    end
	#calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vpamuzrs_1, fvec_in.pdf, advect,
					scratch_dummy.buffer_vpamurs_1, scratch_dummy.buffer_vpamurs_2,
					scratch_dummy.buffer_vpamurs_3,scratch_dummy.buffer_vpamurs_4,
					scratch_dummy.buffer_vpamurs_5,scratch_dummy.buffer_vpamurs_6,
					z_spectral,z)
					
    # advance z-advection equation
    @loop_s_r_mu_vpa is ir imu ivpa begin
        @. @views z.scratch = scratch_dummy.buffer_vpamuzrs_1[ivpa,imu,:,ir,is]
        @views advance_f_df_precomputed!(f_out[ivpa,imu,:,ir,is],
          z.scratch, advect[is], ivpa, imu, ir, z, dt, z_spectral)
    end
end


"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, fields, vpa, mu, z, r, t, geometry)
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck mu.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        # bzed = B_z/B only used for z.advection.option == "default"
        bzed = geometry.bzed
        ExBfac = -0.5*geometry.rhostar
        @inbounds begin
            
            @loop_r_mu_vpa ir imu ivpa begin
                @. @views advect.speed[:,ivpa,imu,ir] = vpa.grid[ivpa]*bzed + ExBfac*fields.Er[:,ir]
            end
        
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            @loop_r_mu_vpa ir imu ivpa begin
                @views advect.speed[:,ivpa,imu,ir] .= z.advection.constant_speed
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            @loop_r_mu_vpa ir imu ivpa begin
                @. @views advect.speed[:,ivpa,imu,ir] = z.advection.constant_speed*(z.grid[i]+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            @loop_r_mu_vpa ir imu ivpa begin
                @. @views advect.speed[:,ivpa,imu,ir] = z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        @loop_r_mu_vpa ir imu ivpa begin
            @views advect.modified_speed[:,ivpa,imu,ir] .= advect.speed[:,ivpa,imu,ir]
        end
    end
    return nothing
end

end
