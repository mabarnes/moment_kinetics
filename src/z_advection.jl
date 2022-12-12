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
function z_advection!(f_out, fvec_in, fields, advect, z, vpa, vperp, r, dt, t, z_spectral, composition, geometry, scratch_dummy)
    
    begin_s_r_vperp_vpa_region()
    
    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fields, vpa, vperp, z, r, t, geometry)
        # update adv_fac
		advect[is].adv_fac[:,:,:,:] .= -dt.*advect[is].speed[:,:,:,:]
		# calculate the derivative along z
		# centered
	#	derivative_z!(scratch_dummy.buffer_vpavperpzr,fvec_in.pdf[:,:,:,:,is],
	#				scratch_dummy.buffer_vpavperpr_1, scratch_dummy.buffer_vpavperpr_2,
	#				scratch_dummy.buffer_vpavperpr_3,scratch_dummy.buffer_vpavperpr_4,
	#				z_spectral,z)
		#upwind
		derivative_z!(scratch_dummy.buffer_vpavperpzr,fvec_in.pdf[:,:,:,:,is], advect[is].adv_fac[:,:,:,:],
					scratch_dummy.buffer_vpavperpr_1, scratch_dummy.buffer_vpavperpr_2,
					scratch_dummy.buffer_vpavperpr_3,scratch_dummy.buffer_vpavperpr_4,
					scratch_dummy.buffer_vpavperpr_5,scratch_dummy.buffer_vpavperpr_6,
					z_spectral,z)
					
        # advance z-advection equation
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @. z.scratch = scratch_dummy.buffer_vpavperpzr[ivpa,ivperp,:,ir]
            @views advance_f_df_precomputed!(f_out[ivpa,ivperp,:,ir,is], 
			  z.scratch, advect[is], ivpa, ivperp, ir, z, dt, z_spectral)
        end
    end
end


"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, fields, vpa, vperp, z, r, t, geometry)
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        # bzed = B_z/B only used for z.advection.option == "default"
        bzed = geometry.bzed
        ExBfac = -0.5*geometry.rhostar
        @inbounds begin
            
            @loop_r_vperp_vpa ir ivperp ivpa begin
                    @views advect.speed[:,ivpa,ivperp,ir] .= vpa.grid[ivpa]*bzed .+ ExBfac*fields.Er[:,ir]
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
                @views advect.speed[:,ivpa,ivperp,ir] .= z.advection.constant_speed*(z.grid[i]+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            @loop_r_vperp_vpa ir ivperp ivpa begin
                @views advect.speed[:,ivpa,ivperp,ir] .= z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @views advect.modified_speed[:,ivpa,ivperp,ir] .= advect.speed[:,ivpa,ivperp,ir]
        end
    end
    return nothing
end

end
