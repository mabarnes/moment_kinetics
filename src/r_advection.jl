"""
"""
module r_advection

export r_advection!
export update_speed_r!

using ..advection: advance_f_local!, update_boundary_indices!
using ..chebyshev: chebyshev_info
using ..looping

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function r_advection!(f_out, fvec_in, fields, advect, r, z, vperp, vpa, 
                      dt, r_spectral, composition, geometry)
    
    begin_s_z_vperp_vpa_region()
    
    @loop_s is begin
        # get the updated speed along the r direction using the current f
        @views update_speed_r!(advect[is], fields, vpa, vperp, z, r, geometry)
        # update the upwind/downwind boundary indices and upwind_increment
        @views update_boundary_indices!(advect[is], loop_ranges[].vpa, loop_ranges[].vperp, loop_ranges[].z)
        
        # advance r-advection equation
        @loop_z_vperp_vpa iz ivperp ivpa begin
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in r.scratch
            @. r.scratch = fvec_in.pdf[ivpa,ivperp,iz,:,is]

            @views advance_f_local!(f_out[ivpa,ivperp,iz,:,is], r.scratch,
                                    advect[is], ivpa, ivperp, iz,
                                    r, dt, r_spectral)
        end
    end
end


"""
calculate the advection speed in the r-direction at each grid point
"""
function update_speed_r!(advect, fields, vpa, vperp, z, r, geometry)
    @boundscheck z.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck r.n == size(advect.speed,1) || throw(BoundsError(speed))
    if r.advection.option == "default" && r.n > 1
        ExBfac = 0.5*geometry.rstar
        @inbounds begin
            @loop_z_vperp_vpa iz ivperp ivpa begin
                @views advect.speed[:,ivpa,ivperp,iz] .= ExBfac*fields.Ez[iz,:]
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
    # the default for modified_speed is simply speed.
    @inbounds begin
        @loop_z_vperp_vpa iz ivperp ivpa begin
            @views advect.modified_speed[:,ivpa,ivperp,iz] .= advect.speed[:,ivpa,ivperp,iz]
        end
    end
    return nothing
end

end
