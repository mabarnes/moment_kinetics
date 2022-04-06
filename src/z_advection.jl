"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..advection: advance_f_local!, update_boundary_indices!
using ..chebyshev: chebyshev_info
using ..calculus: derivative!
using ..looping

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function z_advection!(f_out, fvec_in, ff, fields, moments, SL, advect, z, vpa, vperp, r, 
                      use_semi_lagrange, dt, t, z_spectral, r_spectral, composition, geometry, scratch_dummy, istage)
    
    
    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fields, fvec_in.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, vperp, z, r, t, geometry, scratch_dummy, r_spectral)
        # update the upwind/downwind boundary indices and upwind_increment
        @views update_boundary_indices!(advect[is], loop_ranges[].vpa, loop_ranges[].vperp, loop_ranges[].r)

        # advance z-advection equation
        @loop_r_vperp_vpa ir ivperp ivpa begin
            @. z.scratch = fvec_in.pdf[ivpa,ivperp,:,ir,is]
            @views advance_f_local!(f_out[ivpa,ivperp,:,ir,is], z.scratch, ff[ivpa,ivperp,:,ir,is], SL, advect[is], ivpa, ivperp, ir,
                                    z, dt, istage, z_spectral, use_semi_lagrange)
        end
    end
end


"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, fields, upar, vth, evolve_upar, evolve_ppar, vpa, vperp, z, r, t, geometry, scratch_dummy, r_spectral)
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        # kpar only used for z.advection.option == "default"
        kpar = geometry.Bzed/geometry.Bmag
        ExBfac = 0.5*geometry.rstar
        @inbounds begin
            # calculate d phi / d r and store it in the dummy_zr array,
            # unless r dimension is size 1, in which case set to 0.
            if r.n > 1 
                @loop_z iz begin
                    derivative!(r.scratch, view(fields.phi,iz,:), r, r_spectral)
                    @loop_r ir begin
                    scratch_dummy.dummy_zr[iz,ir] = r.scratch[ir]
                    end
                end
            else
                @. scratch_dummy.dummy_zr[:,:] = 0.
            end

            @loop_r_vperp_vpa ir ivperp ivpa begin
                    @views advect.speed[:,ivpa,ivperp,ir] .= vpa.grid[ivpa]*kpar .+ ExBfac*scratch_dummy.dummy_zr[:,ir]
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
