module vperp_advection

export vperp_advection!
export update_speed_vperp!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..looping

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
function vperp_advection!(f_out, fvec_in, advect, r, z, vperp, vpa,
                      dt, spectral, composition)
    @loop_s is begin
        # get the updated speed along the r direction using the current f
        @views update_speed_vperp!(advect[is], vpa, vperp, z, r)
        @loop_r_z_vpa ir iz ivpa begin
            @views advance_f_local!(f_out[ivpa,:,iz,ir,is], vperp.scratch, advect[is], ivpa,
                                    r, dt, spectral)
        end
    end
end

# calculate the advection speed in the vperp-direction at each grid point
function update_speed_vperp!(advect, vpa, vperp, z, r)
    @boundscheck z.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect.speed,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(speed))
    if vperp.advection.option == "default" || vperp.advection.option == "constant"
        @inbounds begin
            @loop_r_z_vpa ir iz ivpa begin
                @views advect.speed[:,ivpa,iz,ir] .= vperp.advection.constant_speed
            end
        end
    end
    return nothing
end

end
