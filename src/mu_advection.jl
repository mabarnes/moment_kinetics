module mu_advection

export mu_advection!
export update_speed_mu!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..looping

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
function mu_advection!(f_out, fvec_in, advect, r, z, mu, vpa,
                      dt, spectral, composition)
    @loop_s is begin
        # get the updated speed along the r direction using the current f
        @views update_speed_mu!(advect[is], vpa, mu, z, r)
        @loop_r_z_vpa ir iz ivpa begin
            @views advance_f_local!(f_out[ivpa,:,iz,ir,is], mu.scratch, advect[is], ivpa,
                                    r, dt, spectral)
        end
    end
end

# calculate the advection speed in the mu-direction at each grid point
function update_speed_mu!(advect, vpa, mu, z, r)
    @boundscheck z.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck mu.n == size(advect.speed,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck r.n == size(advect.speed,4) || throw(BoundsError(speed))
    if mu.advection.option == "default" || mu.advection.option == "constant"
        @inbounds begin
            @loop_r_z_vpa ir iz ivpa begin
                @views advect.speed[:,ivpa,iz,ir] .= 0.0
                # mu = vperp^2/2B has no advection velocity
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        @loop_r_z_vpa ir iz ivpa begin
            @views advect.modified_speed[:,ivpa,iz,ir] .= advect.speed[:,ivpa,iz,ir]
        end
    end
    return nothing
end

end
