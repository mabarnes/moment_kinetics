module z_advection

export z_advection!
export update_speed_z!

using semi_lagrange: find_approximate_characteristic!
using advection: advance_f_local!, update_boundary_indices!
using chebyshev: chebyshev_info

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
function z_advection!(f_out, f_in, ff, SL, source, z, vpa,
                      use_semi_lagrange, dt, t, spectral, istage)
    # get the updated speed along the z direction using the current f
    update_speed_z!(source, vpa, z, t)
    # update the upwind/downwind boundary indices and upwind_increment
    update_boundary_indices!(source)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for ivpa ∈ 1:vpa.n
            find_approximate_characteristic!(SL[ivpa], source[ivpa], z, dt)
        end
    end
    # advance z-advection equation
    for ivpa ∈ 1:vpa.n
        @views advance_f_local!(f_out[:,ivpa], f_in[:,ivpa],
            ff[:,ivpa], SL[ivpa], source[ivpa], z, dt, istage, spectral, use_semi_lagrange)
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_z!(source, vpa, z, t)
    @boundscheck vpa.n == size(source,1) || throw(BoundsError(source))
    @boundscheck z.n == size(source[1].speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = vpa.grid[j]
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = z.advection.constant_speed
                end
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = z.advection.constant_speed*(z.grid[i]+0.5*z.L)
                end
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
                end
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        for j ∈ 1:vpa.n
            @. source[j].modified_speed = source[j].speed
        end
    end
    return nothing
end

end
