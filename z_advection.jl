module z_advection

export z_advection!
export update_speed_z!

using semi_lagrange: find_approximate_characteristic!
using time_advance: advance_f_local!, rk_update_f!
using source_terms: update_boundary_indices!
using chebyshev: chebyshev_info

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function z_advection!(ff, ff_scratch, SL, source, z, vpa, n_rk_stages,
                      use_semi_lagrange, dt, t, spectral)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
#    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
#    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
#    @boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
#    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
#    @boundscheck size(ff_scratch,3) == n_rk_stages+1 || throw(BoundsError(ff_scratch))
    # SSP RK for explicit time advance
    ff_scratch[:,:,1] .= ff
    # NB: memory usage could be made more efficient here, as ff_scratch[:,:,1]
    # not really needed; just easier to read/write code with it available
    for istage ∈ 1:n_rk_stages
        # for SSP RK3, need to redefine ff_scratch[3]
        if istage == 3
            @. ff_scratch[:,:,istage] = 0.25*(ff_scratch[:,:,istage] +
                ff_scratch[:,:,istage-1] + 2.0*ff)
        end
        z_advection_single_stage!(ff_scratch, ff, SL, source, z, vpa,
                              use_semi_lagrange, dt, t, spectral, istage)
    end
    rk_update_f!(ff, ff_scratch, z.n, vpa.n, n_rk_stages)
end
# do a single stage time advance (potentially as part of a multi-stage RK scheme)
function z_advection_single_stage!(ff_scratch, ff, SL, source, z, vpa,
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
        @views advance_f_local!(ff_scratch[:,ivpa,istage+1], ff_scratch[:,ivpa,istage],
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
