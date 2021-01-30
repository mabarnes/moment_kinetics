module z_advection

export z_advection!
export update_speed_z!

using moment_kinetics_input: advection_speed, advection_speed_option_z
using moment_kinetics_input: z_adv_oscillation_amplitude, z_adv_frequency
using semi_lagrange: find_approximate_characteristic!
using time_advance: advance_f_local!
using source_terms: update_boundary_indices!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function z_advection!(ff, ff_scratch, SL, source, z, vpa, use_semi_lagrange,
                      dt, spectral, t)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
    @boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,3) == 3 || throw(BoundsError(ff_scratch))
    # Heun's method (RK2) for explicit time advance
    # NB: TMP FOR TESTING !!
    #jend = 1
    jend = 2
    ff_scratch[:,:,1] .= ff
    # NB: memory usage could be made more efficient here, as ff_scratch[:,:,1]
    # not really needed; just easier to read/write code with it available
    for j ∈ 1:jend
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
            @views advance_f_local!(ff_scratch[:,ivpa,j+1], ff_scratch[:,ivpa,j],
                ff[:,ivpa], SL[ivpa], source[ivpa], z, dt, spectral, j)
        end
    end
    #NB: TMP FOR TESTING !!
    if jend == 1
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                for iz ∈ 1:z.n
                    ff[iz,ivpa] = ff_scratch[iz,ivpa,2]
                end
            end
        end
    else
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                for iz ∈ 1:z.n
                    ff[iz,ivpa] = 0.5*(ff_scratch[iz,ivpa,2] + ff_scratch[iz,ivpa,3])
                end
            end
        end
    end
end
# for use with finite difference scheme
function z_advection!(ff, ff_scratch, SL, source, z, vpa, use_semi_lagrange, dt, t)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
    @boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,3) == 3 || throw(BoundsError(ff_scratch))
    # Heun's method (RK2) for explicit time advance
    # NB: TMP FOR TESTING !!
    #jend = 1
    jend = 2
    ff_scratch[:,:,1] .= ff
    for j ∈ 1:jend
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
        for ivpa ∈ 1:vpa.n
            @views advance_f_local!(ff_scratch[:,ivpa,j+1], ff_scratch[:,ivpa,j],
                ff[:,ivpa], SL[ivpa], source[ivpa], z, dt, j)
        end
    end
    if jend == 1
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                for iz ∈ 1:z.n
                    ff[iz,ivpa] = ff_scratch[iz,ivpa,2]
                end
            end
        end
    else
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                for iz ∈ 1:z.n
                    ff[iz,ivpa] = 0.5*(ff_scratch[iz,ivpa,2] + ff_scratch[iz,ivpa,3])
                end
            end
        end
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_z!(source, vpa, z, t)
    @boundscheck vpa.n == size(source,1) || throw(BoundsError(source))
    @boundscheck z.n == size(source[1].speed,1) || throw(BoundsError(speed))
    if advection_speed_option_z == "default"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = vpa.grid[j]
                end
            end
        end
    elseif advection_speed_option_z == "constant"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = advection_speed
                end
            end
        end
    elseif advection_speed_option_z == "linear"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = advection_speed*(z.grid[i]+0.5*z.L)
                end
            end
        end
    elseif advection_speed_option_z == "oscillating"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = advection_speed*(1.0
                        + z_adv_oscillation_amplitude*sinpi(t*z_adv_frequency))
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
