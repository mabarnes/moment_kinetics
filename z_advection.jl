module z_advection

export z_advection!
export update_speed_z!

using finite_differences: update_df_finite_difference!
using moment_kinetics_input: advection_speed, advection_speed_option
using semi_lagrange: find_approximate_characteristic!
using source_terms: update_advection_factor!
using source_terms: calculate_explicit_source!
using source_terms: update_f!
using source_terms: update_boundary_indices!
using chebyshev: update_fcheby!
using chebyshev: update_df_chebyshev!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function z_advection!(ff, SL, source, z, vpa, use_semi_lagrange, dt, spectral)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
    @boundscheck size(ff,3) == 3 || throw(BoundsError(ff))
    # get the updated speed along the z direction
    update_speed_z!(source, vpa, z)
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
#=        for iz ∈ 1:z.n
            println("iz: ", iz, "  z: ", z.grid[iz], "  crossing_time: ", SL[2].crossing_time[iz],
                "  dep_pts: ", SL[2].dep_pts[iz], "  dep_idx: ", SL[2].dep_idx[iz],
                "  characteristic_speed: ", SL[2].characteristic_speed[iz])
        end
=#
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
    for j ∈ 1:jend
        for ivpa ∈ 1:vpa.n
            # calculate the factor appearing in front of df/dz in the advection
            # term at time level n in the frame moving with the approximate
            # characteristic
            update_advection_factor!(source[ivpa].adv_fac,
                source[ivpa].speed, source[ivpa].upwind_idx, source[ivpa].downwind_idx,
                source[ivpa].upwind_increment, SL[ivpa], z.n, dt, j)
            # Chebyshev transform f to get Chebyshev spectral coefficients
            # and use them to calculate f'
            update_fcheby!(spectral, view(ff,:,ivpa,j), z)
            update_df_chebyshev!(source[ivpa].df, spectral, z)
            # calculate the explicit source terms on the rhs of the equation;
            # i.e., -Δt⋅δv⋅f'
            calculate_explicit_source!(source[ivpa].rhs, source[ivpa].df,
                source[ivpa].adv_fac, source[ivpa].upwind_idx, source[ivpa].downwind_idx,
                source[ivpa].upwind_increment, SL[ivpa].dep_idx, z.n, j)
            # update ff at time level n+1 using an explicit Runge-Kutta method
            # along approximate characteristics
            update_f!(view(ff,:,ivpa,:), source[ivpa].rhs, source[ivpa].upwind_idx,
                source[ivpa].downwind_idx, source[ivpa].upwind_increment, SL[ivpa].dep_idx,
                z.n, j)
        end
        if j != jend
            # calculate the advection speed corresponding to current f
            update_speed_z!(source, vpa, z)
            # update the upwind/downwind boundary indices and upwind_increment
            update_boundary_indices!(source)
        end
    end
    @inbounds begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa,1] = 0.5*(ff[iz,ivpa,2] + ff[iz,ivpa,3])
            end
        end
    end
end
# for use with finite difference scheme
function z_advection!(ff, SL, source, z, vpa, use_semi_lagrange, dt)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(f,1) == z.n || throw(BoundsError(f))
    @boundscheck size(f,2) == vpa.n || throw(BoundsError(f))
    @boundscheck size(f,3) == 3 || throw(BoundsError(f))
    # get the updated speed along the z direction
    update_speed_z!(source, vpa, z)
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
    # Heun's method (RK2) for explicit time advance
    jend = 2
    for j ∈ 1:jend
        for ivpa ∈ 1:vpa.n
            # calculate the factor appearing in front of df/dz in the advection
            # term at time level n in the frame moving with the approximate
            # characteristic
            update_advection_factor!(source[ivpa].adv_fac,
                source[ivpa].speed, source[ivpa].upwind_idx, source[ivpa].downwind_idx,
                source[ivpa].upwind_increment, SL[ivpa], z.n, dt, j)
            # calculate the derivative of f
            update_df_finite_difference!(source[ivpa].df, view(ff,:,ivpa,j),
                z.cell_width, source[ivpa].adv_fac, z.bc)
            # calculate the explicit source terms on the rhs of the equation;
            # i.e., -Δt⋅δv⋅f'
            calculate_explicit_source!(source[ivpa].rhs, source[ivpa].df,
                source[ivpa].adv_fac, source[ivpa].upwind_idx, source[ivpa].downwind_idx,
                source[ivpa].upwind_increment, SL[ivpa].dep_idx, z.n, j)
            # update ff at time level n+1 using an explicit Runge-Kutta method
            # along approximate characteristics
            update_f!(view(ff,:,ivpa,:), source[ivpa].rhs, source[ivpa].upwind_idx,
                source[ivpa].downwind_idx, source[ivpa].upwind_increment, SL[ivpa].dep_idx,
                z.n, j)
        end
        # calculate the advection speed corresponding to current f
        if j != jend
            update_speed_z!(source, vpa, z)
        end
    end
    @inbounds begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa,1] = 0.5*(ff[iz,ivpa,2] + ff[iz,ivpa,3])
            end
        end
    end
end

# calculate the advection speed in the z-direction at each grid point
function update_speed_z!(source, vpa, z)
    @boundscheck vpa.n == size(source,1) || throw(BoundsError(source))
    @boundscheck z.n == size(source[1].speed,1) || throw(BoundsError(speed))
    if advection_speed_option == "default"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = vpa.grid[j]
                end
            end
        end
    elseif advection_speed_option == "constant"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = advection_speed
                end
            end
        end
    elseif advection_speed_option == "linear"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    source[j].speed[i] = advection_speed*(z.grid[i]+0.5*z.L)
                end
            end
        end
    end
    return nothing
end

end
