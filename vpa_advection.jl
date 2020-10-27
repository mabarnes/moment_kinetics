module vpa_advection

export vpa_advection!
export update_speed_vpa!

using finite_differences: update_df_finite_difference!
using moment_kinetics_input: advection_speed, advection_speed_option
using semi_lagrange: find_approximate_characteristic!
using source_terms: update_advection_factor!
using source_terms: calculate_explicit_source!
using source_terms: update_f!
using chebyshev: update_fcheby!
using chebyshev: update_df_chebyshev!
using em_fields: update_phi!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function vpa_advection!(ff, phi, moments, SL, source, vpa, z, use_semi_lagrange, dt,
	vpa_spectral)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
    @boundscheck size(ff,3) == 3 || throw(BoundsError(ff))
    # get the updated speed along the vpa direction
    update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,1), vpa, z)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for iz ∈ 1:z.n
            find_approximate_characteristic!(SL[iz], view(source.speed,iz,:), vpa, dt)
        end
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
    for j ∈ 1:jend
        for iz ∈ 1:z.n
            # calculate the factor appearing in front of df/dvpa in the advection
            # term at time level n in the frame moving with the approximate
            # characteristic
            update_advection_factor!(view(source.adv_fac,iz,:),
                view(source.speed,iz,:), SL[iz], vpa.n, dt, j)
            # Chebyshev transform f to get Chebyshev spectral coefficients
            # and use them to calculate f'
            update_fcheby!(vpa_spectral, view(ff,iz,:,j), vpa)
            update_df_chebyshev!(view(source.df,iz,:), vpa_spectral, vpa)
            # calculate the explicit source terms on the rhs of the equation;
            # i.e., -Δt⋅δv⋅f'
            calculate_explicit_source!(view(source.rhs,iz,:), view(source.df,iz,:),
                view(source.adv_fac,iz,:), SL[iz].dep_idx, vpa.n, j)
            # update ff at time level n+1 using an explicit Runge-Kutta method
            # along approximate characteristics
            update_f!(view(ff,iz,:,:), view(source.rhs,iz,:), SL[iz].dep_idx, vpa.n, j)
        end
        moments.dens_updated = false ; moments.ppar_updated = false
        # calculate the advection speed corresponding to current f
        if j != jend
            update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,j+1), vpa, z)
        end
    end
    @inbounds @fastmath begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa,1] = 0.5*(ff[iz,ivpa,2] + ff[iz,ivpa,3])
            end
        end
    end
end
# for use with finite difference scheme
function vpa_advection!(ff, phi, moments, SL, source, vpa, z, use_semi_lagrange, dt)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(f,1) == z.n || throw(BoundsError(f))
    @boundscheck size(f,2) == vpa.n || throw(BoundsError(f))
    @boundscheck size(f,3) == 3 || throw(BoundsError(f))
    # get the updated speed along the vpa direction
    update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,1), vpa, z)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for iz ∈ 1:z.n
            find_approximate_characteristic!(SL[iz], view(source.speed,iz,:), vpa, dt)
        end
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
    for j ∈ 1:jend
        for iz ∈ 1:z.n
            # calculate the factor appearing in front of df/dz in the advection
            # term at time level n in the frame moving with the approximate
            # characteristic
            @views update_advection_factor!(source.adv_fac[iz,:],
                source.speed[iz,:], SL[iz], vpa.n, dt, j)
            # calculate the derivative of f
            @views update_df_finite_difference!(source.df[iz,:], ff[iz,:,j],
                vpa.cell_width, source.adv_fac[iz,:], vpa.bc)
            # calculate the explicit source terms on the rhs of the equation;
            # i.e., -Δt⋅δv⋅f'
            @views calculate_explicit_source!(source.rhs[iz,:], source.df[iz,:],
                source.adv_fac[iz,:], SL[iz].dep_idx, vpa.n, j)
            # update ff at time level n+1 using an explicit Runge-Kutta method
            # along approximate characteristics
            @views update_f!(ff[iz,:,:], source.rhs[iz,:], SL[iz].dep_idx, vpa.n, j)
        end
        moments.dens_updated = false ; moments.ppar_updated = false
        # calculate the advection speed corresponding to current f
		if j != jend
	   		update_speed_vpa!(source.speed, phi, moments, view(ff,:,:,j+1), vpa, z)
		end
    end
    @inbounds @fastmath begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa,1] = 0.5*(ff[iz,ivpa,2] + ff[iz,ivpa,3])
            end
        end
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(speed, phi, moments, ff, vpa, z)
    @boundscheck z.n == size(speed,1) || throw(BoundsError(speed))
    @boundscheck vpa.n == size(speed,2) || throw(BoundsError(speed))
    if advection_speed_option == "default"
        update_phi!(phi, moments, ff, vpa, z.n)
        @inbounds @fastmath begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    speed[i,j] = advection_speed
                end
            end
        end
    elseif advection_speed_option == "constant"
        @inbounds @fastmath begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    speed[i,j] = advection_speed
                end
            end
        end
    elseif advection_speed_option == "linear"
        @inbounds @fastmath begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    speed[i,j] = advection_speed*(vpa.grid[i]+0.5*vpa.L)
                end
            end
        end
    end
    return nothing
end

end
