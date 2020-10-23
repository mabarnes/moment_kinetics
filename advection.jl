module advection

export advection_1d!

import semi_lagrange: find_approximate_characteristic!
import source_terms: update_speed!
import source_terms: update_advection_factor!
import source_terms: calculate_explicit_source!
import source_terms: update_f!
import chebyshev: update_fcheby!
import chebyshev: update_df_chebyshev!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function advection_1d!(ff, SL, source, coord, use_semi_lagrange, dt, chebyshev)

    if use_semi_lagrange
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        find_approximate_characteristic!(SL, source, coord, dt)
    end
    # Heun's method (RK2) for explicit time advance
    for j ∈ 1:2
        # calculate the advection speed corresponding to current f
        update_speed!(source.speed, coord)
        # calculate the factor appearing in front of df/dz in the advection
        # term at time level n in the frame moving with the approximate
        # characteristic
        update_advection_factor!(source, SL, coord.n, dt, j)
        # Chebyshev transform f to get Chebyshev spectral coefficients
        # and use them to calculate f'
        update_fcheby!(chebyshev, view(ff,:,j), coord)
        update_df_chebyshev!(source.df, chebyshev, coord)
        # calculate the explicit source terms on the rhs of the equation;
        # i.e., -Δt⋅δv⋅f'
        calculate_explicit_source!(source, SL.dep_idx, coord.n, j)
        # update ff at time level n+1 using an explicit Runge-Kutta method
        # along approximate characteristics
        update_f!(ff, source.rhs, SL.dep_idx, coord.n, j)
    end
    @views @. ff[:,1] = 0.5*(ff[:,2] + ff[:,3])
end
# missing argument chebyshev indicates that a finite difference method is being used
function advection_1d!(ff, SL, source, coord, use_semi_lagrange, dt)

    if use_semi_lagrange
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        find_approximate_characteristic!(SL, source, coord, dt)
    end
    # Heun's method (RK2) for explicit time advance
    for j ∈ 1:2
        # calculate the advection speed corresponding to current f
        update_speed!(source.speed, coord)
        # calculate the factor appearing in front of df/dz in the advection
        # term at time level n in the frame moving with the approximate
        # characteristic
        update_advection_factor!(source, SL, coord.n, dt, j)
        # calculate the derivative of f
        update_df_finite_difference!(source.df, ff, coord.cell_width, j,
                source.adv_fac, coord.bc)
        # calculate the explicit source terms on the rhs of the equation;
        # i.e., -Δt⋅δv⋅f'
        calculate_explicit_source!(source, SL.dep_idx, coord.n, j)
        # update ff at time level n+1 using an explicit Runge-Kutta method
        # along approximate characteristics
        update_f!(ff, source.rhs, SL.dep_idx, coord.n, j)
    end
    @views @. ff[:,1] = 0.5*(ff[:,2] + ff[:,3])
end

function update_df_finite_difference!(df, f, del, j, adv_fac, bc)
    n = length(del)
    @boundscheck n == length(df) || throw(BoundsError(df))
    @boundscheck n == length(del) || throw(BoundsError(del))
    @inbounds begin
        if bc == "zero"
            df[1] = 0.
        elseif bc == "periodic"
            df[1] = (f[1,j]-f[n-1,j])/del[1]
        end
        for i ∈ 3:n-2
            if adv_fac[i] < 0
                df[i] =  (3*f[i,j]-4*f[i-1,j]+f[i-2,j])/(2*del[i])
            else
                df[i] = (-f[i+2,j]+4*f[i+1,j]-3*f[i,j])/(2*del[i+1])
            end
        end
        if adv_fac[1] > 0
            df[1] = (-f[3,j]+4*f[2,j]-3*f[1,j])/(2*del[2])
        end
        if adv_fac[2] > 0
            df[2] = (-f[4,j]+4*f[3,j]-3*f[2,j])/(2*del[3])
        else
            # have to modify for periodic
            df[2] = (3*f[2,j]-4*f[1,j])/(2*del[2])
        end
        if adv_fac[n] < 0
            df[n] = (3*f[n,j]-4*f[n-1,j]+f[n-2,j])/(2*del[n])
        end
        if adv_fac[n-1] < 0
            df[n-1] = (3*f[n-1,j]-4*f[n-2,j]+f[n-3,j])/(2*del[n-1])
        else
            # have to modify for periodic
            df[n-1] = (4*f[n,j]-3*f[n-1,j])/(2*del[n])
        end
    end
end

end
