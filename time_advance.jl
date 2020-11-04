module time_advance

export update_f!
export advance_f_local!

using finite_differences: update_df_finite_difference!
using chebyshev: update_fcheby!
using chebyshev: update_df_chebyshev!
using source_terms: update_advection_factor!
using source_terms: calculate_explicit_source!
using source_terms: update_boundary_indices!

# do all the work needed to update f(z) at a single vpa grid point
# using chebyshev spectral method for derivatives
function advance_f_local!(f_new, f_current, f_old, SL, source, coord, dt, spectral, j)
    # calculate the factor appearing in front of df/dz in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    update_advection_factor!(source.adv_fac,
        source.speed, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL, coord.n, dt, j)
    # Chebyshev transform f to get Chebyshev spectral coefficients
    # and use them to calculate f'
    update_fcheby!(spectral, f_current, coord)
    update_df_chebyshev!(source.df, spectral, coord)
    # calculate the explicit source terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    calculate_explicit_source!(source.rhs, source.df,
        source.adv_fac, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, j)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    update_f!(f_new, f_old, source.rhs, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, j)
end
# do all the work needed to update f(z) at a single vpa grid point
# using finite difference method for derivatives
function advance_f_local!(f_new, f_current, f_old, SL, source, coord, dt, j)
    # calculate the factor appearing in front of df/dz in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    update_advection_factor!(source.adv_fac,
        source.speed, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL, coord.n, dt, j)
    # calculate the derivative of f
    update_df_finite_difference!(source.df, f_current,
        coord.cell_width, source.adv_fac, coord.bc)
    # calculate the explicit source terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    calculate_explicit_source!(source.rhs, source.df,
        source.adv_fac, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, j)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    update_f!(f_new, f_old, source.rhs, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, j)
end
# update ff at time level n+1 using an explicit Runge-Kutta method
# along approximate characteristics
function update_f!(f_new, f_old, rhs, up_idx, down_idx, up_incr, dep_idx, n, j)
    @boundscheck n == length(f_new) || throw(BoundsError(f_new))
    @boundscheck n == length(rhs) || throw(BoundsError(rhs))
    @boundscheck n == length(dep_idx) || throw(BoundsError(dep_idx))
    @boundscheck n == length(f_old) || throw(BoundsError(f_old))

    @inbounds for i ∈ up_idx:-up_incr:down_idx
        # dep_idx is the index of the departure point for the approximate
        # characteristic passing through grid point i
        # if semi-Lagrange is not used, then dep_idx = i
        idx = dep_idx[i]
        if idx != up_idx + up_incr
            f_new[i] = f_old[idx] + rhs[i]
        else
            # NB: need to re-examine this for case with non-advective terms
            #f_new[i] = f_old[up_idx]
            f_new[i] = 0.0
        end
    end
    return nothing
end

end
