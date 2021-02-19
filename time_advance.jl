module time_advance

export update_f!
export advance_f_local!
export rk_update_f!

using finite_differences: derivative_finite_difference!
using chebyshev: chebyshev_derivative!
using chebyshev: chebyshev_info
using source_terms: update_advection_factor!
using source_terms: calculate_explicit_source!
using source_terms: update_boundary_indices!

# update the righthand side of the equation to account for 1d advection in this coordinate
function update_rhs!(source, f_current, SL, coord, dt, j, spectral)
    # calculate the factor appearing in front of df/dcoord in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    update_advection_factor!(source.adv_fac,
        source.modified_speed, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL, coord.n, dt, j, coord)
    # calculate df/dcoord
    derivative!(source.df, f_current, coord, source.adv_fac, spectral)
    # calculate the explicit source terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    calculate_explicit_source!(source.rhs, source.df,
        source.adv_fac, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, coord.ngrid, coord.nelement,
        coord.igrid, coord.ielement, j)
end
# Chebyshev transform f to get Chebyshev spectral coefficients and use them to calculate f'
function derivative!(df, f, coord, adv_fac, spectral::chebyshev_info)
    chebyshev_derivative!(df, f, spectral, coord)
end
# calculate the derivative of f using finite differences; stored in df
function derivative!(df, f, coord, adv_fac, not_spectral::Bool)
    derivative_finite_difference!(df, f, coord.cell_width, adv_fac,
        coord.bc, coord.fd_option, coord.igrid, coord.ielement)
end
# do all the work needed to update f(coord) at a single value of other coords
function advance_f_local!(f_new, f_current, f_old, SL, source, coord, dt, j, spectral, use_SL)
    # update the rhs of the equation accounting for 1d advection in coord
    update_rhs!(source, f_current, SL, coord, dt, j, spectral)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    update_f!(f_new, f_old, source.rhs, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, coord.bc, use_SL)
end
#=
# do all the work needed to update f(z) at a single vpa grid point
# using chebyshev spectral method for derivatives
function advance_f_local!(f_new, f_current, f_old, SL, source, coord, dt, j, spectral::chebyshev_info)
    # calculate the factor appearing in front of df/dz in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    update_advection_factor!(source.adv_fac,
        source.modified_speed, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL, coord.n, dt, j, coord)
    # Chebyshev transform f to get Chebyshev spectral coefficients
    # and use them to calculate f'
    chebyshev_derivative!(source.df, f_current, spectral, coord)
    # calculate the explicit source terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    calculate_explicit_source!(source.rhs, source.df,
        source.adv_fac, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, coord.ngrid, coord.nelement,
        coord.igrid, coord.ielement, j)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    update_f!(f_new, f_old, source.rhs, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, coord.bc)
end
# do all the work needed to update f(z) at a single vpa grid point
# using finite difference method for derivatives
# note that the 'not_spectral' is a dummy input whose type indicates
# whether a finite difference or spectral discretization scheme is used
function advance_f_local!(f_new, f_current, f_old, SL, source, coord, dt, j, not_spectral::Bool)
#function advance_f_local!(f_new, f_current, f_old, SL, source, coord, dt, j)
    # calculate the factor appearing in front of df/dz in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    update_advection_factor!(source.adv_fac,
        source.modified_speed, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL, coord.n, dt, j, coord)
    # calculate the derivative of f (f_current) using finite differences,
    # stored in source.df
    derivative_finite_difference!(source.df, f_current,
        coord.cell_width, source.adv_fac, coord.bc, coord.fd_option,
        coord.igrid, coord.ielement)
    # calculate the explicit source terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    calculate_explicit_source!(source.rhs, source.df,
        source.adv_fac, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, coord.ngrid, coord.nelement,
        coord.igrid, coord.ielement, j)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    update_f!(f_new, f_old, source.rhs, source.upwind_idx, source.downwind_idx,
        source.upwind_increment, SL.dep_idx, coord.n, coord.bc)
end
=#
# update ff at time level n+1 using an explicit Runge-Kutta method
# along approximate characteristics
function update_f!(f_new, f_old, rhs, up_idx, down_idx, up_incr, dep_idx, n, bc, use_SL)
    @boundscheck n == length(f_new) || throw(BoundsError(f_new))
    @boundscheck n == length(rhs) || throw(BoundsError(rhs))
    @boundscheck n == length(dep_idx) || throw(BoundsError(dep_idx))
    @boundscheck n == length(f_old) || throw(BoundsError(f_old))

    if use_SL
        # do not update the upwind boundary, where the constant incoming BC has been imposed
        if bc != "periodic"
            f_new[up_idx] = f_old[up_idx]
            istart = up_idx-up_incr
        else
            istart = up_idx
        end
        #@inbounds for i ∈ up_idx-up_incr:-up_incr:down_idx
        @inbounds for i ∈ up_idx:-up_incr:down_idx
            # dep_idx is the index of the departure point for the approximate
            # characteristic passing through grid point i
            # if semi-Lagrange is not used, then dep_idx = i
            idx = dep_idx[i]
            if idx != up_idx + up_incr
                f_new[i] = f_old[idx] + rhs[i]
            else
                # if departure index is beyond upwind boundary, then
                # set updated value along characteristic equal to the old
                # value at the boundary; i.e., assume f is constant
                # beyond the upwind boundary
                f_new[i] = f_old[up_idx]
            end
        end
    else
        @inbounds for i ∈ up_idx:-up_incr:down_idx
            f_new[i] += rhs[i]
        end
    end
    return nothing
end

function rk_update_f!(ff, ff_rk, nz, nvpa, n_rk_stages)
    @boundscheck nz == size(ff_rk,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff_rk,2) || throw(BoundsError(ff_rk))
    @boundscheck n_rk_stages+1 == size(ff_rk,3) || throw(BoundsError(ff_rk))
    @boundscheck nz == size(ff,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff,2) || throw(BoundsError(ff_rk))
    if n_rk_stages == 1
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = ff_rk[iz,ivpa,2]
                end
            end
        end
    elseif n_rk_stages == 2
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = 0.5*(ff_rk[iz,ivpa,2] + ff_rk[iz,ivpa,3])
                end
            end
        end
    elseif n_rk_stages == 3
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = (2.0*(ff_rk[iz,ivpa,3] + ff_rk[iz,ivpa,4])-ff_rk[iz,ivpa,1])/3.0
                end
            end
        end
    end
end

end
