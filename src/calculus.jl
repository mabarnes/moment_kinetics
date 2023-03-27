"""
"""
module calculus

export derivative!
export integral

using ..moment_kinetics_structs: chebyshev_info
using ..type_definitions: mk_float

"""
    elementwise_derivative!(coord, f, adv_fac, spectral)
    elementwise_derivative!(coord, f, spectral)

Generic function for element-by-element derivatives

First signature, with `adv_fac`, calculates an upwind derivative, the second signature
calculates a derivative without upwinding information.

Result is stored in coord.scratch_2d.
"""
function elementwise_derivative! end

"""
    derivative!(df, f, coord, adv_fac, spectral)

Upwinding derivative.
"""
function derivative!(df, f, coord, adv_fac, spectral::Union{Bool,<:chebyshev_info})
    # get the derivative at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_derivative!(coord, f, adv_fac, spectral, Val(1))
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the derivative from the upwind element.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord, adv_fac)
end

"""
    derivative!(df, f, coord, spectral, order=Val(1))

Non-upwinding derivative.
"""
function derivative!(df, f, coord, spectral)
    return derivative!(df, f, coord, spectral, Val(1))
end

function derivative!(df, f, coord, spectral, ::Val{1})
    # get the derivative at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_derivative!(coord, f, spectral, Val(1))
    # map the derivative from the elem;ntal grid to the full grid;
    # at element boundaries, use the average of the derivatives from neighboring elements.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord)
end

function derivative!(df, f, coord, spectral::Bool, order::Val{2})
    # Finite difference version must use an appropriate second derivative stencil, not
    # apply the 1st derivative twice as for the spectral element method

    # get the derivative at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_derivative!(coord, f, spectral, order)
    # map the derivative from the elem;ntal grid to the full grid;
    # at element boundaries, use the average of the derivatives from neighboring elements.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord)
end

function derivative!(d2f, f, coord, spectral, ::Val{2})
    # For spectral element methods, calculate second derivative by applying first
    # derivative twice, with special treatment for element boundaries

    # First derivative
    elementwise_derivative!(coord, f, spectral, Val(1))
    derivative_elements_to_full_grid!(coord.scratch3, coord.scratch_2d, coord)
    # Save elementwise first derivative result
    coord.scratch2_2d .= coord.scratch_2d

    # Second derivative for element interiors
    elementwise_derivative!(coord, coord.scratch3, spectral, Val(1))
    derivative_elements_to_full_grid!(d2f, coord.scratch_2d, coord)

    # Add contribution to penalise discontinuous first derivatives at element
    # boundaries. For smooth functions this would do nothing so should not affect
    # convergence of the second derivative. Aims to stabilise numerical instability when
    # spike develops at an element boundary. The coefficient is an arbitrary choice, it
    # should probably be large enough for stability but as small as possible.
    function penalise_discontinuous_first_derivative!(d2f, imin, imax, df)
        # Arbitrary numerical coefficient
        C = 1.0

        # Left element boundary
        d2f[imin] += C * df[1]

        # Right element boundary
        d2f[imax] -= C * df[end]

        return nothing
    end
    @views penalise_discontinuous_first_derivative!(d2f, 1, coord.imax[1],
                                                    coord.scratch2_2d[:,1])
    for ielement ∈ 2:coord.nelement
        @views penalise_discontinuous_first_derivative!(d2f, coord.imin[ielement]-1,
                                                        coord.imax[ielement],
                                                        coord.scratch2_2d[:,ielement])
    end

    if coord.bc ∈ ("wall", "zero", "both_zero")
        # For stability don't contribute to evolution at boundaries, in case these
        # points are not set by a boundary condition.
        d2f[1] = 0.0
        d2f[end] = 0.0
    end

    return nothing
end

"""
"""
function derivative_elements_to_full_grid!(df1d, df2d, coord, adv_fac::AbstractArray{mk_float,1})
    # no changes need to be made for the derivative at points away from element boundaries
    elements_to_full_grid_interior_pts!(df1d, df2d, coord)
    # resolve the multi-valued nature of the derivative at element boundaries
    # by using the derivative from the upwind element
    reconcile_element_boundaries_upwind!(df1d, df2d, coord, adv_fac)
    return nothing
end

"""
"""
function derivative_elements_to_full_grid!(df1d, df2d, coord)
    # no changes need to be made for the derivative at points away from element boundaries
    elements_to_full_grid_interior_pts!(df1d, df2d, coord)
    # resolve the multi-valued nature of the derivative at element boundaries
    # by using the derivative from the upwind element
    reconcile_element_boundaries_centered!(df1d, df2d, coord)
    return nothing
end

"""
maps the derivative at points away from element boundaries
from the grid/element representation to the full grid representation
"""
function elements_to_full_grid_interior_pts!(df1d, df2d, coord)
    # for efficiency, define ngm1 to be ngrid-1, as it will be used repeatedly
    ngm1 = coord.ngrid-1
    # treat the first element
    df1d[2:ngm1] .= @view df2d[2:ngm1,1]
    # deal with any additional elements
    if coord.nelement > 1
        for ielem ∈ 2:coord.nelement
            @. df1d[coord.imin[ielem]:coord.imax[ielem]-1] = @view df2d[2:ngm1,ielem]
        end
    end
    return nothing
end

"""
if at the boundary point within the element, must carefully
choose which value of df to use; this is because
df is multi-valued at the overlapping point at the boundary
between neighboring elements.
here we choose to use the value of df from the upwind element.
"""
function reconcile_element_boundaries_upwind!(df1d, df2d, coord, adv_fac::AbstractArray{mk_float,1})
    # note that the first ngrid points are classified as belonging to the first element
    # and the next ngrid-1 points belonging to second element, etc.

    # first deal with domain boundaries
    if coord.bc == "periodic"
        # consider left domain boundary
        if adv_fac[1] > 0.0
            # adv_fac > 0 corresponds to negative advection speed, so
            # use derivative information from upwind element at larger coordinate value
            df1d[1] = df2d[1,1]
        elseif adv_fac[1] < 0.0
            # adv_fac < 0 corresponds to positive advection speed, so
            # use derivative information from upwind element at smaller coordinate value
            df1d[1] = df2d[coord.ngrid,coord.nelement]
        else
            # adv_fac = 0, so no upwinding required;
            # use average value
            df1d[1] = 0.5*(df2d[1,1]+df2d[coord.ngrid,coord.nelement])
        end
        # consider right domain boundary
        if adv_fac[coord.n] > 0.0
            # adv_fac > 0 corresponds to negative advection speed, so
            # use derivative information from upwind element at larger coordinate value
            df1d[coord.n] = df2d[1,1]
        elseif adv_fac[coord.ngrid] < 0.0
            # adv_fac < 0 corresponds to positive advection speed, so
            # use derivative information from upwind element at smaller coordinate value
            df1d[coord.n] = df2d[coord.ngrid,coord.nelement]
        else
            # adv_fac = 0, so no upwinding required;
            # use average value
            df1d[coord.n] = 0.5*(df2d[1,1]+df2d[coord.ngrid,coord.nelement])
        end
    else
        df1d[1] = df2d[1,1]
        df1d[coord.n] = df2d[coord.ngrid,coord.nelement]
    end
    # next consider remaining elements, if any.
    # only need to consider interior element boundaries
    if coord.nelement > 1
        for ielem ∈ 2:coord.nelement
            im1 = ielem-1
            # consider left element boundary
            if adv_fac[coord.imax[im1]] > 0.0
                # adv_fac > 0 corresponds to negative advection speed, so
                # use derivative information from upwind element at larger coordinate value
                df1d[coord.imax[im1]] = df2d[1,ielem]
            elseif adv_fac[coord.imax[im1]] < 0.0
                # adv_fac < 0 corresponds to positive advection speed, so
                # use derivative information from upwind element at smaller coordinate value
                df1d[coord.imax[im1]] = df2d[coord.ngrid,im1]
            else
                # adv_fac = 0, so no upwinding required;
                # use average value
                df1d[coord.imax[im1]] = 0.5*(df2d[1,ielem]+df2d[coord.ngrid,im1])
            end
        end
    end
    return nothing
end

"""
if at the boundary point within the element, must carefully
choose which value of df to use; this is because
df is multi-valued at the overlapping point at the boundary
between neighboring elements.
here we choose to use the value of df from the upwind element.
"""
function reconcile_element_boundaries_centered!(df1d, df2d, coord)
    # note that the first ngrid points are classified as belonging to the first element
    # and the next ngrid-1 points belonging to second element, etc.

    # first deal with domain boundaries
    if coord.bc == "periodic"
        # consider left domain boundary
        df1d[1] = 0.5*(df2d[1,1]+df2d[coord.ngrid,coord.nelement])
        # consider right domain boundary
        df1d[coord.n] = df1d[1]
    else
        df1d[1] = df2d[1,1]
        df1d[coord.n] = df2d[coord.ngrid,coord.nelement]
    end
    # next consider remaining elements, if any.
    # only need to consider interior element boundaries
    if coord.nelement > 1
        for ielem ∈ 2:coord.nelement
            im1 = ielem-1
            # consider left element boundary
            df1d[coord.imax[im1]] = 0.5*(df2d[1,ielem]+df2d[coord.ngrid,im1])
        end
    end
    return nothing
end

"""
Computes the integral of the integrand, using the input wgts
"""
function integral(integrand, wgts)
    # n is the number of grid points
    n = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n == length(integrand) || throw(BoundsError(integrand))
    @boundscheck n == length(wgts) || throw(BoundsError(wgts))
    @inbounds for i ∈ 1:n
        integral += integrand[i]*wgts[i]
    end
    return integral
end

"""
Computes the integral of the integrand multiplied by v, using the input wgts
"""
function integral(integrand, v, wgts)
    # n is the number of grid points
    n = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n == length(integrand) || throw(BoundsError(integrand))
    @boundscheck n == length(v) || throw(BoundsError(v))
    @boundscheck n == length(wgts) || throw(BoundsError(wgts))
    @inbounds for i ∈ 1:n
        integral += integrand[i] * v[i] * wgts[i]
    end
    return integral
end

"""
Computes the integral of the integrand multiplied by v^n, using the input wgts
"""
function integral(integrand, v, n, wgts)
    # n is the number of grid points
    n_v = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n_v == length(integrand) || throw(BoundsError(integrand))
    @boundscheck n_v == length(v) || throw(BoundsError(v))
    @boundscheck n_v == length(wgts) || throw(BoundsError(wgts))
    @inbounds for i ∈ 1:n_v
        integral += integrand[i] * v[i] ^ n * wgts[i]
    end
    return integral
end

end
