"""
Polynomial spectral methods using Lagrange polynomials

Calculations done by matrix multiplication.

See https://en.wikipedia.org/wiki/Lagrange_polynomial
"""
module lagrange

export setup_lagrange_pseudospectral, lagrange_weights

using LinearAlgebra

using ..array_allocation: allocate_float
using ..type_definitions: mk_float
import ..calculus: elementwise_derivative!
import ..interpolation: interpolate_to_grid_1d!

# Quadmath provides the Float128 type, which we use for increased precision when
# pre-calculating matrix elements.
using Quadmath
const lagrange_float = Float128

"""
Information for operations with Lagrange polynomials

This version is for the case when the collocation points are separated by 'physical'
(possibly normalised) distances, so that no scale factor is required for the result of
the derivative operation.
"""
struct lagrange_info
    collocation_points::Vector{mk_float}
    derivative::Matrix{mk_float}
    barycentric_weights::Vector{mk_float}
    element_scratch::Vector{mk_float}
end

"""
Information for operations with Lagrange polynomials

This version is for the case when the collocation points are given on the interval
[-1,1], so that a scale factor is required for the result of the derivative operation.
"""
struct scaled_lagrange_info
    collocation_points::Vector{mk_float}
    derivative::Matrix{mk_float}
    barycentric_weights::Vector{mk_float}
    element_scratch::Vector{mk_float}
    scale_factor::mk_float
end

"""
Create arrays for Lagrange polynomial operations
"""
function setup_lagrange_pseudospectral(collocation_points::Vector{mk_float}; scale_factor=nothing)
    n = length(collocation_points)

    # Calculate barycentric_weights, used for interpolation
    barycentric_weights = Vector{mk_float}(undef, n)
    for i ∈ 1:n
        product = lagrange_float(1.0)
        for j ∈ 1:n
            j == i && continue
            product *= collocation_points[i] - collocation_points[j]
        end
        barycentric_weights[i] = inv(product)
    end

    if scale_factor === nothing
        return lagrange_info(collocation_points,
                             construct_derivative_matrix(collocation_points),
                             barycentric_weights, allocate_float(n))
    else
        return scaled_lagrange_info(collocation_points,
                                    construct_derivative_matrix(collocation_points),
                                    barycentric_weights, allocate_float(n),
                                    scale_factor)
    end
end

"""
Construct matrix for a Lagrange polynomial derivative from collocation points.
"""
function construct_derivative_matrix(collocation_points_in)::Matrix{mk_float}
    # The Lagrange interpolating polynomial through N points is
    #   ∑_i=1^N f_i l_i(x)
    # where
    #   l_i(x) = Π_j=1,j≠i^N (x - x_j) / (x_i - x_j)
    # and f_i is the value of the function at x_i
    #
    # Therefore the derivative is
    #   ∑_i=1^N f_i l'_i(x)
    # where
    #   l'_i(x) = d/dx{Π_j=1,j≠i^N (x - x_j) / (x_i - x_j)}
    #           = ∑_j=1,j≠i^N 1/(x_i - x_j) Π_k=0,k≠i,j^(N-1) (x - x_k) / (x_i - x_k)
    # which we need to evaluate at the collocation points x_i to express the derivative
    # as
    #   f'_i = ∑_j D_ij f_j
    # with
    #   D_ij = l'j(x_i)

    # Use high-precision arithmetic so rounding errors don't mess up our calculation of
    # matrix elements
    collocation_points = lagrange_float.(collocation_points_in)

    n = length(collocation_points)

    function l_prime(i, x)
        result = lagrange_float(0.0)
        for j ∈ 1:n
            j == i && continue

            product = lagrange_float(1.0)
            for k ∈ 1:n
                (k == i || k == j) && continue

                product *= (x - collocation_points[k]) /
                           (collocation_points[i] - collocation_points[k])
            end
            result += product / (collocation_points[i] - collocation_points[j])
        end

        return result
    end

    derivative_matrix = Matrix{mk_float}(undef, n, n)
    for i ∈ 1:n, j ∈ 1:n
        derivative_matrix[i,j] = l_prime(j, collocation_points[i])
    end

    return derivative_matrix
end

"""
Construct integration weights using Lagrange polynomials
"""
function lagrange_weights(grid, ngrid, nelement, n, L, imin, imax)
    weights = zeros(lagrange_float, n)

    collocation_points = lagrange_float.(grid[1:ngrid])

    # Shift collocation points so that they are centered on 0
    collocation_points .-= (collocation_points[1] + collocation_points[end]) / 2.0

    # Create weights for the first element
    for i ∈ 1:ngrid
        # Need to integrate the i'th Lagrange polynomial
        #   l_i(x) = Π_j=1,j≠i^N (x - x_j) / (x_i - x_j)
        # Let the denominator be denoted by
        #   d_i = Π_j=1,j≠i^N (x_i - x_j)
        # The coefficient of x^(N - 1 - k) in d_i*l_i(x) is
        #   {1, -∑_j=1,j≠i^N x_j, ..., (-1)^k*k!*∑_j1=1,j1≠i^N x_j1 ∑_j2=j1+1,j2≠i^N x_j2 ...∑_jk=j(k-1),jk≠i^N x_jk
        # and the integral of x^(N-1-k) is
        #   ∫_x0^xN x^k dx = [x^(N-k)/(N-k)]_x0^xN = (xN^(N-k) - x0^(N-k))/(N-k)

        d_i = lagrange_float(1.0)
        for j ∈ 1:ngrid
            j == i && continue
            d_i *= (collocation_points[i] - collocation_points[j])
        end

        function sumfunc(k, level=0, lstart=1)
            if level == k
                return 1
            end
            result = lagrange_float(0.0)
            for l ∈ lstart:ngrid
                l == i && continue
                result += collocation_points[l] * sumfunc(k, level + 1, l + 1)
            end
            return result
        end
        for k ∈ 0:(ngrid - 1)
            weights[i] += (
                (-1)^k *
                sumfunc(k) *
                (collocation_points[ngrid]^(ngrid - k) -
                 collocation_points[1]^(ngrid - k)) /
                (ngrid - k)
            )
        end

        weights[i] /= d_i
    end

    # Copy the weights from the first element to the remaining elements.
    # Done in reverse order so that weights[ngrid] gets altered last, after it is read
    # for the last time.
    for ielement ∈ nelement:-1:2
        weights[imin[ielement]-1:imax[ielement]] += weights[1:ngrid]
    end

    return mk_float.(weights)
end

"""
    elementwise_derivative!(coord, ff, lagrange::lagrange_info)

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication.
"""
function elementwise_derivative!(coord, ff, lagrange::lagrange_info)
    df = coord.scratch_2d

    k = 0
    # calculate the pseudospectral derivative on each element
    @inbounds for j ∈ 1:coord.nelement
        # imin is the minimum index on the full grid for this (jth) element
        # the 'k' below accounts for the fact that the first element includes
        # both boundary points, while each additional element shares a boundary
        # point with neighboring elements.  the choice was made when defining
        # coord.imin to exclude the lower boundary point in each element other
        # than the first so that no point is double-counted
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        # Calculate matrix-mulitply using LinearAlgebra (which should ultimately call
        # LAPACK/BLAS)
        @views mul!(df[:,j], lagrange.derivative, ff[imin:imax])

        k = 1
    end

    return nothing
end

"""
    elementwise_derivative!(coord, ff, lagrange::scaled_lagrange_info)

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication
and including a scale factor to convert from a coordinate on the interval [-1,1] to the
physical coordinate.
"""
function elementwise_derivative!(coord, ff, lagrange::scaled_lagrange_info)
    df = coord.scratch_2d

    k = 0
    # calculate the pseudospectral derivative on each element
    @inbounds for j ∈ 1:coord.nelement
        # imin is the minimum index on the full grid for this (jth) element
        # the 'k' below accounts for the fact that the first element includes
        # both boundary points, while each additional element shares a boundary
        # point with neighboring elements.  the choice was made when defining
        # coord.imin to exclude the lower boundary point in each element other
        # than the first so that no point is double-counted
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        # Calculate matrix-mulitply using LinearAlgebra (which should ultimately call
        # LAPACK/BLAS)
        @views mul!(df[:,j], lagrange.derivative, ff[imin:imax])
        # and multiply by scaling factor needed to go from scaled coordinate on [-1,1]
        # to actual coordinate
        @views df[:,j] .= lagrange.scale_factor

        k = 1
    end

    return nothing
end

"""
    elementwise_derivative!(coord, ff, adv_fac, lagrange::lagrange_info)

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication.

Note: Lagrange derivative does not make use of upwinding information within each element.
"""
function elementwise_derivative!(coord, ff, adv_fac,
                                 lagrange::Union{lagrange_info,scaled_lagrange_info})
    return elementwise_derivative!(coord, ff, lagrange)
end

"""
Interpolation from a regular grid to a 1d grid with arbitrary spacing

Arguments
---------
new_grid : Array{mk_float, 1}
    Grid of points to interpolate `coord` to
f : Array{mk_float}
    Field to be interpolated
coord : coordinate
    `coordinate` struct giving the coordinate along which f varies
lagrange : lagrange_info
    struct containing information for Lagrange pseudospectral operations

Returns
-------
result : Array
    Array with the values of `f` interpolated to the points in `new_grid`.
"""
function interpolate_to_grid_1d!(result, newgrid, f, coord, lagrange::Union{lagrange_info,scaled_lagrange_info})
    # define local variable nelement for convenience
    nelement = coord.nelement

    n_new = size(newgrid)[1]
    # Find which points belong to which element.
    # kstart[j] contains the index of the first point in newgrid that is within element
    # j, and kstart[nelement+1]=n_new.
    # Assumes points in newgrid are sorted.
    # May not be the moste efficient algorithm.
    kstart = [1]
    k = 1
    @inbounds for j ∈ 1:nelement
        while true
            if k == n_new+1 || newgrid[k] > coord.grid[coord.imax[j]]
                push!(kstart, k)
                break
            end

            k += 1

            if k == n_new+1 || newgrid[k] > coord.grid[coord.imax[j]]
                push!(kstart, k)
                break
            end
        end
    end

    # First element includes both boundary points, while all others have only one (to
    # avoid duplication), so calculate the first element outside the loop.
    if kstart[1] < kstart[2]
        result[kstart[1]:kstart[2]-1] =
            lagrange_interpolate_single_element(newgrid[kstart[1]:kstart[2]-1],
                                                f[coord.imin[1]:coord.imax[1]], 1,
                                                coord, lagrange)
    end
    @inbounds for j ∈ 2:nelement
        if kstart[j] < kstart[j+1]
            result[kstart[j]:kstart[j+1]-1] =
                lagrange_interpolate_single_element(newgrid[kstart[j]:kstart[j+1]-1],
                                                    f[coord.imin[j]-1:coord.imax[j]], j,
                                                    coord, lagrange)
        end
    end

    return result
end

"""
    lagrange_interpolate_single_element(newgrid, f, j, coord, lagrange::lagrange_info)
    lagrange_interpolate_single_element(newgrid, f, j, coord, lagrange::scaled_lagrange_info)

Note: implemented as a single generic function with a couple of conditional statements
depending on the type of `lagrange`. These conditionals only depend on 'static' (i.e.
compile-time) information, so should be resolved by the compiler with no run-time cost.
Implementing like this avoids duplicated code.

Algorithm note
--------------

The interpolation is currently implemented by directly evaluating the Lagrange
polynomials. This will be O(n_new*n^2) where n_new=length(newgrid),
n=length(lagrange.collocation_points). If performance profiling shows that this function
is a significant run-time cost, it would be worth evaluating the performance/robustness
trade-off offered by the alternatives discussed below.

It might be quicker to evaluate by first doing a matrix multiplication to turn the
function values into coefficients of either monomials (x^i) or Chebyshev polynomials
(the matrix could be pre-calculated during setup), and then using those coefficients to
evaluate the interpolated values. Using coefficients should be O(n^2) (for the matrix
multiplication) plus O(n_new*n) (i.e. a vector operation for each point of newgrid),
because the monomials can be evaluated simply (like z_pow_i *= z at each step of the
loop) and the Chebyshev polynomials have a recursion relation that can be used (like in
[chebyshev.chebyshev_interpolate_single_element](@ref)).

However, the present version might be more robust - if the `newgrid` points are exactly
on the original grid points, this implementation should have minimal rounding errors,
because the contributions from all components f[j] at newgrid[i] that have j≠i will be
multiplied by exactly 0.0.

Note that there are cheaper methods for evaluating the Lagrange interpolation function
(as described on the Wikipedia page), known as the 'barycentric form' and the 'second
form or true form of the barycentric interpolation formula'. These should be O(n_new*n),
skipping the matrix-multiplication O(n^2) cost of the monomial or Chebyshev methods
suggested above. However, these formulas are bad numerically, because they involve
dividing by (x - x_i), so produce `Inf` values when x==x_i.
"""
function lagrange_interpolate_single_element(newgrid, f, j, coord, lagrange::T) where
        T <: Union{lagrange_info,scaled_lagrange_info}

    # Array for the result
    result = similar(newgrid, mk_float)

    scratch = lagrange.element_scratch

    # Need to transform newgrid values to a scaled z-coordinate associated with the
    # collocation points. Transform is a shift and scale so that the element coordinate
    # goes from -1 to 1
    imin = j == 1 ? coord.imin[1] : coord.imin[j] - 1
    imax = coord.imax[j]
    shift = 0.5 * (coord.grid[imin] + coord.grid[imax])
    if T === scaled_lagrange_info
        scale = 2.0 / (coord.grid[imax] - coord.grid[imin])
    end

    for (i, x) ∈ enumerate(newgrid)
        if T === scaled_lagrange_info
            z = scale * (x - shift)
        else
            z = x - shift
        end

        # Note 'barycentric weights' are also give the x-independent denominators of the
        # Lagrange polynomials
        @. scratch = f * lagrange.barycentric_weights
        for k ∈ 1:length(f)
            factor = (z - lagrange.collocation_points[k])
            for l ∈ 1:length(f)
                l == k && continue
                scratch[l] *= factor
            end
        end

        result[i] = sum(scratch)
    end

    return result
end

end # lagrange
