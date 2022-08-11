"""
Polynomial spectral methods using Lagrange polynomials

Calculations done by matrix multiplication.

See https://en.wikipedia.org/wiki/Lagrange_polynomial
"""
module lagrange

using LinearAlgebra

using ..array_allocation: allocate_float
using ..type_definitions: mk_float
import ..calculus: abstract_spectral_info, elementwise_derivative!
import ..interpolation: interpolate_to_grid_1d!

# Quadmath provides the Float128 type, which we use for increased precision when
# pre-calculating matrix elements.
using Quadmath
const lagrange_float = Float128
# Make implicit conversion to a Float128 an error, to ensure that all parts of a
# calculation are done in Float128 precision. [Note this redefinition gives warning
# messages, so best to just uncomment it for debugging. Also cospi uses a conversion
# which triggers this error, even though it should not actually introduce any
# inaccuracy, so have to replace cospi(x) with cos(float_type(π)*x) when debugging like
# this.]
#Quadmath.convert(::Type{Float128}, x::Number) = error(
#    "lagrange.jl has disabled implicit conversion to Float128 to ensure Float128 "
#    * "calculations use full precision at every step.")
#Quadmath.convert(::Type{Float128}, x::Float128) = x
#Quadmath.convert(::Type{Float128}, x::Integer) = Float128(x)

"""
Information for operations with Lagrange polynomials

This version is for the case when the collocation points are separated by 'physical'
(possibly normalised) distances, so that no scale factor is required for the result of
the derivative operation.
"""
struct lagrange_info <: abstract_spectral_info
    collocation_points::Vector{mk_float}
    derivative::Matrix{mk_float}
    second_derivative::Matrix{mk_float}
    barycentric_weights::Vector{mk_float}
    element_scratch::Vector{mk_float}
end

"""
Information for operations with Lagrange polynomials

This version is for the case when the collocation points are given on the interval
[-1,1], so that a scale factor is required for the result of the derivative operation.
"""
struct scaled_lagrange_info <: abstract_spectral_info
    collocation_points::Vector{mk_float}
    derivative::Matrix{mk_float}
    barycentric_weights::Vector{mk_float}
    second_derivative::Matrix{mk_float}
    element_scratch::Vector{mk_float}
    scale_factor::mk_float
end

"""
    setup_lagrange_pseudospectral(collocation_points::Vector{mk_float},
                                  fine_collocation_points::Vector{lagrange_float},
                                  wgts::Vector{mk_float}; scale_factor=nothing)

Create arrays for Lagrange polynomial operations.

`collocation_points` gives the grid point positions within an element.
`fine_collocation_points` gives a set of (2p+1)=(2*ngrid-1) Gauss-Lobatto points that
can be used for exact numerical integration of order 2p polynomials. It is used to
calculate the exact mass matrix.
`wgts` gives the integration weights on the standard grid.
`scale_factor` can be passed if the collocation_points are given on the standard
interval [-1,1] rather than already being scaled.
"""
function setup_lagrange_pseudospectral(collocation_points::Vector{lagrange_float},
                                       fine_collocation_points::Vector{lagrange_float},
                                       fine_wgts::Vector{lagrange_float}; scale_factor=nothing)
    n = length(collocation_points)

    # Calculate barycentric_weights, used for interpolation
    barycentric_weights = Vector{mk_float}(undef, n)
    for i ∈ 1:n
        product = lagrange_float(1)
        for j ∈ 1:n
            j == i && continue
            product *= collocation_points[i] - collocation_points[j]
        end
        barycentric_weights[i] = inv(product)
    end

    first_derivative_matrix = construct_derivative_matrix(collocation_points)
    mass_matrix = construct_mass_matrix(collocation_points, fine_collocation_points, fine_wgts)
    if scale_factor === nothing
        return lagrange_info(collocation_points,
                             mk_float.(first_derivative_matrix),
                             construct_second_derivative_matrix(collocation_points,
                                                                first_derivative_matrix,
                                                                mass_matrix),
                             barycentric_weights, allocate_float(n))
    else
        return scaled_lagrange_info(collocation_points,
                                    mk_float.(first_derivative_matrix),
                                    construct_second_derivative_matrix(collocation_points,
                                                                       first_derivative_matrix,
                                                                       mass_matrix),
                                    barycentric_weights, allocate_float(n),
                                    scale_factor)
    end
end

"""
Construct 'mass matrix' for Lagrange polynomial basis

The 'mass matrix' M_ij is
  M_ij = ∫ l_i(x) l_j(x) = ∑_I W_I l_i(x_I) l_j(x_I)
where x_I are the Gauss-Lobatto points of a grid with (2p+1) points (the Lagrange basis
functions are order p).
"""
function construct_mass_matrix(collocation_points::Vector{lagrange_float},
                               fine_collocation_points::Vector{lagrange_float},
                               fine_wgts::Vector{lagrange_float})
    n = length(collocation_points)
    n_fine = length(fine_collocation_points)

    # The Lagrange basis functions are given by
    #  l_i(x) = Π_j=1,j≠i^N (x - x_j) / (x_i - x_j)
    function l(i, x)
        return reduce(*, (x - collocation_points[j]) /
                         (collocation_points[i] - collocation_points[j])
                         for j in 1:n if j≠i)
    end
    l_on_fine_grid = Matrix{lagrange_float}(undef, n_fine, n)
    for i ∈ 1:n, I ∈ 1:n_fine
        l_on_fine_grid[I,i] = l(i, fine_collocation_points[I])
    end

    mass_matrix = Matrix{lagrange_float}(undef, n, n)
    for j ∈ 1:n, i ∈ 1:n
        mass_matrix[i,j] = sum(fine_wgts[I] * l_on_fine_grid[I, i] * l_on_fine_grid[I, j]
                               for I ∈ 1:n_fine)
    end

    return mass_matrix
end

"""
Construct matrix for a Lagrange polynomial derivative from collocation points.
"""
function construct_derivative_matrix(collocation_points)::Matrix{lagrange_float}
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

    derivative_matrix = Matrix{lagrange_float}(undef, n, n)
    for j ∈ 1:n, i ∈ 1:n
        derivative_matrix[i,j] = l_prime(j, collocation_points[i])
    end

    return derivative_matrix
end

"""
Construct matrix for a Lagrange polynomial second derivative from collocation points.

This method really represents a diffusion operator with a constant diffusion
coefficient. To get an operator with sensible behaviour at element boundaries, it is
necessary to use the 'weak formulation', and an operator with a non-constant diffusion
coefficient would need to be represented as a 3d array, like (with i,j,k indices of
points within an element)
(∂/∂x(D(x)∂f/∂x))_i = ∑_j,k M_i,j,k*D_j*f_k
or
(D(x) ∂^2f/∂x^2))_i = ∑_j,k N_i,j,k*D_j*f_k


To define the matrix constructed in this method:

All integrations in the following are over a single element, grid points and indices are
within that element.

A function f(x) is represented by an expansion on Lagrange interpolating polynomials
l_i(x), which are each 1 on a single grid point and 0 on all the others, so the
coefficients of the expansion polynomials are just the function values at the grid
points
f(x) ≈ ∑_i f(x_i) l_i(x) ≡ ∑_i f_i l_i(x)    (*)

The derivative of a basis function is a degree (p-1) polynomial, so it can be
represented as a sum of l_i(x)
∂l_i(x)/∂x = ∑_j l_j(x) D_ji    (†)
(note this 'derivative matrix' D_ji has nothing to do with the diffusion coefficient D)
and this also gives the discrete approximation to ∂f/∂x as
∂f/∂x ≈ ∂/∂x ∑_i f_i l_i(x) = ∑_i f_i ∂l_i(x)/∂x = ∑_i,j f_i l_j(x) D_ji
so
(∂f/∂x)_j = ∑_i D_ji f_i

To make numerical integration exact for polynomials up to order 2p, we can use a finer
grid (with (2p+1) points) than we use everywhere else.

Using this numerical integration, the inner product of two basis functions defines the
'mass matrix'
M_ij = ∫ l_i(x) l_j(x) dx = ∑_I W_I l_i(x_I) l_j(x_I)    (‡)

Ignoring any terms other than the diffusion operator, the evolution equation is
∂f/∂t = D ∂^2f/∂x^2
which in weak form is
∫ l_i(x) ∂f/∂t dx = D ∫ l_i(x) ∂^2f/∂x^2 dx
and integrating by parts
∫ l_i(x) ∂f/∂t dx = -D ∫ ∂l_i(x)/∂x ∂f/∂x dx + D [l_i(x) ∂f/∂x]_x_0^x_n
where x_0 is the grid point on the lower edge of the element and x_n is the grid point
at the upper edge.
Substituting in the exansion of f (*) and using the derivative of a basis function (†)
and the inner product of basis functions (‡)
∑_j M_ij ∂f_j/∂t = -D ∑_j,k,l D_ji f_k D_lk M_jl + D ∑_k f_k [D_nk δ_in - D_1k δ_i1]
            = -D ∑_k (∑_j,l D_ji M_jl D_lk) f_k + D ∑_k f_k [D_nk δ_in - D_1k δ_i1]
            = D ∑_k (D_nk δ_in - D_1k δ_i1 - ∑_j,l D_ji M_jl D_lk) f_k
∂f_i/∂t = D ∑_k,m (M^-1)_im(D_nk δ_mn - D_1k δ_m1 - ∑_j,l D_jm M_jl D_lk) * f_k
        = D ∑_k ((M^-1)_in D_nk - (M^-1)_i1 D_1k - ∑_m,j,l (M^-1)_im D_jm M_jl D_lk) * f_k
"""
function construct_second_derivative_matrix(collocation_points,
                                            first_deriv_matrix,
                                            mass_matrix)::Matrix{mk_float}
    n = length(collocation_points)

    inverse_mass_matrix = inv(mass_matrix)

    derivative_matrix = -inverse_mass_matrix * transpose(first_deriv_matrix) *
                         mass_matrix * first_deriv_matrix
    for k ∈ 1:n, i ∈ 1:n
        derivative_matrix[i,k] += inverse_mass_matrix[i,n] * first_deriv_matrix[n,k]
        derivative_matrix[i,k] -= inverse_mass_matrix[i,1] * first_deriv_matrix[1,k]
    end

    return mk_float.(derivative_matrix)
end

"""
    elementwise_derivative!(coord, ff, lagrange::lagrange_info, order::Val{1})

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication.
"""
function elementwise_derivative!(coord, ff, lagrange::lagrange_info, order::Val{1})
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
    elementwise_derivative!(coord, ff, lagrange::scaled_lagrange_info, order::Val{1})

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication
and including a scale factor to convert from a coordinate on the interval [-1,1] to the
physical coordinate.
"""
function elementwise_derivative!(coord, ff, lagrange::scaled_lagrange_info, order::Val{1})
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
        @views df[:,j] .*= lagrange.scale_factor

        k = 1
    end

    return nothing
end

"""
    elementwise_derivative!(coord, ff, lagrange::lagrange_info, order::Val{2})

Calculate f'' using a spectral polynomial method, implemented as a matrix multiplication.

Really a weak-form representation of a diffusion operator with constant diffusion
coefficient, see `construct_second_derivative_matrix()`.
"""
function elementwise_derivative!(coord, ff, lagrange::lagrange_info, order::Val{2})
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
        @views mul!(df[:,j], lagrange.second_derivative, ff[imin:imax])

        k = 1
    end

    return nothing
end

"""
    elementwise_derivative!(coord, ff, lagrange::scaled_lagrange_info, order::Val{2})

Calculate f'' using a spectral polynomial method, implemented as a matrix multiplication
and including a scale factor to convert from a coordinate on the interval [-1,1] to the
physical coordinate.

Really a weak-form representation of a diffusion operator with constant diffusion
coefficient, see `construct_second_derivative_matrix()`.
"""
function elementwise_derivative!(coord, ff, lagrange::scaled_lagrange_info, order::Val{2})
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
        @views mul!(df[:,j], lagrange.second_derivative, ff[imin:imax])
        # and multiply by scaling factor needed to go from scaled coordinate on [-1,1]
        # to actual coordinate
        @views df[:,j] .*= lagrange.scale_factor^2

        k = 1
    end

    return nothing
end

"""
    elementwise_derivative!(coord, ff, adv_fac,
                            lagrange::Union{lagrange_info,scaled_lagrange_info},
                            order::Val{1})

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication.

Note: Lagrange derivative does not make use of upwinding information within each element.
"""
function elementwise_derivative!(coord, ff, adv_fac,
                                 lagrange::Union{lagrange_info,scaled_lagrange_info},
                                 order::Val{1})
    return elementwise_derivative!(coord, ff, lagrange, order)
end

"""
    elementwise_derivative!(coord, ff, adv_fac,
                            lagrange::Union{lagrange_info,scaled_lagrange_info},
                            order::Val{2})

Calculating f'' with upwinding is not implemented (does it even make sense?).
"""
function elementwise_derivative!(coord, ff, adv_fac,
                                 lagrange::Union{lagrange_info,scaled_lagrange_info},
                                 order::Val{2})
    error("Second derivative with upwinding is not implemented")
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
