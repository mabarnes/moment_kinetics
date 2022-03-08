"""
Polynomial spectral methods using Lagrange polynomials

Calculations done by matrix multiplication
"""
module lagrange

using LinearAlgebra

using ..type_definitions: mk_float
import ..calculus: elementwise_derivative!
import ..interpolation: interpolate_to_grid_1d

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
end

"""
Information for operations with Lagrange polynomials

This version is for the case when the collocation points are given on the interval
[-1,1], so that a scale factor is required for the result of the derivative operation.
"""
struct lagrange_info_scaled
    collocation_points::Vector{mk_float}
    derivative::Matrix{mk_float}
    scale_factor::mk_float
end

"""
Create arrays for Lagrange polynomial operations
"""
function setup_lagrange_pseudospectral(collocation_points::Vector{mk_float}; scale_factor=nothing)
    if scale_factor === nothing
        return lagrange_info(collocation_points,
                             construct_derivative_matrix(collocation_points))
    else
        return lagrange_info_scaled(collocation_points,
                                    construct_derivative_matrix(collocation_points),
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
    collocation_points = lagrange_float.(collocation_points)

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
        derivative_matrix[i,j] = l_prime(i, collocation_points[j])
    end

    return derivative_matrix
end

"""
    elementwise_derivative!(coord, ff, lagrange::lagrange_info)

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication.
"""
function elementwise_derivative!(coord, ff, lagrange::lagrange_info)
    df = coord.scratch_2d

    # Calculate matrix-mulitply using LinearAlgebra (which should ultimately call
    # LAPACK/BLAS)
    mul!(df, lagrange::derivative, ff)

    return nothing
end

"""
    elementwise_derivative!(coord, ff, lagrange::lagrange_info_scaled)

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication
and including a scale factor to convert from a coordinate on the interval [-1,1] to the
physical coordinate.
"""
function elementwise_derivative!(coord, ff, lagrange::lagrange_info_scaled)
    df = coord.scratch_2d

    # Calculate matrix-mulitply using LinearAlgebra (which should ultimately call
    # LAPACK/BLAS)
    mul!(df, lagrange::derivative, ff)

    df .*= lagrange.scale_factor

    return nothing
end

"""
    elementwise_derivative!(coord, ff, adv_fac, lagrange::lagrange_info)

Calculate f' using a spectral polynomial method, implemented as a matrix multiplication.

Note: Lagrange derivative does not make use of upwinding information within each element.
"""
function elementwise_derivative!(coord, ff, adv_fac,
                                 lagrange::Union{lagrange_info,lagrange_info_scaled})
    return elementwise_derivative!(coord, ff, spectral)
end

end # lagrange
