"""
Polynomial spectral methods using Lagrange polynomials

Calculations done by matrix multiplication
"""
module lagrange

using ..type_definitions: mk_float

"""
Information for operations with Lagrange polynomials
"""
struct lagrange_info
    collocation_points::Vector{mk_float}
    derivative::Matrix{mk_float}
end

"""
Create arrays for Lagrange polynomial operations
"""
function setup_lagrange(collocation_points::Vector{mk_float})
    return lagrange_info(collocation_points,
                         construct_derivative_matrix(collocation_points))
end

"""
Construct matrix for a Lagrange polynomial derivative from collocation points.
"""
function construct_derivative_matrix(collocation_points)
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
    derivative_matrix = Matrix{mk_float}(undef, n, n)

    function l_prime(i, x)
        result = 0.0
        for j ∈ 1:n
            j == i && continue

            product = 1.0
            for k ∈ 1:n
                (k == i || k == j) && continue

                product *= (x - collocation_points[k]) /
                           (collocation_points[i] - collocation_points[k])
            end
            result += product / (collocation_points[i] - collocation_points[j])
        end

        return result
    end

    for i ∈ 1:n, j ∈ 1:n
        derivative_matrix[i,j] = l_prime(i, collocation_points[j])
    end

    return derivative_matrix
end

end # lagrange
