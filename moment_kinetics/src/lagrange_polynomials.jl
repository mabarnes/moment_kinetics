"""
Lagrange polynomials can be useful for finite element methods on any set of basis points,
as they give a representation of the interpolating function within an element whose
coefficients are just the function values at the grid points.

This module collects some functions related to the use of Lagrange polynomials, to avoid
their being scattered (and possibly duplicated) in other modules.
"""
module lagrange_polynomials

export lagrange_poly, lagrange_poly_optimised, lagrange_poly_derivative_optimised

"""
Lagrange polynomial
args:
j - index of l_j from list of nodes
x_nodes - array of x node values
x - point where interpolated value is returned
"""
function lagrange_poly(j,x_nodes,x)
    # get number of nodes
    n = size(x_nodes,1)
    # location where l(x0) = 1
    x0 = x_nodes[j]
    # evaluate polynomial
    poly = 1.0
    for i in 1:j-1
            poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
    end
    for i in j+1:n
            poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
    end
    return poly
end

"""
    lagrange_poly_optimised(other_nodes, one_over_denominator, x)

Optimised version of Lagrange polynomial calculation, making use of pre-calculated quantities.

`other_nodes` is a vector of the grid points in this element where this Lagrange
polynomial is zero (the other nodes than the one where it is 1).

`one_over_denominator` is `1/prod(x0 - n for n ∈ other_nodes)` where `x0` is the grid
point where this Lagrange polynomial is 1.

`x` is the point to evaluate the Lagrange polynomial at.
"""
function lagrange_poly_optimised(other_nodes, one_over_denominator, x)
    return prod(x - n for n ∈ other_nodes) * one_over_denominator
end

"""
    lagrange_poly_derivative_optimised(other_nodes, one_over_denominator, x)

Optimised calculation of the first derivative of a Lagrange polynomial, making use of
pre-calculated quantities.

`other_nodes` is a vector of the grid points in this element where this Lagrange
polynomial is zero (the other nodes than the one where it is 1).

`one_over_denominator` is `1/prod(x0 - n for n ∈ other_nodes)` where `x0` is the grid
point where this Lagrange polynomial is 1.

`x` is the point to evaluate the Lagrange polynomial at.
"""
function lagrange_poly_derivative_optimised(other_nodes, one_over_denominator, x)
    result = 0.0
    k = length(other_nodes)
    # Is there a more efficient way of doing this? Not a big deal for now because this
    # function will only be used to calculate a preconditioner matrix, which is done
    # rarely.
    for i ∈ 1:k
        result += prod(x - other_nodes[j] for j ∈ 1:k if j ≠ i)
    end
    result *= one_over_denominator
    return result
end

end
