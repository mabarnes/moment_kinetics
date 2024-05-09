"""
Lagrange polynomials can be useful for finite element methods on any set of basis points,
as they give a representation of the interpolating function within an element whose
coefficients are just the function values at the grid points.

This module collects some functions related to the use of Lagrange polynomials, to avoid
their being scattered (and possibly duplicated) in other modules.
"""
module lagrange_polynomials

export lagrange_poly

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

end
