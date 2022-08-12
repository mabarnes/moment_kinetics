"""
"""
module clenshaw_curtis

export clenshawcurtisweights

"""
Compute nodes of the Clenshaw—Curtis quadrature rule.
"""
clenshawcurtisnodes(float_type::Type, N::Int) = chebyshevpoints(float_type, N)

"""
Compute weights of the Clenshaw—Curtis quadrature rule with modified Chebyshev moments of the first kind (μ)
"""
clenshawcurtisweights()
clenshawcurtisweights(μ) = clenshawcurtisweights!(copy(μ))
function clenshawcurtisweights!(μ::Vector{float_type}) where float_type
    N = length(μ)
    μ ./= float_type(N - 1)
    μ = inefficient_dct(μ)
    μ[1] *= float_type(0.5); μ[N] *= float_type(0.5)
    return μ
end

"""
"""
function chebyshevpoints(float_type, n)
    grid = zeros(float_type, n)
    nfac = 1/float_type(n-1)
    @inbounds begin
        # calculate z = cos(θ) ∈ [1,-1]
        for j ∈ 1:n
            grid[j] = cospi((j-1)*nfac)
        end
    end
    return grid
end

"""
Inefficient but pure-Julia calculation of the 'type-I DCT' (called REDFT00 by FFTW).
Useful because it works correctly with arbitrary-precision floats (e.g.
Quadmath.Float128). Not efficient, but only used during initialisation.

Uses the formula (note this formula uses zero-based indexing)
  Y_k = X_0 + (-1)^k X_{n-1} + 2∑_{j=1}^{n-2} X_j cos(π j k / (n-1))
from https://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
"""
function inefficient_dct(X::Vector{float_type}) where float_type
    n = length(X)
    Y = similar(X)
    for k ∈ 0:n-1
        Y[k+1] = X[1] + float_type((-1)^k) * X[end] +
                 float_type(2)*sum(X[j+1] * cospi(float_type(j*k)/float_type(n-1)) for j ∈ 1:(n-2))
    end
    return Y
end

end
