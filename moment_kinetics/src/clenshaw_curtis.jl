"""
"""
module clenshaw_curtis

using FFTW
using LinearAlgebra

export clenshawcurtisweights

"""
"""
plan_clenshawcurtis(μ) = length(μ) > 1 ? FFTW.plan_r2r!(μ, FFTW.REDFT00) : fill!(similar(μ),1)

"""
Compute nodes of the Clenshaw—Curtis quadrature rule.
"""
clenshawcurtisnodes(::Type{T}, N::Int) where T = chebyshevpoints(N)

"""
Compute weights of the Clenshaw—Curtis quadrature rule with modified Chebyshev moments of the first kind (μ)
"""
clenshawcurtisweights()
clenshawcurtisweights(μ) = clenshawcurtisweights!(copy(μ))
clenshawcurtisweights!(μ) = clenshawcurtisweights!(μ, plan_clenshawcurtis(μ))
function clenshawcurtisweights!(μ, plan)
    N = length(μ)
    μ .*= inv(N-1)
    plan*μ
    μ[1] *= 0.5; μ[N] *= 0.5
    return μ
end

"""
"""
function chebyshevpoints(n)
    grid = allocate_float(n)
    nfac = 1/(n-1)
    @inbounds begin
        # calculate z = cos(θ) ∈ [1,-1]
        for j ∈ 1:n
            grid[j] = cospi((j-1)*nfac)
        end
    end
    return grid
end

end
