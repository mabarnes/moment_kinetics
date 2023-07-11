"""
module for Gauss-Legendre-Lobatto and Gauss-Legendre-Radau spectral element grids
"""
module gausslegendre

export gausslobattolegendre_differentiation_matrix!
export gaussradaulegendre_differentiation_matrix!
export GaussLegendreLobatto_mass_matrix!
export GaussLegendreLobatto_inverse_mass_matrix!
export GaussLegendreLobatto_K_matrix!
export GaussLegendreLobatto_S_matrix!

using FastGaussQuadrature
using LegendrePolynomials: Pl
using LinearAlgebra: mul!

"""
Formula for differentiation matrix taken from p196 of Chpt `The Spectral Elemtent Method' of 
`Computational Seismology'. Heiner Igel First Edition. Published in 2017 by Oxford University Press.
Or https://doc.nektar.info/tutorials/latest/fundamentals/differentiation/fundamentals-differentiationch2.html

D -- differentiation matrix 
x -- Gauss-Legendre-Lobatto points in [-1,1]
ngrid -- number of points per element (incl. boundary points)
L -- total length of full domain
nelement -- total number of elements
"""
function gausslobattolegendre_differentiation_matrix!(D::Array{Float64,2},x::Array{Float64,1},ngrid::Int64,L::Float64,nelement::Int64)
    D[:,:] .= 0.0
    for ix in 1:ngrid
        for ixp in 1:ngrid
            if !(ix == ixp)
                D[ix,ixp] = (Pl(x[ix],ngrid-1)/Pl(x[ixp],ngrid-1))/(x[ix]-x[ixp])
            end
        end
    end
    # uncomment for analytical diagonal values 
    #D[1,1] = -0.25*(ngrid - 1)*ngrid
    #D[ngrid,ngrid] = 0.25*(ngrid - 1)*ngrid
    #for ix in 1:ngrid-1
    #   D[ix,ix] = 0.0
    #end
    # get diagonal values from sum of nonzero off diagonal values 
    for ix in 1:ngrid
        D[ix,ix] = -sum(D[ix,:])
    end
    #multiply by scale factor for element length
    D .= (2.0*float(nelement)/L).*D
    return nothing
end
"""
From 
https://doc.nektar.info/tutorials/latest/fundamentals/differentiation/fundamentals-differentiationch2.html

D -- differentiation matrix 
x -- Gauss-Legendre-Radau points in [-1,1)
ngrid -- number of points per element (incl. boundary points)
L -- total length of full domain
nelement -- total number of elements
"""
function gaussradaulegendre_differentiation_matrix!(D::Array{Float64,2},x::Array{Float64,1},ngrid::Int64,L::Float64,nelement::Int64)
    D[:,:] .= 0.0
    for ix in 1:ngrid
        for ixp in 1:ngrid
            if !(ix == ixp)
                D[ix,ixp] = (Pl(x[ix],ngrid-1)/Pl(x[ixp],ngrid-1))*((1.0 - x[ixp])/(1.0 - x[ix]))/(x[ix]-x[ixp])
            end
        end
    end
    # uncomment for analytical diagonal values 
    #D[1,1] = -0.25*(ngrid - 1)*(ngrid + 1)
    #for ix in 2:ngrid
    #   D[ix,ix] = 0.5/(1.0 - x[ix])
    #end
    # get diagonal values from sum of nonzero off diagonal values 
    for ix in 1:ngrid
        D[ix,ix] = -sum(D[ix,:])
    end
    #multiply by scale factor for element length
    D .= (2.0*float(nelement)/L).*D
    return nothing
end

"""
result of the inner product of Legendre polys of order k
"""
function Legendre_h_n(k)
    h_n = 2.0/(2.0*k + 1)
    return h_n
end 
"""
difference prefac between Gauss-Legendre 
and Gauss-Legendre-Lobatto points for the mass matrix 
"""
function alpha_n(N)
    gamma_n = 2.0/N
    h_n = Legendre_h_n(N)
    alpha = (h_n - gamma_n)/(gamma_n^2)
    return alpha
end

function beta_n(N)
    gamma_n = 2.0/N
    h_n = Legendre_h_n(N)
    beta = (gamma_n - h_n)/(gamma_n*h_n)
    return beta
end

"""
assign Gauss-Legendre-Lobatto mass matrix on a 1D line with Jacobian = 1
"""
function GaussLegendreLobatto_mass_matrix!(MM,ngrid,x,wgts,L)
    N = ngrid - 1
    alpha = alpha_n(N)
    MM .= 0.0
    ## off diagonal components
    for i in 1:ngrid
        for j in 1:ngrid
            MM[i,j] = alpha*wgts[i]*wgts[j]*Pl(x[i],N)*Pl(x[j],N)
        end
    end
    ## diagonal components
    for i in 1:ngrid 
        MM[i,i] += wgts[i]
    end
    @. MM *= (L/2.0)
    return nothing
end
"""
exact inverse of Gauss-Legendre-Lobatto mass matrix for testing
"""
function GaussLegendreLobatto_inverse_mass_matrix!(MM,ngrid,x,wgts,L)
    N = ngrid - 1
    beta = beta_n(N)
    MM .= 0.0
    ## off diagonal components
    for i in 1:ngrid
        for j in 1:ngrid
            MM[i,j] = beta*Pl(x[i],N)*Pl(x[j],N)
        end
    end
    ## diagonal components
    for i in 1:ngrid 
        MM[i,i] += 1.0/wgts[i]
    end
    @. MM *= 1.0/(L/2.0)
    return nothing
end
"""
Gauss-Legendre-Lobatto S matrix Sjk = < lj | l'k > 
Use that Djk = l'k(xj)
"""
function GaussLegendreLobatto_S_matrix!(SS,ngrid,DD,wgts,L)
    N = ngrid - 1
    SS .= 0.0
    for j in 1:ngrid 
        for i in 1:ngrid 
            SS[i,j] += (L/2.0)*wgts[i]*DD[i,j]
        end
    end
    return nothing
end
"""
Gauss-Legendre-Lobatto K matrix Kjk = -< l'j | l'k > 
Use that Djk = l'k(xj)
"""
function GaussLegendreLobatto_K_matrix!(KK,ngrid,DD,wgts,L)
    N = ngrid - 1
    KK .= 0.0
    for j in 1:ngrid 
        for i in 1:ngrid 
            for m in 1:ngrid
                KK[i,j] -= (L/2.0)*wgts[m]*DD[m,i]*DD[m,j]
            end
        end
    end
    return nothing
end

end
