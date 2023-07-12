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
export scaled_gauss_legendre_lobatto_grid
export scaled_gauss_legendre_radau_grid
export gausslegendre_derivative!
export setup_gausslegendre_pseudospectral

using FastGaussQuadrature
using LegendrePolynomials: Pl
using LinearAlgebra: mul!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float


"""
structs for passing around matrices for taking
the derivatives on Gauss-Legendre points in 1D
"""
struct gausslegendre_base_info{}
    # elementwise differentiation matrix (ngrid*ngrid)
    Dmat::Array{mk_float,2}
end

struct gausslegendre_info{}
    lobatto::gausslegendre_base_info
    radau::gausslegendre_base_info
end

function setup_gausslegendre_pseudospectral(coord)
    lobatto = setup_gausslegendre_pseudospectral_lobatto(coord)
    radau = setup_gausslegendre_pseudospectral_radau(coord)
    return gausslegendre_info(lobatto,radau)
end

function setup_gausslegendre_pseudospectral_lobatto(coord)
    x, w = gausslobatto(coord.ngrid)
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    gausslobattolegendre_differentiation_matrix!(Dmat,x,coord.ngrid,coord.L,coord.nelement_global)
    return gausslegendre_base_info(Dmat)
end

function setup_gausslegendre_pseudospectral_radau(coord)
    x, w = gaussradau(coord.ngrid)
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    gaussradaulegendre_differentiation_matrix!(Dmat,x,coord.ngrid,coord.L,coord.nelement_global)
    return gausslegendre_base_info(Dmat)
end 
"""
function for taking the first derivative on Gauss-Legendre points
"""
function gausslegendre_derivative!(df, ff, gausslegendre, coord)
    # define local variable nelement for convenience
    nelement = coord.nelement_local
    # check array bounds
    @boundscheck nelement == size(df,2) && coord.ngrid == size(df,1) || throw(BoundsError(df))
    
    # variable k will be used to avoid double counting of overlapping point
    k = 0
    j = 1 # the first element
    imin = coord.imin[j]-k
    # imax is the maximum index on the full grid for this (jth) element
    imax = coord.imax[j]        
    if coord.name == "vperp" && coord.irank == 0 # differentiate this element with the Radau scheme
        @views mul!(df[:,j],gausslegendre.radau.Dmat[:,:],ff[imin:imax])
    else #differentiate using the Lobatto scheme
        @views mul!(df[:,j],gausslegendre.lobatto.Dmat[:,:],ff[imin:imax])
    end
    # calculate the derivative on each element
    @inbounds for j ∈ 2:nelement
        k = 1 
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        @views mul!(df[:,j],gausslegendre.lobatto.Dmat[:,:],ff[imin:imax])
    end

    return nothing
end

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
    
    # get into correct order for a grid on (-1,1]
    Dreverse = copy(D)
    for ix in 1:ngrid
        for ixp in 1:ngrid
            Dreverse[ngrid-ix+1,ngrid-ixp+1] = -D[ix,ixp]
        end
    end
    D .= Dreverse
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

"""
function for setting up the full Gauss-Legendre-Lobatto
grid and collocation point weights
"""
function scaled_gauss_legendre_lobatto_grid(ngrid, nelement_global, nelement_local,
 n, irank, box_length, imin, imax)
    # get Gauss-Legendre-Lobatto points and weights on [-1,1]
    x, w = gausslobatto(ngrid)
    # factor with maps [-1,1] -> [-L/2, L/2]
    scale_factor = 0.5*box_length/float(nelement_global)
    # grid and weights arrays
    grid = allocate_float(n)
    wgts = allocate_float(n)
    wgts .= 0.0
    #integer to deal with the overlap of element boundaries
    k = 1
    @inbounds for j in 1:nelement_local
        # calculate the grid avoiding overlap
        iel_global = j + irank*nelement_local
        shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
        @. grid[imin[j]:imax[j]] = scale_factor*x[k:ngrid] + shift
        
        # calculate the weights
        # remembering on boundary points to include weights
        # from both left and right elements
        #println(imin[j]," ",imax[j])
        @. wgts[imin[j] - k + 1:imax[j]] += scale_factor*w[1:ngrid] 
        
        k = 2        
    end
    return grid, wgts
end

"""
function for setting up the full Gauss-Legendre-Radau
grid and collocation point weights
see comments of Gauss-Legendre-Lobatto routine above
"""
function scaled_gauss_legendre_radau_grid(ngrid, nelement_global, nelement_local,
 n, irank, box_length, imin, imax)
    # get Gauss-Legendre-Lobatto points and weights on [-1,1]
    x_lob, w_lob = gausslobatto(ngrid)
    # get Gauss-Legendre-Radau points and weights on [-1,1)
    x_rad, w_rad = gaussradau(ngrid)
    # transform to a Gauss-Legendre-Radau grid on (-1,1]
    x_rad, w_rad = -reverse(x_rad), reverse(w_rad)#
    
    # factor with maps [-1,1] -> [-L/2, L/2]
    scale_factor = 0.5*box_length/float(nelement_global)
    # grid and weights arrays
    grid = allocate_float(n)
    wgts = allocate_float(n)
    wgts .= 0.0
    if irank == 0
        # for 1st element, fill in with Gauss-Legendre-Radau points
        j = 1
        iel_global = j + irank*nelement_local
        shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
        @. grid[imin[j]:imax[j]] = scale_factor*x_rad[1:ngrid] + shift
        @. wgts[imin[j]:imax[j]] += scale_factor*w_rad[1:ngrid]         
        
        #integer to deal with the overlap of element boundaries
        k = 2
        @inbounds for j in 2:nelement_local
            # calculate the grid avoiding overlap
            iel_global = j + irank*nelement_local
            shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
            @. grid[imin[j]:imax[j]] = scale_factor*x_lob[k:ngrid] + shift
            @. wgts[imin[j] - k + 1:imax[j]] += scale_factor*w_lob[1:ngrid]         
        end
    else # all elements are Gauss-Legendre-Lobatto
        #integer to deal with the overlap of element boundaries
        k = 1
        @inbounds for j in 1:nelement_local
            # calculate the grid avoiding overlap
            iel_global = j + irank*nelement_local
            shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
            @. grid[imin[j]:imax[j]] = scale_factor*x_lob[k:ngrid] + shift
            @. wgts[imin[j] - k + 1:imax[j]] += scale_factor*w_lob[1:ngrid]
            
            k = 2 
        end
    end
    return grid, wgts
end

end
