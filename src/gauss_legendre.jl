"""
module for Gauss-Legendre-Lobatto and Gauss-Legendre-Radau spectral element grids
"""
module gauss_legendre

export gausslobattolegendre_differentiation_matrix!
export gaussradaulegendre_differentiation_matrix!
export GaussLegendreLobatto_mass_matrix!
export GaussLegendre_mass_matrix_1!
export GaussLegendreLobatto_inverse_mass_matrix!
export GaussLegendreLobatto_K_matrix!
export GaussLegendreLobatto_S_matrix!
export GaussLegendre_S_matrix_1!
export scaled_gauss_legendre_lobatto_grid
export scaled_gauss_legendre_radau_grid
export gausslegendre_derivative!
export gausslegendre_apply_Kmat!
export gausslegendre_apply_Lmat!
export gausslegendre_mass_matrix_solve!
export setup_gausslegendre_pseudospectral
export GaussLegendre_weak_product_matrix!
export ielement_global_func
export get_QQ_local!

using FastGaussQuadrature
using LegendrePolynomials: Pl, dnPl
using LinearAlgebra: mul!, lu, LU
using SparseArrays: sparse, AbstractSparseArray
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float


"""
structs for passing around matrices for taking
the derivatives on Gauss-Legendre points in 1D
"""
struct gausslegendre_base_info{}
    # elementwise differentiation matrix (ngrid*ngrid)
    Dmat::Array{mk_float,2}
    # local mass matrix 
    Mmat::Array{mk_float,2}
    # local K matrix (for second derivatives)
    Kmat::Array{mk_float,2}
    # local mass matrix type 0
    M0::Array{mk_float,2}
    # local mass matrix type 1
    M1::Array{mk_float,2}
    # local mass matrix type 2
    M2::Array{mk_float,2}
    # local S (weak derivative) matrix type 0
    S0::Array{mk_float,2}
    # local S (weak derivative) matrix type 1
    S1::Array{mk_float,2}
    # local K (weak second derivative) matrix type 0
    K0::Array{mk_float,2}
    # local K (weak second derivative) matrix type 1
    K1::Array{mk_float,2}
    # local K (weak second derivative) matrix type 2
    K2::Array{mk_float,2}
    # local P (weak derivative no integration by parts) matrix type 0
    P0::Array{mk_float,2}
    # local P (weak derivative no integration by parts) matrix type 1
    P1::Array{mk_float,2}
    # local P (weak derivative no integration by parts) matrix type 2
    P2::Array{mk_float,2}
    # boundary condition differentiation matrix (for vperp grid using radau points)
    D0::Array{mk_float,1}
end

struct gausslegendre_info{}
    lobatto::gausslegendre_base_info
    radau::gausslegendre_base_info
    # global (1D) mass matrix
    mass_matrix::Array{mk_float,2}
    # global (1D) weak derivative matrix
    #S_matrix::Array{mk_float,2}
    S_matrix::AbstractSparseArray{mk_float,Ti,2} where Ti
    # global (1D) weak second derivative matrix
    K_matrix::Array{mk_float,2}
    # global (1D) weak Laplacian derivative matrix
    L_matrix::Array{mk_float,2}
    # global (1D) LU object
    mass_matrix_lu::T where T
    # dummy matrix for local operators
    Qmat::Array{mk_float,2}
end

function setup_gausslegendre_pseudospectral(coord)
    lobatto = setup_gausslegendre_pseudospectral_lobatto(coord)
    radau = setup_gausslegendre_pseudospectral_radau(coord)
    mass_matrix = allocate_float(coord.n,coord.n)
    S_matrix = allocate_float(coord.n,coord.n)
    K_matrix = allocate_float(coord.n,coord.n)
    L_matrix = allocate_float(coord.n,coord.n)
    setup_global_mass_matrix!(mass_matrix, lobatto, radau, coord)
    
    setup_global_weak_form_matrix!(mass_matrix, lobatto, radau, coord, "M")
    setup_global_weak_form_matrix!(S_matrix, lobatto, radau, coord, "S")
    setup_global_weak_form_matrix!(K_matrix, lobatto, radau, coord, "K")
    setup_global_weak_form_matrix!(L_matrix, lobatto, radau, coord, "L")
    mass_matrix_lu = lu(sparse(mass_matrix))
    Qmat = allocate_float(coord.ngrid,coord.ngrid)
    return gausslegendre_info(lobatto,radau,mass_matrix,sparse(S_matrix),K_matrix,L_matrix,mass_matrix_lu,Qmat)
end

function setup_gausslegendre_pseudospectral_lobatto(coord)
    x, w = gausslobatto(coord.ngrid)
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    gausslobattolegendre_differentiation_matrix!(Dmat,x,coord.ngrid,coord.L,coord.nelement_global)
    Mmat = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendreLobatto_mass_matrix!(Mmat,coord.ngrid,x,w,coord.L,coord.nelement_global)
    Kmat = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendreLobatto_K_matrix!(Kmat,coord.ngrid,Dmat,w,coord.L,coord.nelement_global)
    
    M0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(M0,coord.ngrid,x,w,coord.L,coord.nelement_global,"M0")
    M1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(M1,coord.ngrid,x,w,coord.L,coord.nelement_global,"M1")
    M2 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(M2,coord.ngrid,x,w,coord.L,coord.nelement_global,"M2")
    S0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(S0,coord.ngrid,x,w,coord.L,coord.nelement_global,"S0")
    S1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(S1,coord.ngrid,x,w,coord.L,coord.nelement_global,"S1")
    K0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(K0,coord.ngrid,x,w,coord.L,coord.nelement_global,"K0")
    K1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(K1,coord.ngrid,x,w,coord.L,coord.nelement_global,"K1")
    K2 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(K2,coord.ngrid,x,w,coord.L,coord.nelement_global,"K2")
    P0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(P0,coord.ngrid,x,w,coord.L,coord.nelement_global,"P0")
    P1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(P1,coord.ngrid,x,w,coord.L,coord.nelement_global,"P1")
    P2 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(P2,coord.ngrid,x,w,coord.L,coord.nelement_global,"P2")
    D0 = allocate_float(coord.ngrid)
    #@. D0 = Dmat[1,:] # values at lower extreme of element
    GaussLegendre_derivative_vector!(D0,-1.0,coord.ngrid,x,w,coord.L,coord.nelement_global)
    return gausslegendre_base_info(Dmat,Mmat,Kmat,M0,M1,M2,S0,S1,K0,K1,K2,P0,P1,P2,D0)
end

function setup_gausslegendre_pseudospectral_radau(coord)
    # Gauss-Radau points on [-1,1)
    x, w = gaussradau(coord.ngrid)
    # Gauss-Radau points on (-1,1] 
    xreverse, wreverse = -reverse(x), reverse(w)
    # elemental differentiation matrix
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    gaussradaulegendre_differentiation_matrix!(Dmat,x,coord.ngrid,coord.L,coord.nelement_global)
    # elemental mass matrix
    Mmat = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendreLobatto_mass_matrix!(Mmat,coord.ngrid,x,w,coord.L,coord.nelement_global)
    Kmat = allocate_float(coord.ngrid, coord.ngrid)
    
    M0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(M0,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"M0",radau=true)
    M1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(M1,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"M1",radau=true)
    M2 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(M2,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"M2",radau=true)
    S0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(S0,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"S0",radau=true)
    S1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(S1,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"S1",radau=true)
    K0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(K0,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"K0",radau=true)
    K1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(K1,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"K1",radau=true)
    K2 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(K2,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"K2",radau=true)
    P0 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(P0,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"P0",radau=true)
    P1 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(P1,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"P1",radau=true)
    P2 = allocate_float(coord.ngrid, coord.ngrid)
    GaussLegendre_weak_product_matrix!(P2,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,"P2",radau=true)
    D0 = allocate_float(coord.ngrid)
    GaussLegendre_derivative_vector!(D0,-1.0,coord.ngrid,xreverse,wreverse,coord.L,coord.nelement_global,radau=true)
    return gausslegendre_base_info(Dmat,Mmat,Kmat,M0,M1,M2,S0,S1,K0,K1,K2,P0,P1,P2,D0)
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
function for taking the weak-form second derivative on Gauss-Legendre points
"""
function gausslegendre_apply_Kmat!(df, ff, gausslegendre, coord)
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
    get_KK_local!(gausslegendre.Qmat,j,gausslegendre.lobatto,gausslegendre.radau,coord)
    #println(gausslegendre.Qmat)
    @views mul!(df[:,j],gausslegendre.Qmat[:,:],ff[imin:imax])
    zero_gradient_bc_lower_boundary = false#true
    if coord.name == "vperp" && zero_gradient_bc_lower_boundary
       # set the 1st point of the RHS vector to zero 
       # consistent with use with the mass matrix with D f = 0 boundary conditions
       df[1,j] = 0.0
    end
    # calculate the derivative on each element
    @inbounds for j ∈ 2:nelement
        k = 1 
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        #@views mul!(df[:,j],gausslegendre.lobatto.Kmat[:,:],ff[imin:imax])
        get_KK_local!(gausslegendre.Qmat,j,gausslegendre.lobatto,gausslegendre.radau,coord)
        #println(gausslegendre.Qmat)
        @views mul!(df[:,j],gausslegendre.Qmat[:,:],ff[imin:imax])
    end
    #for j in 1:nelement
    #    println(df[:,j])
    #end
    return nothing
end

"""
function for taking the weak-form Laplacian derivative on Gauss-Legendre points
"""
function gausslegendre_apply_Lmat!(df, ff, gausslegendre, coord)
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
    get_LL_local!(gausslegendre.Qmat,j,gausslegendre.lobatto,gausslegendre.radau,coord)
    #println(gausslegendre.Qmat)
    @views mul!(df[:,j],gausslegendre.Qmat[:,:],ff[imin:imax])
    zero_gradient_bc_lower_boundary = false#true
    boundary_flux_terms = true#false
    if coord.name == "vperp" && zero_gradient_bc_lower_boundary
       # set the 1st point of the RHS vector to zero 
       # consistent with use with the mass matrix with D f = 0 boundary conditions
       df[1,j] = 0.0
    end
    # boundary flux terms
    if boundary_flux_terms
        if coord.name=="vperp" # only include a flux term from the upper boundary
            @. coord.scratch[imin:imax] = gausslegendre.radau.Dmat[coord.ngrid,:]*ff[imin:imax]
            df[coord.ngrid,j] += coord.jacobian[imax]*sum(coord.scratch[imin:imax])
        else
            @. coord.scratch[imin:imax] = gausslegendre.lobatto.Dmat[1,:]*ff[imin:imax]
            df[1,j] -= coord.jacobian[imin]*sum(coord.scratch[imin:imax])
            @. coord.scratch[imin:imax] = gausslegendre.lobatto.Dmat[coord.ngrid,:]*ff[imin:imax]
            df[coord.ngrid,j] += coord.jacobian[imax]*sum(coord.scratch[imin:imax])
        end
    end
    # calculate the derivative on each element
    @inbounds for j ∈ 2:nelement
        k = 1 
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        #@views mul!(df[:,j],gausslegendre.lobatto.Kmat[:,:],ff[imin:imax])
        get_LL_local!(gausslegendre.Qmat,j,gausslegendre.lobatto,gausslegendre.radau,coord)
        #println(gausslegendre.Qmat)
        @views mul!(df[:,j],gausslegendre.Qmat[:,:],ff[imin:imax])
        # boundary flux terms 
        if boundary_flux_terms
            @. coord.scratch[imin:imax] = gausslegendre.lobatto.Dmat[1,:]*ff[imin:imax]
            df[1,j] -= coord.jacobian[imin]*sum(coord.scratch[imin:imax])
            @. coord.scratch[imin:imax] = gausslegendre.lobatto.Dmat[coord.ngrid,:]*ff[imin:imax]
            df[coord.ngrid,j] += coord.jacobian[imax]*sum(coord.scratch[imin:imax])
        end
    end
    #for j in 1:nelement
    #    println(df[:,j])
    #end
    return nothing
end

function gausslegendre_mass_matrix_solve!(f,b,spectral)
    y = spectral.mass_matrix_lu \ b
    @. f = y
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
Gauss-Legendre derivative at arbitrary x values, for boundary condition on radau points
D0 -- the vector
xj -- the x location where the derivative is evaluated 
ngrid -- number of points in x
x -- the grid from -1, 1
L -- size of physical domain
"""
function GaussLegendre_derivative_vector!(D0,xj,ngrid,x,wgts,L,nelement_global;radau=false)
    # coefficient in expansion of 
    # lagrange polys in terms of Legendre polys
    gamma = allocate_float(ngrid)
    for i in 1:ngrid-1
        gamma[i] = Legendre_h_n(i-1)
    end
    if radau
        gamma[ngrid] = Legendre_h_n(ngrid-1)
    else
        gamma[ngrid] = 2.0/(ngrid - 1)
    end
    
    @. D0 = 0.0
    for i in 1:ngrid
        for k in 1:ngrid
            D0[i] += wgts[i]*Pl(x[i],k-1)*dnPl(xj,k-1,1)/gamma[k]
        end
    end
    # set `diagonal' value
    D0[1] = 0.0
    D0[1] = -sum(D0[:])
    @. D0 *= 2.0*float(nelement_global)/L
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
function GaussLegendreLobatto_mass_matrix!(MM,ngrid,x,wgts,L,nelement_global)
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
    @. MM *= (0.5*L/nelement_global)
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
function GaussLegendreLobatto_K_matrix!(KK,ngrid,DD,wgts,L,nelement_global)
    N = ngrid - 1
    KK .= 0.0
    for j in 1:ngrid 
        for i in 1:ngrid 
            for m in 1:ngrid
                KK[i,j] -= (0.5*L/nelement_global)*wgts[m]*DD[m,i]*DD[m,j]
            end
        end
    end
    return nothing
end

"""
assign abitrary weak inner product matrix Q on a 1D line with Jacobian = 1
"""
function GaussLegendre_weak_product_matrix!(QQ,ngrid,x,wgts,L,nelement_global,option;radau=false)
    # coefficient in expansion of 
    # lagrange polys in terms of Legendre polys
    gamma = allocate_float(ngrid)
    for i in 1:ngrid-1
        gamma[i] = Legendre_h_n(i-1)
    end
    if radau
        gamma[ngrid] = Legendre_h_n(ngrid-1)
    else
        gamma[ngrid] = 2.0/(ngrid - 1)
    end
    # appropriate inner product of Legendre polys
    # definition depends on required matrix 
    # for M0: AA = < P_i P_j >
    # for M1: AA = < P_i P_j x >
    # for M2: AA = < P_i P_j x^2 >
    # for S0: AA = -< P'_i P_j >
    # for S1: AA = -< P'_i P_j x >
    # for K0: AA = -< P'_i P'_j >
    # for K1: AA = -< P'_i P'_j x >
    # for K2: AA = -< P'_i P'_j x^2 >
    # for P0: AA = < P_i P'_j >
    # for P1: AA = < P_i P'_j x >
    # for P2: AA = < P_i P'_j x^2 >
    AA = allocate_float(ngrid,ngrid)
    nquad = 2*ngrid
    zz, wz = gausslegendre(nquad)
    @. AA = 0.0
    if option == "M0"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] += wz[k]*Pl(zz[k],i-1)*Pl(zz[k],j-1)
                end
            end
        end
    elseif option == "M1"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] += zz[k]*wz[k]*Pl(zz[k],i-1)*Pl(zz[k],j-1)
                end
            end
        end
    elseif option == "M2"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] += (zz[k]^2)*wz[k]*Pl(zz[k],i-1)*Pl(zz[k],j-1)
                end
            end
        end
    elseif option == "S0"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] -= wz[k]*dnPl(zz[k],i-1,1)*Pl(zz[k],j-1)
                end
            end
        end
    elseif option == "S1"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] -= zz[k]*wz[k]*dnPl(zz[k],i-1,1)*Pl(zz[k],j-1)
                end
            end
        end
    elseif option == "K0"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] -= wz[k]*dnPl(zz[k],i-1,1)*dnPl(zz[k],j-1,1)
                end
            end
        end
    elseif option == "K1"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] -= zz[k]*wz[k]*dnPl(zz[k],i-1,1)*dnPl(zz[k],j-1,1)
                end
            end
        end
    elseif option == "K2"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] -= (zz[k]^2)*wz[k]*dnPl(zz[k],i-1,1)*dnPl(zz[k],j-1,1)
                end
            end
        end
    elseif option == "P0"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] += wz[k]*Pl(zz[k],i-1)*dnPl(zz[k],j-1,1)
                end
            end
        end
    elseif option == "P1"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] += zz[k]*wz[k]*Pl(zz[k],i-1)*dnPl(zz[k],j-1,1)
                end
            end
        end
    elseif option == "P2"
        for j in 1:ngrid
            for i in 1:ngrid
                for k in 1:nquad
                    AA[i,j] += (zz[k]^2)*wz[k]*Pl(zz[k],i-1)*dnPl(zz[k],j-1,1)
                end
            end
        end
    end
    
    QQ .= 0.0
    for j in 1:ngrid
        for i in 1:ngrid
            for l in 1:ngrid
                for k in 1:ngrid
                    QQ[i,j] += wgts[i]*wgts[j]*Pl(x[i],k-1)*Pl(x[j],l-1)*AA[k,l]/(gamma[k]*gamma[l])
                end
            end
        end
    end
    #if option == "K0" || option == "K1" || option == "S0" || option == "P0"
        # compute diagonal from off-diagonal values 
        # to ensure numerical stability 
    #    for i in 1:ngrid
    #        QQ[i,i] = 0.0
    #        QQ[i,i] = -sum(QQ[i,:])
    #    end
    #end
    # return normalised Q (no scale factors)
    #@. QQ *= (0.5*L/nelement_global)
    return nothing
end

"""
assign mass matrix M1mn = < lm|x|ln > on a 1D line with Jacobian = 1
"""
function GaussLegendre_mass_matrix_1!(MM,ngrid,x,wgts,L,nelement_global)
    # coefficient in expansion of 
    # lagrange polys in terms of Legendre polys
    gamma = allocate_float(ngrid)
    for i in 1:ngrid-1
        gamma[i] = Legendre_h_n(i-1)
    end
    gamma[ngrid] = 2.0/(ngrid - 1)
    # appropriate inner product of Legendre polys
    # < P_i P_j x >
    AA = allocate_float(ngrid,ngrid)
    nquad = 2*ngrid
    zz, wz = gausslegendre(nquad)
    @. AA = 0.0
    for j in 1:ngrid
        for i in 1:ngrid
            for k in 1:nquad
                AA[i,j] += zz[k]*wz[k]*Pl(zz[k],i-1)*Pl(zz[k],j-1)
            end
        end
    end
    
    MM .= 0.0
    for i in 1:ngrid
        for j in 1:ngrid
            for k in 1:ngrid
                for l in 1:ngrid
                    MM[i,j] += wgts[i]*wgts[j]*Pl(x[i],k-1)*Pl(x[j],l-1)*AA[k,l]/(gamma[k]*gamma[l])
                end
            end
        end
    end
    @. MM *= (0.5*L/nelement_global)
    return nothing
end

"""
assign derivative matrix S1mn = < l'm|x|ln > on a 1D line with Jacobian = 1
"""
function GaussLegendre_S_matrix_1!(SS,ngrid,x,wgts,L,nelement_global)
    # coefficient in expansion of 
    # lagrange polys in terms of Legendre polys
    gamma = allocate_float(ngrid)
    for i in 1:ngrid-1
        gamma[i] = Legendre_h_n(i-1)
    end
    gamma[ngrid] = 2.0/(ngrid - 1)
    # appropriate inner product of Legendre polys
    # < P'_i P_j x >
    AA = allocate_float(ngrid,ngrid)
    nquad = 2*ngrid
    zz, wz = gausslegendre(nquad)
    @. AA = 0.0
    for j in 1:ngrid
        for i in 1:ngrid
            for k in 1:nquad
                AA[i,j] += zz[k]*wz[k]*dnPl(zz[k],i-1,1)*Pl(zz[k],j-1)
            end
        end
    end
    
    SS .= 0.0
    for i in 1:ngrid
        for j in 1:ngrid
            for k in 1:ngrid
                for l in 1:ngrid
                    SS[i,j] -= wgts[i]*wgts[j]*Pl(x[i],k-1)*Pl(x[j],l-1)*AA[k,l]/(gamma[k]*gamma[l])
                end
            end
        end
    end
    @. SS *= (0.5*L/nelement_global)
    return nothing
end

function scale_factor_func(L,nelement_global)
    return 0.5*L/float(nelement_global)
end

function shift_factor_func(L,nelement_global,nelement_local,irank,ielement_local)
    #ielement_global = ielement_local # for testing + irank*nelement_local
    ielement_global = ielement_local + irank*nelement_local # proper line for future distributed memory MPI use
    shift = L*((float(ielement_global)-0.5)/float(nelement_global) - 0.5)
    return shift
end

function ielement_global_func(nelement_local,irank,ielement_local)
    return ielement_global = ielement_local + irank*nelement_local
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
    scale_factor = scale_factor_func(box_length,nelement_global) #0.5*box_length/float(nelement_global)
    # grid and weights arrays
    grid = allocate_float(n)
    wgts = allocate_float(n)
    wgts .= 0.0
    #integer to deal with the overlap of element boundaries
    k = 1
    @inbounds for j in 1:nelement_local
        # calculate the grid avoiding overlap
        #iel_global = j + irank*nelement_local
        #shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
        shift = shift_factor_func(box_length,nelement_global,nelement_local,irank,j)
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
    scale_factor = scale_factor_func(box_length,nelement_global)
    #scale_factor = 0.5*box_length/float(nelement_global)
    # grid and weights arrays
    grid = allocate_float(n)
    wgts = allocate_float(n)
    wgts .= 0.0
    if irank == 0
        # for 1st element, fill in with Gauss-Legendre-Radau points
        j = 1
        #iel_global = j + irank*nelement_local
        #shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
        shift = shift_factor_func(box_length,nelement_global,nelement_local,irank,j)
        @. grid[imin[j]:imax[j]] = scale_factor*x_rad[1:ngrid] + shift
        @. wgts[imin[j]:imax[j]] += scale_factor*w_rad[1:ngrid]         
        
        #integer to deal with the overlap of element boundaries
        k = 2
        @inbounds for j in 2:nelement_local
            # calculate the grid avoiding overlap
            #iel_global = j + irank*nelement_local
            #shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
            shift = shift_factor_func(box_length,nelement_global,nelement_local,irank,j)
            @. grid[imin[j]:imax[j]] = scale_factor*x_lob[k:ngrid] + shift
            @. wgts[imin[j] - k + 1:imax[j]] += scale_factor*w_lob[1:ngrid]         
        end
    else # all elements are Gauss-Legendre-Lobatto
        #integer to deal with the overlap of element boundaries
        k = 1
        @inbounds for j in 1:nelement_local
            # calculate the grid avoiding overlap
            #iel_global = j + irank*nelement_local
            #shift = box_length*((float(iel_global)-0.5)/float(nelement_global) - 0.5)
            shift = shift_factor_func(box_length,nelement_global,nelement_local,irank,j)
            @. grid[imin[j]:imax[j]] = scale_factor*x_lob[k:ngrid] + shift
            @. wgts[imin[j] - k + 1:imax[j]] += scale_factor*w_lob[1:ngrid]
            
            k = 2 
        end
    end
    return grid, wgts
end

"""
function that assigns the local mass matrices to 
a global array for later solving weak form of required
1D equation
"""
function setup_global_mass_matrix!(mass_matrix::Array{mk_float,2},
                               lobatto::gausslegendre_base_info,
                               radau::gausslegendre_base_info, 
                               coord)
    ngrid = coord.ngrid
    imin = coord.imin
    imax = coord.imax
    @. mass_matrix = 0.0
    if coord.name == "vperp"
        # use the Radau mass matrix for the 1st element
        Mmat_fel = radau.Mmat
    else
        # use the Lobatto mass matrix 
        Mmat_fel = lobatto.Mmat
    end
    zero_bc_upper_boundary = coord.bc == "zero" || coord.bc == "zero_upper"
    zero_bc_lower_boundary = coord.bc == "zero" || coord.bc == "zero_lower"
    
    # fill in first element 
    j = 1
    if zero_bc_lower_boundary #x.bc == "zero"
        mass_matrix[imin[j],imin[j]:imax[j]] .+= Mmat_fel[1,:]./2.0 #contributions from this element/2
        mass_matrix[imin[j],imin[j]] += Mmat_fel[ngrid,ngrid]/2.0 #contribution from missing `zero' element/2
    else 
        mass_matrix[imin[j],imin[j]:imax[j]] .+= Mmat_fel[1,:]
    end
    for k in 2:imax[j]-imin[j] 
        mass_matrix[k,imin[j]:imax[j]] .+= Mmat_fel[k,:]
    end
    if zero_bc_upper_boundary && coord.nelement_local == 1
        mass_matrix[imax[j],imin[j]:imax[j]] .+= Mmat_fel[ngrid,:]./2.0 #contributions from this element/2
        mass_matrix[imax[j],imax[j]] += lobatto.Mmat[1,1]/2.0              #contribution from missing `zero' element/2
    elseif coord.nelement_local > 1 #x.bc == "zero"
        mass_matrix[imax[j],imin[j]:imax[j]] .+= Mmat_fel[ngrid,:]./2.0
    else
        mass_matrix[imax[j],imin[j]:imax[j]] .+= Mmat_fel[ngrid,:]
    end 
    # remaining elements recalling definitions of imax and imin
    for j in 2:coord.nelement_local
        #lower boundary condition on element
        mass_matrix[imin[j]-1,imin[j]-1:imax[j]] .+= lobatto.Mmat[1,:]./2.0
        for k in 2:imax[j]-imin[j]+1 
            mass_matrix[k+imin[j]-2,imin[j]-1:imax[j]] .+= lobatto.Mmat[k,:]
        end
        # upper boundary condition on element 
        if j == coord.nelement_local && !(zero_bc_upper_boundary)
            mass_matrix[imax[j],imin[j]-1:imax[j]] .+= lobatto.Mmat[ngrid,:]
        elseif j == coord.nelement_local && zero_bc_upper_boundary
            mass_matrix[imax[j],imin[j]-1:imax[j]] .+= lobatto.Mmat[ngrid,:]./2.0 #contributions from this element/2
            mass_matrix[imax[j],imax[j]] += lobatto.Mmat[1,1]/2.0 #contribution from missing `zero' element/2
        else 
            mass_matrix[imax[j],imin[j]-1:imax[j]] .+= lobatto.Mmat[ngrid,:]./2.0
        end
    end
        
    return nothing
end

"""
function that assigns the local weak-form matrices to 
a global array QQ_global for later solving weak form of required
1D equation

option choosing type of matrix to be constructed -- "M" (mass matrix), "S" (derivative matrix)
"""
function setup_global_weak_form_matrix!(QQ_global::Array{mk_float,2},
                               lobatto::gausslegendre_base_info,
                               radau::gausslegendre_base_info, 
                               coord,option)
    QQ_j = allocate_float(coord.ngrid,coord.ngrid)
    QQ_jp1 = allocate_float(coord.ngrid,coord.ngrid)
    
    ngrid = coord.ngrid
    imin = coord.imin
    imax = coord.imax
    @. QQ_global = 0.0
    
    if coord.name == "vperp"
        zero_bc_upper_boundary = true
        zero_bc_lower_boundary = false
        zero_gradient_bc_lower_boundary = false#true
    else 
        zero_bc_upper_boundary = coord.bc == "zero" || coord.bc == "zero_upper"
        zero_bc_lower_boundary = coord.bc == "zero" || coord.bc == "zero_lower"
        zero_gradient_bc_lower_boundary = false
    end
    # fill in first element 
    j = 1
    # N.B. QQ varies with ielement for vperp, but not vpa
    get_QQ_local!(QQ_j,j,lobatto,radau,coord,option)
    get_QQ_local!(QQ_jp1,j+1,lobatto,radau,coord,option)
    
    if zero_bc_lower_boundary #x.bc == "zero"
        QQ_global[imin[j],imin[j]:imax[j]] .+= QQ_j[1,:]./2.0 #contributions from this element/2
        QQ_global[imin[j],imin[j]] += QQ_j[ngrid,ngrid]/2.0 #contribution from missing `zero' element/2
    elseif zero_gradient_bc_lower_boundary
        if option == "M" && coord.name == "vperp"
            #QQ_global[imin[j],imin[j]:imax[j]] .= lobatto.D0[:]
            QQ_global[imin[j],imin[j]:imax[j]] .= radau.D0[:]
        else 
            QQ_global[imin[j],imin[j]:imax[j]] .= 0.0
        end
    else 
        QQ_global[imin[j],imin[j]:imax[j]] .+= QQ_j[1,:]
    end
    for k in 2:imax[j]-imin[j] 
        QQ_global[k,imin[j]:imax[j]] .+= QQ_j[k,:]
    end
    if zero_bc_upper_boundary && coord.nelement_local == 1
        QQ_global[imax[j],imin[j]:imax[j]] .+= QQ_j[ngrid,:]./2.0 #contributions from this element/2
        QQ_global[imax[j],imax[j]] += QQ_jp1[1,1]/2.0              #contribution from missing `zero' element/2
    elseif coord.nelement_local > 1 #x.bc == "zero"
        QQ_global[imax[j],imin[j]:imax[j]] .+= QQ_j[ngrid,:]./2.0
    else
        QQ_global[imax[j],imin[j]:imax[j]] .+= QQ_j[ngrid,:]
    end 
    # remaining elements recalling definitions of imax and imin
    for j in 2:coord.nelement_local
        get_QQ_local!(QQ_j,j,lobatto,radau,coord,option)
        get_QQ_local!(QQ_jp1,j+1,lobatto,radau,coord,option)
    
        #lower boundary condition on element
        QQ_global[imin[j]-1,imin[j]-1:imax[j]] .+= QQ_j[1,:]./2.0
        for k in 2:imax[j]-imin[j]+1 
            QQ_global[k+imin[j]-2,imin[j]-1:imax[j]] .+= QQ_j[k,:]
        end
        # upper boundary condition on element 
        if j == coord.nelement_local && !(zero_bc_upper_boundary)
            QQ_global[imax[j],imin[j]-1:imax[j]] .+= QQ_j[ngrid,:]
        elseif j == coord.nelement_local && zero_bc_upper_boundary
            QQ_global[imax[j],imin[j]-1:imax[j]] .+= QQ_j[ngrid,:]./2.0 #contributions from this element/2
            QQ_global[imax[j],imax[j]] += QQ_jp1[1,1]/2.0 #contribution from missing `zero' element/2
        else 
            QQ_global[imax[j],imin[j]-1:imax[j]] .+= QQ_j[ngrid,:]./2.0
        end
    end
        
    return nothing
end

function get_QQ_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord,option)
  
        if option == "M"
            get_MM_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "R"
            get_MR_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "N"
            get_MN_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "P"
            get_PP_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "U"
            get_PU_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "S"
            get_SS_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "K"
            get_KK_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "J"
            get_KJ_local!(QQ,ielement,lobatto,radau,coord)
        elseif option == "L"
            get_LL_local!(QQ,ielement,lobatto,radau,coord)
        end
        return nothing
end

function get_MM_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  (shift_factor*lobatto.M0 + scale_factor*lobatto.M1)*scale_factor
            else # radau points 
                @. QQ =  (shift_factor*radau.M0 + scale_factor*radau.M1)*scale_factor
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.M0*scale_factor
        end 
        return nothing
end

function get_SS_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  shift_factor*lobatto.S0 + scale_factor*lobatto.S1
            else # radau points 
                @. QQ =  shift_factor*radau.S0 + scale_factor*radau.S1
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.S0
        end
        return nothing
end

function get_KK_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            # P0 factors make this a d^2 / dvperp^2 rather than (1/vperp) d ( vperp d (.) / d vperp)
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  (shift_factor/scale_factor)*lobatto.K0 + lobatto.K1 - lobatto.P0
            else # radau points 
                @. QQ =  (shift_factor/scale_factor)*radau.K0 + radau.K1 - radau.P0
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.K0/scale_factor
        end
        return nothing
end

# second derivative matrix with vperp^2 Jacobian factor if 
# coord is vperp. Not useful for the vpa coordinate
function get_KJ_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            # P0 factors make this a d^2 / dvperp^2 rather than (1/vperp) d ( vperp d (.) / d vperp)
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ = (lobatto.K0*((shift_factor^2)/scale_factor) +
                         lobatto.K1*shift_factor +
                         lobatto.K2*scale_factor)
            else # radau points 
                @. QQ =  (radau.K0*((shift_factor^2)/scale_factor) +
                         radau.K1*shift_factor +
                         radau.K2*scale_factor)
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.K0/scale_factor
        end
        return nothing
end

function get_LL_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            #  (1/vperp) d ( vperp d (.) / d vperp)
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  (shift_factor/scale_factor)*lobatto.K0 + lobatto.K1
            else # radau points 
                @. QQ =  (shift_factor/scale_factor)*radau.K0 + radau.K1
            end
        else # d^2 (.) d vpa^2 -- assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.K0/scale_factor
        end
        return nothing
end

# mass matrix without vperp factor (matrix N)
# only useful for the vperp coordinate
function get_MN_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        #shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  lobatto.M0*scale_factor
            else # radau points 
                @. QQ =  radau.M0*scale_factor
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.M0*scale_factor
        end 
        return nothing
end

# mass matrix with vperp^2 factor (matrix R)
# only useful for the vperp coordinate
function get_MR_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  (lobatto.M0*shift_factor^2 +
                          lobatto.M1*2.0*shift_factor*scale_factor +
                          lobatto.M2*scale_factor^2)*scale_factor
            else # radau points 
                @. QQ =  (radau.M0*shift_factor^2 +
                          radau.M1*2.0*shift_factor*scale_factor +
                          radau.M2*scale_factor^2)*scale_factor
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.M0*scale_factor
        end 
        return nothing
end

# derivative matrix (matrix P, no integration by parts)
# with vperp Jacobian factor if coord is vperp (matrix P)
function get_PP_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  lobatto.P0*shift_factor + lobatto.P1*scale_factor
            else # radau points 
                @. QQ =  radau.P0*shift_factor + radau.P1*scale_factor
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.P0
        end 
        return nothing
end

# derivative matrix (matrix P, no integration by parts)
# with vperp^2 Jacobian factor if coord is vperp (matrix U)
# not useful for vpa coordinate
function get_PU_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        coord)
        
        scale_factor = scale_factor_func(coord.L,coord.nelement_global)
        shift_factor = shift_factor_func(coord.L,coord.nelement_global,coord.nelement_local,coord.irank,ielement) + 0.5*coord.L
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            # extra scale and shift factors required because of vperp in integral
            if ielement > 1 || coord.irank > 0 # lobatto points
                @. QQ =  (lobatto.P0*shift_factor^2 + 
                          lobatto.P1*2.0*shift_factor*scale_factor +
                          lobatto.P2*scale_factor^2)
            else # radau points 
                @. QQ =  (radau.P0*shift_factor^2 + 
                          radau.P1*2.0*shift_factor*scale_factor +
                          radau.P2*scale_factor^2)
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            @. QQ = lobatto.P0
        end 
        return nothing
end


end
