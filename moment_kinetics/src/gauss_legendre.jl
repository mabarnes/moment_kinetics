"""
module for Gauss-Legendre-Lobatto and Gauss-Legendre-Radau spectral element grids
"""
module gauss_legendre

export gausslobattolegendre_differentiation_matrix!
export gaussradaulegendre_differentiation_matrix!
export scaled_gauss_legendre_lobatto_grid
export scaled_gauss_legendre_radau_grid
export setup_gausslegendre_pseudospectral
export integration_matrix!

using FastGaussQuadrature
using LegendrePolynomials: Pl, dnPl
using LinearAlgebra: mul!, lu, ldiv!
using SparseArrays: sparse, AbstractSparseArray
using SparseMatricesCSR
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float
import ..calculus: elementwise_derivative!, mass_matrix_solve!,
                   elementwise_indefinite_integration!
import ..interpolation: single_element_interpolate!,
                        fill_single_element_interpolation_matrix!
using ..lagrange_polynomials: lagrange_poly_optimised, lagrange_poly_derivative_optimised, lagrange_poly
using ..moment_kinetics_structs: weak_discretization_info
using FiniteElementMatrices: element_coordinates,
                            lagrange_x,
                            d_lagrange_dx,
                            finite_element_matrix

"""
A struct for passing around elemental matrices
on Gauss-Legendre points in 1D
"""
struct gausslegendre_base_info
    # flag for whether we are Radau or Lobatto
    is_lobatto::Bool
    # elementwise differentiation matrix (ngrid*ngrid)
    Dmat::Array{mk_float,2}
    # elementwise integration matrix (ngrid*ngrid)
    indefinite_integration_matrix::Array{mk_float,2}
    # boundary condition differentiation matrix (for vperp grid using radau points)
    D0::Array{mk_float,1}
end

"""
A struct for Gauss-Legendre arrays needed for global operations in 1D,
contains the struct of elemental matrices for Lobatto and Radau points,
as well as some assembled 1D global matrices.
"""
struct gausslegendre_info{TSparse, TSparseCSR, TLU, TLmat, TLmatLU} <: weak_discretization_info
    lobatto::gausslegendre_base_info
    radau::gausslegendre_base_info
    # global (1D) mass matrix
    mass_matrix::Array{mk_float,2}
    # global (1D) weak second derivative matrix
    K_matrix::TSparse
    # global (1D) weak Laplacian derivative matrix
    L_matrix::TSparse
    # global (1D) strong first derivative matrix
    D_matrix::TSparse
    # global (1D) strong first derivative matrix in Compressed Sparse Row (CSR) format
    D_matrix_csr::TSparseCSR
    # global (1D) weak second derivative matrix, with inverse mass matrix included (so
    # matrix is dense)
    dense_second_deriv_matrix::Array{mk_float,2}
    # global (1D) weak Laplacian derivative matrix with boundary conditions - might be
    # `nothing` if boundary conditions are not supported
    L_matrix_with_bc::TLmat
    # mass matrix global (1D) LU object
    mass_matrix_lu::TLU
    # Laplacian global (1D) LU object - might be
    # `nothing` if boundary conditions are not supported
    L_matrix_lu::TLmatLU
    # dummy matrix for local operators
    Qmat::Array{mk_float,2}
end

"""
Function to create `gausslegendre_info` struct.
"""
function setup_gausslegendre_pseudospectral(coord)
    fem_coord_input = Array{element_coordinates,1}(undef,coord.nelement_local)
    # setup coordinate input for FiniteElementMatrices
    for ielement in 1:coord.nelement_local
        imin = coord.igrid_full[1,ielement]
        imax = coord.igrid_full[end,ielement]
        scale = coord.element_scale[ielement]
        shift = coord.element_shift[ielement]
        refnodes = allocate_float(coord.ngrid)
        @. refnodes = (coord.grid[imin:imax] - shift)/scale
        fem_coord_input[ielement] = element_coordinates(refnodes,scale,shift)
    end
    lobatto = setup_gausslegendre_pseudospectral_lobatto(coord)
    radau = setup_gausslegendre_pseudospectral_radau(coord)

    mass_matrix = allocate_float(coord.n,coord.n)
    K_matrix = allocate_float(coord.n,coord.n)
    L_matrix = allocate_float(coord.n,coord.n)
    D_matrix = allocate_float(coord.n,coord.n)

    dirichlet_bc = (coord.bc in ["zero", "constant"]) # and further options in future
    periodic_bc = (coord.bc == "periodic")
    setup_global_weak_form_matrix!(mass_matrix, lobatto, radau, fem_coord_input, coord, "M"; periodic_bc=periodic_bc)
    setup_global_weak_form_matrix!(K_matrix, lobatto, radau, fem_coord_input, coord, "K_with_BC_terms"; periodic_bc=periodic_bc)
    setup_global_weak_form_matrix!(L_matrix, lobatto, radau, fem_coord_input, coord, "L_with_BC_terms")
    setup_global_strong_form_matrix!(D_matrix, lobatto, radau, coord, "D"; periodic_bc=periodic_bc)
    dense_second_deriv_matrix = inv(mass_matrix) * K_matrix
    mass_matrix_lu = lu(sparse(mass_matrix))
    if dirichlet_bc || periodic_bc
        L_matrix_with_bc = allocate_float(coord.n,coord.n)
        setup_global_weak_form_matrix!(L_matrix_with_bc, lobatto, radau, fem_coord_input, coord, "L", dirichlet_bc=dirichlet_bc, periodic_bc=periodic_bc)
        L_matrix_with_bc = sparse(L_matrix_with_bc )
        L_matrix_lu = lu(sparse(L_matrix_with_bc))
    else
        L_matrix_with_bc = nothing
        L_matrix_lu = nothing
    end    

    Qmat = allocate_float(coord.ngrid,coord.ngrid)

    return gausslegendre_info(lobatto,radau,mass_matrix,sparse(K_matrix),sparse(L_matrix),sparse(D_matrix),convert(SparseMatrixCSR{1,mk_float,mk_int},D_matrix),dense_second_deriv_matrix,L_matrix_with_bc,
                              mass_matrix_lu,L_matrix_lu,Qmat)
end

"""
Function that creates the `gausslegendre_base_info` struct for Lobatto points.
"""
function setup_gausslegendre_pseudospectral_lobatto(coord)
    x, w = gausslobatto(coord.ngrid)
    x = mk_float.(x)
    w = mk_float.(w)
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    gausslobattolegendre_differentiation_matrix!(Dmat,x,coord.ngrid)
    indefinite_integration_matrix = allocate_float(coord.ngrid, coord.ngrid)
    integration_matrix!(indefinite_integration_matrix,x,coord.ngrid)
    D0 = allocate_float(coord.ngrid)
    #@. D0 = Dmat[1,:] # values at lower extreme of element
    GaussLegendre_derivative_vector!(D0,-1.0,coord.ngrid,x,w)
    return gausslegendre_base_info(true,Dmat,indefinite_integration_matrix,D0)
end

"""
Function that creates the `gausslegendre_base_info` struct for Radau points.
"""
function setup_gausslegendre_pseudospectral_radau(coord)
    # Gauss-Radau points on [-1,1)
    x, w = gaussradau(coord.ngrid)
    x = mk_float.(x)
    w = mk_float.(w)
    # Gauss-Radau points on (-1,1] 
    xreverse, wreverse = -reverse(x), reverse(w)
    # elemental differentiation matrix
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    gaussradaulegendre_differentiation_matrix!(Dmat,x,coord.ngrid)
    indefinite_integration_matrix = allocate_float(coord.ngrid, coord.ngrid)
    integration_matrix!(indefinite_integration_matrix,xreverse,coord.ngrid)
    D0 = allocate_float(coord.ngrid)
    GaussLegendre_derivative_vector!(D0,-1.0,coord.ngrid,xreverse,wreverse,radau=true)
    return gausslegendre_base_info(false,Dmat, indefinite_integration_matrix,D0)
end 

"""
A function that takes the first derivative in each element of `coord.grid`,
leaving the result (element-wise) in `coord.scratch_2d`.
"""
function elementwise_derivative!(coord, ff, gausslegendre::gausslegendre_info)
    df = coord.scratch_2d
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
    if coord.radau_first_element && coord.irank == 0 # differentiate this element with the Radau scheme
        @views mul!(df[:,j],gausslegendre.radau.Dmat[:,:],ff[imin:imax])
    else #differentiate using the Lobatto scheme
        @views mul!(df[:,j],gausslegendre.lobatto.Dmat[:,:],ff[imin:imax])
    end
    # transform back to the physical coordinate scale
    for i in 1:coord.ngrid
        df[i,j] /= coord.element_scale[j]
    end
    # calculate the derivative on each element
    @inbounds for j ∈ 2:nelement
        k = 1 
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        @views mul!(df[:,j],gausslegendre.lobatto.Dmat[:,:],ff[imin:imax])        
        # transform back to the physical coordinate scale
        for i in 1:coord.ngrid
            df[i,j] /= coord.element_scale[j]
        end
    end

    return nothing
end

"""
Wrapper function for element-wise derivatives with advection.
Note that Gauss-Legendre spectral the element method implemented here
does not use upwinding within an element.
"""
# Spectral element method does not use upwinding within an element
function elementwise_derivative!(coord, ff, adv_fac, spectral::gausslegendre_info)
    return elementwise_derivative!(coord, ff, spectral)
end

"""
Function to perform interpolation on a single element.
"""
function single_element_interpolate!(result, newgrid, f, imin, imax, ielement, coord,
                                     gausslegendre::gausslegendre_base_info,
                                     derivative::Val{0})
    n_new = length(newgrid)

    i = 1
    other_nodes = @view coord.other_nodes[:,i,ielement]
    one_over_denominator = coord.one_over_denominator[i,ielement]
    this_f = f[i]
    for j ∈ 1:n_new
        result[j] = this_f * lagrange_poly_optimised(other_nodes, one_over_denominator, newgrid[j])
    end
    for i ∈ 2:coord.ngrid
        other_nodes = @view coord.other_nodes[:,i,ielement]
        one_over_denominator = coord.one_over_denominator[i,ielement]
        this_f = f[i]
        for j ∈ 1:n_new
            result[j] += this_f * lagrange_poly_optimised(other_nodes, one_over_denominator, newgrid[j])
        end
    end

    return nothing
end

"""
Function to carry out a 1D (global) mass matrix solve.
"""
# Evaluate first derivative of the interpolating function
function single_element_interpolate!(result, newgrid, f, imin, imax, ielement, coord,
                                     gausslegendre::gausslegendre_base_info,
                                     derivative::Val{1})
    n_new = length(newgrid)

    i = 1
    other_nodes = @view coord.other_nodes[:,i,ielement]
    one_over_denominator = coord.one_over_denominator[i,ielement]
    this_f = f[i]
    for j ∈ 1:n_new
        result[j] = this_f * lagrange_poly_derivative_optimised(other_nodes, one_over_denominator, newgrid[j])
    end
    for i ∈ 2:coord.ngrid
        other_nodes = @view coord.other_nodes[:,i,ielement]
        one_over_denominator = coord.one_over_denominator[i,ielement]
        this_f = f[i]
        for j ∈ 1:n_new
            result[j] += this_f * lagrange_poly_derivative_optimised(other_nodes, one_over_denominator, newgrid[j])
        end
    end

    return nothing
end

function fill_single_element_interpolation_matrix!(
             matrix_slice, newgrid, jelement, coord,
             gausslegendre::gausslegendre_base_info)
    n_new = length(newgrid)

    for j ∈ 1:coord.ngrid
        other_nodes = @view coord.other_nodes[:,j,jelement]
        one_over_denominator = coord.one_over_denominator[j,jelement]
        for i ∈ 1:n_new
            matrix_slice[i,j] = lagrange_poly_optimised(other_nodes, one_over_denominator, newgrid[i])
        end
    end

    return nothing
end

"""
Function to carry out a 1D (global) mass matrix solve.
"""
function mass_matrix_solve!(f, b, spectral::gausslegendre_info)
    # invert mass matrix system
    ldiv!(f, spectral.mass_matrix_lu, b)
    return nothing
end

"""
Function to calculate the elemental anti-differentiation (integration) matrix.
This function forms the primitive
```math
F(x) = \\int^x_{x_{\\rm min}} f(x^\\prime) d x^\\prime
```
of the function 
```math
f(x) = \\sum_j f_j l_j(x),
```
where \$l_j(x)\$ is the \$j^{\\rm th}\$ Lagrange polynomial on the element and \$f_j = f(x_j)\$,
with \$x_j\$ \$j^{\\rm th}\$ collocation point on the element. We find \$F(x)\$ at the collocation
points on the element, giving a series of integrals to evaluate:
```math
F(x_i) = \\int^{x_i}_{-1} f(x^\\prime) d x^\\prime = \\sum_j f_j \\int^{x_i}_{-1} l_j(x^\\prime) d x^\\prime,
```
where we have used that \$x_{\\rm min} = -1\$ on the elemental grid.
Changing to a normalised coordinate \$y\$ suitable for Gaussian quadrature 
```math
x^\\prime = \\frac{x_i + 1}{2} y + \\frac{x_i - 1}{2}
```
we can write the operation in matrix form:
```math
F(x_i) = \\sum_{j}A_{ij}f_j,
```
with the matrix \$A_{ij}\$ defined by
```math
A_{ij} = \\left(\\frac{x_i + 1}{2}\\right) \\int^1_{-1} l_j \\left( \\frac{(x_i + 1)y + x_i - 1}{2} \\right) dy,
```
or in discretised form
```math
A_{ij} = \\left(\\frac{x_i + 1}{2}\\right) \\sum_k l_j \\left( \\frac{(x_i + 1)y_k + x_i - 1}{2} \\right) w_k,
```
with \$y_k\$ and \$w_k\$ Gauss-quadrature points and weights, respectively.
"""
function integration_matrix!(A::Array{Float64,2},x::Array{Float64,1},ngrid::Int64)
    nquad = 2*ngrid
    zz, wz = gausslegendre(nquad)
    # set lower limit
    for j in 1:ngrid
        A[1,j] = 0.0
    end
    # set the remaining points
    for i in 1:ngrid
        for j in 1:ngrid
            xi = x[i] # limit of ith integral
            scale = 0.5*(xi + 1)
            shift = 0.5*(xi - 1)
            # calculate the sum with Gaussian quadrature
            A[i,j] = 0.0
            for k in 1:nquad
                xval = scale*zz[k] + shift
                A[i,j] += scale*lagrange_poly(j,x,xval)*wz[k]
            end
        end
    end
    return nothing
end

"""
Formula for Gauss-Legendre-Lobatto differentiation matrix taken from p196 of Chpt `The Spectral Elemtent Method' of 
`Computational Seismology'. Heiner Igel First Edition. Published in 2017 by Oxford University Press.
Or https://doc.nektar.info/tutorials/latest/fundamentals/differentiation/fundamentals-differentiationch2.html

    D -- differentiation matrix 
    x -- Gauss-Legendre-Lobatto points in [-1,1]
    ngrid -- number of points per element (incl. boundary points)

Note that D has does not include a scaling factor
"""
function gausslobattolegendre_differentiation_matrix!(D::Array{mk_float,2},x::Array{mk_float,1},ngrid::mk_int)
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
    return nothing
end

"""
Formula for Gauss-Legendre-Radau differentiation matrix taken from
https://doc.nektar.info/tutorials/latest/fundamentals/differentiation/fundamentals-differentiationch2.html

    D -- differentiation matrix 
    x -- Gauss-Legendre-Radau points in [-1,1)
    ngrid -- number of points per element (incl. boundary points)

Note that D has does not include a scaling factor
"""
function gaussradaulegendre_differentiation_matrix!(D::Array{mk_float,2},x::Array{mk_float,1},ngrid::Int64)
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
Gauss-Legendre derivative at arbitrary x values, for boundary condition on Radau points.
    D0 -- the vector
    xj -- the x location where the derivative is evaluated 
    ngrid -- number of points in x
    x -- the grid from -1, 1
Note that D0 is not scaled to the physical grid with a scaling factor.
"""
function GaussLegendre_derivative_vector!(D0,xj,ngrid,x,wgts;radau=false)
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
    #@. D0 *= 2.0*float(nelement_global)/L
end

"""
Result of the inner product of Legendre polynomials of order k.
"""
function Legendre_h_n(k)
    h_n = 2.0/(2.0*k + 1)
    return h_n
end 

"""
Function for setting up the full Gauss-Legendre-Lobatto
grid and collocation point weights.
"""
function scaled_gauss_legendre_lobatto_grid(ngrid, nelement_local, n_local, element_scale, element_shift, imin, imax)
    # get Gauss-Legendre-Lobatto points and weights on [-1,1]
    x, w = gausslobatto(ngrid)
    # grid and weights arrays
    grid = allocate_float(n_local)
    wgts = allocate_float(n_local)
    wgts .= 0.0
    #integer to deal with the overlap of element boundaries
    k = 1
    @inbounds for j in 1:nelement_local
        # element_scale[j]
        # element_shift[j]
        # factor with maps [-1,1] -> a subset of [-L/2, L/2]
        @. grid[imin[j]:imax[j]] = element_scale[j]*x[k:ngrid] + element_shift[j]
        
        # calculate the weights
        # remembering on boundary points to include weights
        # from both left and right elements
        #println(imin[j]," ",imax[j])
        @. wgts[imin[j] - k + 1:imax[j]] += element_scale[j]*w[1:ngrid] 
        
        k = 2        
    end
    return grid, wgts
end

"""
Function for setting up the full Gauss-Legendre-Radau
grid and collocation point weights.
"""
function scaled_gauss_legendre_radau_grid(ngrid, nelement_local, n_local, element_scale, element_shift, imin, imax, irank)
    # get Gauss-Legendre-Lobatto points and weights on [-1,1]
    x_lob, w_lob = gausslobatto(ngrid)
    # get Gauss-Legendre-Radau points and weights on [-1,1)
    x_rad, w_rad = gaussradau(ngrid)
    # transform to a Gauss-Legendre-Radau grid on (-1,1]
    x_rad, w_rad = -reverse(x_rad), reverse(w_rad)#
    # grid and weights arrays
    grid = allocate_float(n_local)
    wgts = allocate_float(n_local)
    wgts .= 0.0
    if irank == 0
        # for 1st element, fill in with Gauss-Legendre-Radau points
        j = 1
        # element_scale[j]
        # element_shift[j]
        # factor with maps [-1,1] -> a subset of [-L/2, L/2]
        @. grid[imin[j]:imax[j]] = element_scale[j]*x_rad[1:ngrid] + element_shift[j]
        @. wgts[imin[j]:imax[j]] += element_scale[j]*w_rad[1:ngrid]       
        #integer to deal with the overlap of element boundaries
        k = 2
        @inbounds for j in 2:nelement_local
            # element_scale[j]
            # element_shift[j]
            # factor with maps [-1,1] -> a subset of [-L/2, L/2]
            @. grid[imin[j]:imax[j]] = element_scale[j]*x_lob[k:ngrid] + element_shift[j]
            @. wgts[imin[j] - k + 1:imax[j]] += element_scale[j]*w_lob[1:ngrid]         
        end
    else # all elements are Gauss-Legendre-Lobatto
        #integer to deal with the overlap of element boundaries
        k = 1
        @inbounds for j in 1:nelement_local
            # element_scale[j]
            # element_shift[j]
            # factor with maps [-1,1] -> a subset of [-L/2, L/2]
            @. grid[imin[j]:imax[j]] = element_scale[j]*x_lob[k:ngrid] + element_shift[j]
            @. wgts[imin[j] - k + 1:imax[j]] += element_scale[j]*w_lob[1:ngrid]            
            k = 2 
        end
    end
    return grid, wgts
end

"""
A function that assigns the local weak-form matrices to 
a global array QQ_global for later solving weak form of required
1D equation.

The 'option' variable is a flag for 
choosing the type of matrix to be constructed. 
Currently the function is set up to assemble the 
elemental matrices without imposing boundary conditions on the 
first and final rows of the matrix by default. This means that 
the operators constructed from this function can only be used
for differentiation, and not solving 1D ODEs.
This assembly function assumes that the 
coordinate is not distributed. To extend this function to support
distributed-memory MPI, addition of off-memory matrix elements
to the exterior points would be required.

The typical use of this function is to assemble matrixes M and K in

 M * d2f = K * f 
 
where M is the mass matrix and K is the stiffness matrix, and we wish to
solve for d2f given f. To solve 1D ODEs

K * f = b = M * d2f 

for f given boundary data on f
with periodic or dirichlet boundary conditions, set 

`periodic_bc = true`, `b[end] = 0`

or 

`dirichlet_bc = true`, `b[1] = f[1]` (except for cylindrical coordinates), `b[end] = f[end]`

in the function call, and create new matrices for this purpose
in the `gausslegendre_info` struct. Currently the Laplacian matrix
is supported with boundary conditions.
"""
function setup_global_weak_form_matrix!(QQ_global::Array{mk_float,2},
                               lobatto::gausslegendre_base_info,
                               radau::gausslegendre_base_info,
                               fem_coord_input::Array{element_coordinates,1},
                               coord,option; dirichlet_bc=false, periodic_bc=false)
    QQ_j = allocate_float(coord.ngrid,coord.ngrid)
    
    ngrid = coord.ngrid
    imin = coord.imin
    imax = coord.imax
    @. QQ_global = 0.0
    
    # assembly below assumes no contributions 
    # from elements outside of local domain
    k = 0
    for j in 1:coord.nelement_local
        get_QQ_local!(QQ_j,j,lobatto,radau,fem_coord_input[j],coord,option)
        iminl = imin[j]-k
        imaxl = imax[j]
        @. QQ_global[iminl:imaxl,iminl:imaxl] += QQ_j[:,:]
        k = 1
    end

    if dirichlet_bc
        # Make matrix diagonal for first/last grid points so it does not change the values
        # there
        if !(coord.name == "vperp") 
            # modify lower endpoint if not a radial/cylindrical coordinate
            if coord.irank == 0
                QQ_global[1,:] .= 0.0
                QQ_global[1,1] = 1.0
            end
        end
        if coord.irank == coord.nrank - 1
            QQ_global[end,:] .= 0.0
            QQ_global[end,end] = 1.0
        end
        # requires RHS vector b[1],b[end] = boundary values
    end
    if periodic_bc
        # Make periodic boundary condition by modifying elements of matrix for duplicate point
        # add assembly contribution to lower endpoint from upper endpoint
        j = coord.nelement_local
        get_QQ_local!(QQ_j,j,lobatto,radau,fem_coord_input[j],coord,option)
        iminl = imin[j] - mk_int(coord.nelement_local > 1)
        imaxl = imax[j]
        QQ_global[1,iminl:imaxl] .+= QQ_j[end,:]
        # Enforce continuity at the periodic boundary
        # All-zero row in RHS matrix sets last element of `b` vector in
        # `mass_matrix.x = b` to zero
        QQ_global[end,:] .= 0.0
        # enforce periodicity `x[1] = x[end]` using the last row of the
        # `A.x = b` system.
        QQ_global[end,1] = 1.0
        QQ_global[end,end] = -1.0
        if option == "L" # or any other derivative (ODE) matrix requiring two BC (periodicity + value at endpoint)
            QQ_global[1,:] .= 0.0
            QQ_global[1,1] = 1.0
        end
    end
        
    return nothing
end

"""
A function that assigns the local matrices to
a global array QQ_global for later evaluating strong form of required 1D equation.

The 'option' variable is a flag for
choosing the type of matrix to be constructed.
Currently the function is set up to assemble the
elemental matrices without imposing boundary conditions on the
first and final rows of the matrix. This means that
the operators constructed from this function can only be used
for differentiation, and not solving 1D ODEs.
The shared points in the element assembly are
averaged (instead of simply added) to be consistent with the
`derivative_elements_to_full_grid!()` function in `calculus.jl`.
"""
function setup_global_strong_form_matrix!(QQ_global::Array{mk_float,2},
                                          lobatto::gausslegendre_base_info,
                                          radau::gausslegendre_base_info, 
                                          coord,option; periodic_bc=false)
    QQ_j = allocate_float(coord.ngrid,coord.ngrid)
    QQ_jp1 = allocate_float(coord.ngrid,coord.ngrid)

    ngrid = coord.ngrid
    imin = coord.imin
    imax = coord.imax
    @. QQ_global = 0.0

    # fill in first element
    j = 1
    # N.B. QQ varies with ielement for vperp, but not vpa
    # a radau element is used for the vperp grid (see get_QQ_local!())
    get_QQ_local!(QQ_j,j,lobatto,radau,coord,option)
    if periodic_bc && coord.nrank != 1
        error("periodic boundary conditions not supported when dimension is distributed")
    end
    if periodic_bc && coord.nrank == 1
        QQ_global[imax[end], imin[j]:imax[j]] .+= QQ_j[1,:] ./ 2.0
        QQ_global[1,1] += 1.0
        QQ_global[1,end] += -1.0
    else
        QQ_global[imin[j],imin[j]:imax[j]] .+= QQ_j[1,:]
    end
    for k in 2:imax[j]-imin[j] 
        QQ_global[k,imin[j]:imax[j]] .+= QQ_j[k,:]
    end
    if coord.nelement_local > 1
        QQ_global[imax[j],imin[j]:imax[j]] .+= QQ_j[ngrid,:]./2.0
    else
        QQ_global[imax[j],imin[j]:imax[j]] .+= QQ_j[ngrid,:]
    end
    # remaining elements recalling definitions of imax and imin
    for j in 2:coord.nelement_local
        get_QQ_local!(QQ_j,j,lobatto,radau,coord,option)
        #lower boundary assembly on element
        QQ_global[imin[j]-1,imin[j]-1:imax[j]] .+= QQ_j[1,:]./2.0
        for k in 2:imax[j]-imin[j]+1
            QQ_global[k+imin[j]-2,imin[j]-1:imax[j]] .+= QQ_j[k,:]
        end
        # upper boundary assembly on element
        if j == coord.nelement_local
            if periodic_bc && coord.nrank == 1
                QQ_global[imax[j],imin[j]-1:imax[j]] .+= QQ_j[ngrid,:] / 2.0
            else
                QQ_global[imax[j],imin[j]-1:imax[j]] .+= QQ_j[ngrid,:]
            end
        else
            QQ_global[imax[j],imin[j]-1:imax[j]] .+= QQ_j[ngrid,:]./2.0
        end
    end

    return nothing
end

"""
Construction function to provide the appropriate elemental 
matrix `Q` to the global matrix assembly functions.
"""
function get_QQ_local!(QQ::AbstractArray{mk_float,2},ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info,
        fem_coord_input::element_coordinates,
        coord,option)

        if coord.name == "vperp"
            power = 1
        else
            power = 0
        end
        if option == "M"
            QQ .= finite_element_matrix(lagrange_x, lagrange_x, power, fem_coord_input)
        elseif option == "P"
            QQ .= finite_element_matrix(lagrange_x, d_lagrange_dx, power, fem_coord_input)
        elseif option == "K"
            get_KK_local!(QQ,ielement,lobatto,radau,fem_coord_input,coord)
        elseif option == "K_with_BC_terms"
            get_KK_local!(QQ,ielement,lobatto,radau,fem_coord_input,coord,explicit_BC_terms=true)
        elseif option == "L"
            get_LL_local!(QQ,ielement,lobatto,radau,fem_coord_input,coord)
        elseif option == "L_with_BC_terms"
            get_LL_local!(QQ,ielement,lobatto,radau,fem_coord_input,coord,explicit_BC_terms=true)
        end
        return nothing
end
function get_QQ_local!(QQ::AbstractArray{mk_float,2},ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info,
        coord,option)

        if option == "D"
            get_DD_local!(QQ,ielement,lobatto,radau,coord)
        end
        return nothing
end

"""
If called for `coord.name = vperp` elemental matrix `KK` on the \$i^{th}\$ element is
```math
 K_{jk} = -\\int^{v_\\perp^U}_{v_\\perp^L} \\left(v_\\perp \\frac{\\partial\\varphi_j(v_\\perp)}{\\partial v_\\perp} + \\varphi_j(v_\\perp) \\right)
 \\frac{\\partial\\varphi_k(v_\\perp)}{\\partial v_\\perp} d v_\\perp
 = -\\int^1_{-1} ((c_i + x s_i)l_j^\\prime(x) + l_j(x))l_k^\\prime(x) d x /s_i
```
with \$c_i\$ and \$s_i\$ the appropriate shift and scale factors, respectively. 
Otherwise, if called for any other coordinate elemental matrix `KK` is the same as `LL` (see `get_LL_local!()).
If `explicit_BC_terms = true`, boundary terms arising from integration by parts are included at the extreme boundary points.
"""
function get_KK_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info, 
        fem_coord_input::element_coordinates,
        coord;explicit_BC_terms=false)
        nelement = coord.nelement_local
        scale_factor = coord.element_scale[ielement]
        shift_factor = coord.element_shift[ielement]
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            QQ .= -finite_element_matrix(d_lagrange_dx, d_lagrange_dx, 1, fem_coord_input)
            QQ .-= finite_element_matrix(lagrange_x, d_lagrange_dx, 0, fem_coord_input)
            # extra scale and shift factors required because of vperp in integral
            # P0 factors make this a d^2 / dvperp^2 rather than (1/vperp) d ( vperp d (.) / d vperp)
            if ielement > 1 || coord.irank > 0 # lobatto points
                # boundary terms from integration by parts
                if explicit_BC_terms && ielement == 1 && coord.irank == 0
                    imin = coord.imin[ielement] - 1
                    @. QQ[1,:] -= coord.grid[imin]*lobatto.Dmat[1,:]/scale_factor
                end
                if explicit_BC_terms && ielement == nelement && coord.irank == coord.nrank - 1
                    imax = coord.imax[ielement]
                    @. QQ[coord.ngrid,:] += coord.grid[imax]*lobatto.Dmat[coord.ngrid,:]/scale_factor  
                end
            else # radau points 
                # boundary terms from integration by parts
                if explicit_BC_terms && ielement == nelement && coord.irank == coord.nrank - 1 
                    imax = coord.imax[ielement]
                    @. QQ[coord.ngrid,:] += coord.grid[imax]*radau.Dmat[coord.ngrid,:]/scale_factor
                end
            end
        else # assume integrals of form int^infty_-infty (.) d vpa
            QQ .= -finite_element_matrix(d_lagrange_dx, d_lagrange_dx, 0, fem_coord_input)
            if coord.bc != "periodic"
                # boundary terms from integration by parts
                if explicit_BC_terms && ielement == 1 && coord.irank == 0
                    @. QQ[1,:] -= lobatto.Dmat[1,:]/scale_factor
                end
                if explicit_BC_terms && ielement == nelement && coord.irank == coord.nrank - 1
                    @. QQ[coord.ngrid,:] += lobatto.Dmat[coord.ngrid,:]/scale_factor
                end
            end
        end
        return nothing
end

"""
If called for `coord.name = vperp` elemental matrix `LL` on the \$i^{th}\$ element is
```math
 L_{jk} = -\\int^{v_\\perp^U}_{v_\\perp^L} \\frac{\\partial\\varphi_j(v_\\perp)}{\\partial v_\\perp}\\frac{\\partial\\varphi_k(v_\\perp)}{\\partial v_\\perp} v_\\perp d v_\\perp
 = -\\int^1_{-1} (c_i + x s_i)l_j^\\prime(x)l_k^\\prime(x) d x /s_i
```
with \$c_i\$ and \$s_i\$ the appropriate shift and scale factors, respectively. 
Otherwise, if called for any other coordinate elemental matrix `LL` is 
```math
 L_{jk} = -\\int^{v_\\|^U}_{v_\\|^L}  \\frac{\\partial\\varphi_j(v_\\|)}{\\partial v_\\|}\\frac{\\partial\\varphi_k(v_\\|)}{\\partial v_\\|} d v_\\| =
 -\\int^1_{-1} l_j^\\prime(x)l_k^\\prime(x) d x /s_i.
```
If `explicit_BC_terms = true`, boundary terms arising from integration by parts are included at the extreme boundary points.
"""
function get_LL_local!(QQ,ielement,
        lobatto::gausslegendre_base_info,
        radau::gausslegendre_base_info,
        fem_coord_input::element_coordinates,
        coord;explicit_BC_terms=false)
        nelement = coord.nelement_local
        scale_factor = coord.element_scale[ielement]
        shift_factor = coord.element_shift[ielement]
        if coord.name == "vperp" # assume integrals of form int^infty_0 (.) vperp d vperp
            QQ .= -finite_element_matrix(d_lagrange_dx, d_lagrange_dx, 1, fem_coord_input)
            # extra scale and shift factors required because of vperp in integral
            #  (1/vperp) d ( vperp d (.) / d vperp)
            if ielement > 1 || coord.irank > 0 # lobatto points
                # boundary terms from integration by parts
                if explicit_BC_terms && ielement == 1 && coord.irank == 0
                    imin = coord.imin[ielement] - 1
                    @. QQ[1,:] -= coord.grid[imin]*lobatto.Dmat[1,:]/scale_factor
                end
                if explicit_BC_terms && ielement == nelement && coord.irank == coord.nrank - 1
                    imax = coord.imax[ielement]
                    @. QQ[coord.ngrid,:] += coord.grid[imax]*lobatto.Dmat[coord.ngrid,:]/scale_factor
                end
            else # radau points 
                # boundary terms from integration by parts
                if explicit_BC_terms && ielement == nelement && coord.irank == coord.nrank - 1
                    imax = coord.imax[ielement]
                    @. QQ[coord.ngrid,:] += coord.grid[imax]*radau.Dmat[coord.ngrid,:]/scale_factor
                end
            end
        else # d^2 (.) d vpa^2 -- assume integrals of form int^infty_-infty (.) d vpa
            QQ .= -finite_element_matrix(d_lagrange_dx, d_lagrange_dx, 0, fem_coord_input)
            if coord.bc != "periodic"
                # boundary terms from integration by parts
                if explicit_BC_terms && ielement == 1 && coord.irank == 0
                    @. QQ[1,:] -= lobatto.Dmat[1,:]/scale_factor
                end
                if explicit_BC_terms && ielement == nelement && coord.irank == coord.nrank - 1
                    @. QQ[coord.ngrid,:] += lobatto.Dmat[coord.ngrid,:]/scale_factor
                end
            end
        end
        return nothing
end

# Strong-form differentiation matrix
function get_DD_local!(QQ, ielement, lobatto::gausslegendre_base_info,
                       radau::gausslegendre_base_info, coord)
    scale_factor = coord.element_scale[ielement]
    if coord.name == "vperp" && ielement == 1 && coord.irank == 0
        @. QQ = radau.Dmat / scale_factor
    else
        @. QQ = lobatto.Dmat / scale_factor
    end
    return nothing
end

end
