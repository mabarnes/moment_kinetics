using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

using Pkg
Pkg.activate(".")

import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_int
    using moment_kinetics.coordinates: define_coordinate
    using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral, chebyshev_radau_derivative_single_element!
    using moment_kinetics.calculus: derivative!, integral
using moment_kinetics.type_definitions
#import LinearAlgebra
#using IterativeSolvers: jacobi!, gauss_seidel!, idrs!
using LinearAlgebra: mul!, lu, cond, det
using SparseArrays: sparse
using SpecialFunctions: erf
zero = 1.0e-10

function print_matrix(matrix,name,n,m)
    println("\n ",name," \n")
    for i in 1:n
        for j in 1:m
            @printf("%.1f ", matrix[i,j])
        end
        println("")
    end
    println("\n")
end

function print_vector(vector,name,m)
    println("\n ",name," \n")
    for j in 1:m
        @printf("%.3f ", vector[j])
    end
    println("")
    println("\n")
end 

function Djj(x::MKArray{Float64,1},j::Int64)
    return -0.5*x[j]/( 1.0 - x[j]^2)
end
function Djk(x::MKArray{Float64,1},j::Int64,k::Int64,c_j::Float64,c_k::Float64)
    return  (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
end

"""
The function below is based on the numerical method outlined in 
Chapter 8.2 from Trefethen 1994 
https://people.maths.ox.ac.uk/trefethen/8all.pdf
full list of Chapters may be obtained here 
https://people.maths.ox.ac.uk/trefethen/pdetext.html
"""

function cheb_derivative_matrix!(D::MKArray{Float64,2},x::MKArray{Float64,1},n)
    D[:,:] .= 0.0
    
    # top left, bottom right
    D[1,1] = (2.0*(n - 1.0)^2 + 1.0)/6.0
    D[n,n] = -(2.0*(n - 1.0)^2 + 1.0)/6.0
    
    # top row 
    j = 1
    c_j = 2.0 
    c_k = 1.0
    for k in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    k = n 
    c_k = 2.0
    D[j,k] = Djk(x,j,k,c_j,c_k)
    
    # bottom row 
    j = n
    c_j = 2.0 
    c_k = 1.0
    for k in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    k = 1
    c_k = 2.0
    D[j,k] = Djk(x,j,k,c_j,c_k)
    
    #left column
    k = 1
    c_j = 1.0 
    c_k = 2.0
    for j in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    
    #right column
    k = n
    c_j = 1.0 
    c_k = 2.0
    for j in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    
    # interior rows and columns
    for j in 2:n-1
        D[j,j] = Djj(x,j)
        #D[j,j] = -0.5*x[j]/( 1.0 - x[j]^2)
        for k in 2:n-1
            if j == k 
                continue
            end
            c_k = 1.0
            c_j = 1.0
            #D[j,k] = (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
    end
end 

function cheb_derivative_matrix_reversed!(D::MKArray{Float64,2},x)
    D_elementwise = allocate_float(x.ngrid,x.ngrid)
    cheb_derivative_matrix_elementwise_reversed!(D_elementwise,x.ngrid,x.L,x.nelement_global)    
    if x.ngrid < 8
        println("\n D_elementwise \n")
        for i in 1:x.ngrid
            for j in 1:x.ngrid
                @printf("%.1f ", D_elementwise[i,j])
            end
            println("")
        end
    end 
    assign_cheb_derivative_matrix!(D,D_elementwise,x)
end

function cheb_second_derivative_matrix_reversed!(D::MKArray{Float64,2},x)
    D_elementwise = allocate_float(x.ngrid,x.ngrid)
    cheb_derivative_matrix_elementwise_reversed!(D_elementwise,x.ngrid,x.L,x.nelement_global)    
    D2_elementwise = allocate_float(x.ngrid,x.ngrid)
    mul!(D2_elementwise,D_elementwise,D_elementwise)
    if x.ngrid < 8
        print_matrix(D2_elementwise,"D2_elementwise",x.ngrid,x.ngrid)
    end
    assign_cheb_derivative_matrix!(D,D2_elementwise,x)
end

function assign_cheb_derivative_matrix!(D::MKArray{Float64,2},D_elementwise::MKArray{Float64,2},x)
    
    # zero output matrix before assignment 
    D[:,:] .= 0.0
    imin = x.imin
    imax = x.imax
    
    zero_bc_upper_boundary = x.bc == "zero" || x.bc == "zero_upper"
    zero_bc_lower_boundary = x.bc == "zero" || x.bc == "zero_lower"
    
    # fill in first element 
    j = 1
    if zero_bc_lower_boundary #x.bc == "zero"
        D[imin[j],imin[j]:imax[j]] .+= D_elementwise[1,:]./2.0 #contributions from this element/2
        D[imin[j],imin[j]] += D_elementwise[x.ngrid,x.ngrid]/2.0 #contribution from missing `zero' element/2
    else 
        D[imin[j],imin[j]:imax[j]] .+= D_elementwise[1,:]
    end
    for k in 2:imax[j]-imin[j] 
        D[k,imin[j]:imax[j]] .+= D_elementwise[k,:]
    end
    if zero_bc_upper_boundary && x.nelement_local == 1
        D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0 #contributions from this element/2
        D[imax[j],imax[j]] += D_elementwise[1,1]/2.0              #contribution from missing `zero' element/2
    elseif x.nelement_local > 1 #x.bc == "zero"
        D[imax[j],imin[j]:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0
    else
        D[imax[j],imin[j]:imax[j]] .+= D_elementwise[x.ngrid,:]
    end 
    # remaining elements recalling definitions of imax and imin
    for j in 2:x.nelement_local
        #lower boundary condition on element
        D[imin[j]-1,imin[j]-1:imax[j]] .+= D_elementwise[1,:]./2.0
        for k in 2:imax[j]-imin[j]+1 
            D[k+imin[j]-2,imin[j]-1:imax[j]] .+= D_elementwise[k,:]
        end
        # upper boundary condition on element 
        if j == x.nelement_local && !(zero_bc_upper_boundary)
            D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]
        elseif j == x.nelement_local && zero_bc_upper_boundary
            D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0 #contributions from this element/2
            D[imax[j],imax[j]] += D_elementwise[1,1]/2.0 #contribution from missing `zero' element/2
        else 
            D[imax[j],imin[j]-1:imax[j]] .+= D_elementwise[x.ngrid,:]./2.0
        end
    end
    
end

function cheb_derivative_matrix_elementwise_reversed!(D::MKArray{Float64,2},n::Int64,L::Float64,nelement::Int64)
    
    #define Chebyshev points in reversed order x_j = { -1, ... , 1}
    x = allocate_float(n)
    for j in 1:n
        x[j] = cospi((n-j)/(n-1))
    end
    
    # zero matrix before allocating values
    D[:,:] .= 0.0
    
    # top row 
    j = 1
    c_j = 2.0 
    c_k = 1.0
    for k in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    k = n 
    c_k = 2.0
    D[j,k] = Djk(x,j,k,c_j,c_k)
    
    # bottom row 
    j = n
    c_j = 2.0 
    c_k = 1.0
    for k in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    k = 1
    c_k = 2.0
    D[j,k] = Djk(x,j,k,c_j,c_k)
    
    #left column
    k = 1
    c_j = 1.0 
    c_k = 2.0
    for j in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    
    #right column
    k = n
    c_j = 1.0 
    c_k = 2.0
    for j in 2:n-1
        D[j,k] = Djk(x,j,k,c_j,c_k)
    end
    
    
    # top left, bottom right
    #D[n,n] = (2.0*(n - 1.0)^2 + 1.0)/6.0
    #D[1,1] = -(2.0*(n - 1.0)^2 + 1.0)/6.0        
    # interior rows and columns
    for j in 2:n-1
        #D[j,j] = Djj(x,j)
        for k in 2:n-1
            if j == k 
                continue
            end
            c_k = 1.0
            c_j = 1.0
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
    end
    
    # calculate diagonal entries to guarantee that
    # D * (1, 1, ..., 1, 1) = (0, 0, ..., 0, 0)
    for j in 1:n
        D[j,j] = -sum(D[j,:])
    end
    
    #multiply by scale factor for element length
    D .= (2.0*float(nelement)/L).*D
end 

"""
derivative matrix for radau grid 
"""
function calculate_chebyshev_radau_D_matrix_via_FFT!(D::MKArray{Float64,2}, coord, spectral)
    ff_buffer = allocate_float(coord.ngrid)
    df_buffer = allocate_float(coord.ngrid)
    # use response matrix approach to calculate derivative matrix D 
    for j in 1:coord.ngrid 
        ff_buffer .= 0.0 
        ff_buffer[j] = 1.0
        @views chebyshev_radau_derivative_single_element!(df_buffer[:], ff_buffer[:],
            spectral.radau.f[:,1], spectral.radau.df, spectral.radau.fext, spectral.radau.forward, coord)
        @. D[:,j] = df_buffer[:] # assign appropriate column of derivative matrix 
    end
    # correct diagonal elements to gurantee numerical stability
    # gives D*[1.0, 1.0, ... 1.0] = [0.0, 0.0, ... 0.0]
    for j in 1:coord.ngrid
        D[j,j] = 0.0
        D[j,j] = -sum(D[j,:])
    end
    
    #multiply by scale factor for element length
    D .= (2.0*float(coord.nelement_global)/coord.L).*D
end

function cheb_radau_derivative_matrix_reversed!(D::MKArray{Float64,2},x,x_spectral)
    D_lobotto_elementwise = allocate_float(x.ngrid,x.ngrid)
    cheb_derivative_matrix_elementwise_reversed!(D_lobotto_elementwise,x.ngrid,x.L,x.nelement_global) 

    D_radau_elementwise = allocate_float(x.ngrid,x.ngrid)
    calculate_chebyshev_radau_D_matrix_via_FFT!(D_radau_elementwise,x,x_spectral)
    if x.ngrid < 8
        print_matrix(D_lobotto_elementwise,"D_lobotto_elementwise",x.ngrid,x.ngrid)
        print_matrix(D_radau_elementwise,"D_radau_elementwise",x.ngrid,x.ngrid)
    end 
    assign_cheb_derivative_matrix!(D,D_lobotto_elementwise,D_radau_elementwise,x)
end

function cheb_radau_second_derivative_matrix_reversed!(D::MKArray{Float64,2},x,x_spectral)
    D_lobotto_elementwise = allocate_float(x.ngrid,x.ngrid)
    cheb_derivative_matrix_elementwise_reversed!(D_lobotto_elementwise,x.ngrid,x.L,x.nelement_global)    
    D2_lobotto_elementwise = allocate_float(x.ngrid,x.ngrid)
    mul!(D2_lobotto_elementwise,D_lobotto_elementwise,D_lobotto_elementwise)
    
    D_radau_elementwise = allocate_float(x.ngrid,x.ngrid)
    calculate_chebyshev_radau_D_matrix_via_FFT!(D_radau_elementwise,x,x_spectral)
    D2_radau_elementwise = allocate_float(x.ngrid,x.ngrid)
    mul!(D2_radau_elementwise,D_radau_elementwise,D_radau_elementwise)
    
    if x.ngrid < 8
        #print_matrix(D_lobotto_elementwise,"D_lobotto_elementwise",x.ngrid,x.ngrid)
        print_matrix(D2_lobotto_elementwise,"D2_lobotto_elementwise",x.ngrid,x.ngrid)
        #print_matrix(D_radau_elementwise,"D_radau_elementwise",x.ngrid,x.ngrid)
        print_matrix(D2_radau_elementwise,"D2_radau_elementwise",x.ngrid,x.ngrid)
    end
    assign_cheb_derivative_matrix!(D,D2_lobotto_elementwise,D2_radau_elementwise,x)
end


function assign_cheb_derivative_matrix!(D::MKArray{Float64,2},D_lobotto_elementwise::MKArray{Float64,2},D_radau_elementwise::MKArray{Float64,2},x)
    
    # zero output matrix before assignment 
    D[:,:] .= 0.0
    imin = x.imin
    imax = x.imax
    
    zero_bc_upper_boundary = x.bc == "zero" || x.bc == "zero_upper"
    zero_bc_lower_boundary = x.bc == "zero" || x.bc == "zero_lower"
    
    # fill in first element 
    j = 1
    if zero_bc_lower_boundary #x.bc == "zero"
        D[imin[j],imin[j]:imax[j]] .+= D_radau_elementwise[1,:]./2.0 #contributions from this element/2
        D[imin[j],imin[j]] += D_radau_elementwise[x.ngrid,x.ngrid]/2.0 #contribution from missing `zero' element/2
    else 
        D[imin[j],imin[j]:imax[j]] .+= D_radau_elementwise[1,:]
    end
    for k in 2:imax[j]-imin[j] 
        D[k,imin[j]:imax[j]] .+= D_radau_elementwise[k,:]
    end
    if zero_bc_upper_boundary && x.nelement_local == 1
        D[imax[j],imin[j]-1:imax[j]] .+= D_radau_elementwise[x.ngrid,:]./2.0 #contributions from this element/2
        D[imax[j],imax[j]] += D_lobotto_elementwise[1,1]/2.0              #contribution from missing `zero' element/2
    elseif x.nelement_local > 1 #x.bc == "zero"
        D[imax[j],imin[j]:imax[j]] .+= D_radau_elementwise[x.ngrid,:]./2.0
    else
        D[imax[j],imin[j]:imax[j]] .+= D_radau_elementwise[x.ngrid,:]
    end 
    # remaining elements recalling definitions of imax and imin
    for j in 2:x.nelement_local
        #lower boundary condition on element
        D[imin[j]-1,imin[j]-1:imax[j]] .+= D_lobotto_elementwise[1,:]./2.0
        for k in 2:imax[j]-imin[j]+1 
            D[k+imin[j]-2,imin[j]-1:imax[j]] .+= D_lobotto_elementwise[k,:]
        end
        # upper boundary condition on element 
        if j == x.nelement_local && !(zero_bc_upper_boundary)
            D[imax[j],imin[j]-1:imax[j]] .+= D_lobotto_elementwise[x.ngrid,:]
        elseif j == x.nelement_local && zero_bc_upper_boundary
            D[imax[j],imin[j]-1:imax[j]] .+= D_lobotto_elementwise[x.ngrid,:]./2.0 #contributions from this element/2
            D[imax[j],imax[j]] += D_lobotto_elementwise[1,1]/2.0 #contribution from missing `zero' element/2
        else 
            D[imax[j],imin[j]-1:imax[j]] .+= D_lobotto_elementwise[x.ngrid,:]./2.0
        end
    end
    
end

"""
function integrating d y / d t = f(t)
"""
function forward_euler_step!(ynew,yold,f,dt,n)
    for i in 1:n
        ynew[i] = yold[i] + dt*f[i]
    end
end
"""
function creating lu object for A = I - dt*nu*D2
"""
function diffusion_matrix(D2,n,dt,nu;return_A=false)
    A = allocate_float(n,n)
    for i in 1:n
        for j in 1:n
            A[i,j] = - dt*nu*D2[i,j]
        end
        A[i,i] += 1.0
    end
    lu_obj = lu(A)
    if return_A
        return lu_obj, A
    else
        return lu_obj
    end
end

"""
functions for Rosenbluth potential tests
"""
function dH_Maxwellian_dvpa(vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = sqrt(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)
    zero = 1.0e-10
    if eta < zero
        dHdvpa = -(4.0*vpa.grid[ivpa])/(3.0*sqrt(pi))
    else 
        dHdvpa = (vpa.grid[ivpa]/eta)*((2.0/sqrt(pi))*(exp(-eta^2)/eta)  - (erf(eta)/(eta^2)))
    end
    return dHdvpa
end
function d2H_Maxwellian_dvpa2(vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = sqrt(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)
    zero = 1.0e-10
    if eta < zero
        dHdeta_over_eta = -4.0/(3.0*sqrt(pi)) 
        d2Hdeta2 = -4.0/(3.0*sqrt(pi))
    else 
        dHdeta_over_eta = (2.0/sqrt(pi))*(exp(-eta^2)/eta^2)  - (erf(eta)/(eta^3))
        d2Hdeta2 = 2.0*(erf(eta)/(eta^3)) - (4.0/sqrt(pi))*(1.0 + (1.0/eta^2))*exp(-eta^2)
    end
    d2Hdvpa2 = ((vperp.grid[ivperp]^2)/(eta^2))*dHdeta_over_eta + ((vpa.grid[ivpa]^2)/(eta^2))*d2Hdeta2
    return d2Hdvpa2
end
function d2G_Maxwellian_dvpa2(vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = sqrt(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)
    zero = 1.0e-10
    if eta < zero
        dGdeta_over_eta = 4.0/(3.0*sqrt(pi)) 
        d2Gdeta2 = 4.0/(3.0*sqrt(pi))
    else 
        dGdeta_over_eta = (1.0/sqrt(pi))*(exp(-eta^2)/(eta^2)) + (1.0 - (0.5/eta^2))*erf(eta)/eta
        d2Gdeta2 = (erf(eta)/(eta^3)) - (2.0/sqrt(pi))*exp(-eta^2)/eta^2
    end
    d2Gdvpa2 = ((vperp.grid[ivperp]^2)/(eta^2))*dGdeta_over_eta + ((vpa.grid[ivpa]^2)/(eta^2))*d2Gdeta2
    return d2Gdvpa2
end

function dHdvpa_inf(vpa,vperp,ivpa,ivperp)
    eta = sqrt(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)
    dHdvpa_inf = -vpa.grid[ivpa]/eta^3
    return dHdvpa_inf
end
function d2Gdvpa2_inf(vpa,vperp,ivpa,ivperp)
    eta = sqrt(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)
    d2Gdvpa2_inf = ((vpa.grid[ivpa]^2)/eta^5) + ((vperp.grid[ivperp]^2)/eta^3)*( 1.0 - (0.5/eta^2))
    return d2Gdvpa2_inf
end


#using LinearAlgebra.mul
discretization = "chebyshev_pseudospectral"
#discretization = "finite_difference"
    etol = 1.0e-15
outprefix = "derivative_test"
    ###################
    ## df/dx Nonperiodic (No) BC test
    ###################
    
    # define inputs needed for the test
    ngrid = 17 #number of points per element 
    nelement_local = 20 # number of elements per rank
    nelement_global = nelement_local # total number of elements 
    L = 1.0 #physical box size in reference units 
    bc = "" #not required to take a particular value, not used 
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = "fourth_order_centered"
    nrank = 1
irank = 0
comm = MPI.COMM_NULL
# create the 'input' struct containing input info needed to create a
# coordinate
input = OptionsDict("coord" => OptionsDict("ngrid"=>ngrid,
                                           "nelement"=>nelement_global,
                                           "nelement_local"=>nelement_local, "L"=>L,
                                           "discretization"=>discretization,
                                           "finite_difference_option"=>fd_option,
                                           "bc"=>bc))
# create the coordinate struct 'x'
# This test runs effectively in serial, so use `ignore_MPI=true` to avoid
# errors due to communicators not being fully set up.
x, spectral = define_coordinate(input, "coord"; ignore_MPI=true)
    println("made x")
Dx = allocate_float(x.n, x.n)
xchebgrid = allocate_float(x.n)
for i in 1:x.n
    xchebgrid[i] = cos(pi*(i - 1)/(x.n - 1))
end
#println("x",xchebgrid[:])
cheb_derivative_matrix!(Dx,xchebgrid,x.n)
#println("")
#println("Dx \n")
#for i in 1:x.n
#    println(Dx[i,:])
#end

 # create array for the function f(x) to be differentiated/integrated
    f = allocate_float(x.n)
    # create array for the derivative df/dx
    df = allocate_float(x.n)
    df2 = allocate_float(x.n)
    df2cheb = allocate_float(x.n)
df_exact = allocate_float(x.n)
df2_exact = allocate_float(x.n)
df_err = allocate_float(x.n)
df2_err = allocate_float(x.n)
df2cheb_err = allocate_float(x.n)

for ix in 1:x.n
    f[ix] = sin(pi*xchebgrid[ix])
    df_exact[ix] = (pi)*cos(pi*xchebgrid[ix])
end
mul!(df,Dx,f)
for ix in 1:x.n
    df_err[ix] = df[ix]-df_exact[ix]
end
# test standard cheb D f = df 
#println("df \n",df)
#println("df_exact \n",df_exact)
#println("df_err \n",df_err)
input = OptionsDict("coord" => OptionsDict("ngrid"=>ngrid,
                                           "nelement"=>nelement_global,
                                           "nelement_local"=>nelement_local, "L"=>L,
                                           "discretization"=>discretization,
                                           "finite_difference_option"=>fd_option,
                                           "bc"=>"zero"))
# create the coordinate struct 'x'
# This test runs effectively in serial, so use `ignore_MPI=true` to avoid
# errors due to communicators not being fully set up.
x, spectral = define_coordinate(input, "coord"; ignore_MPI=true)

Dxreverse = allocate_float(x.n, x.n)
cheb_derivative_matrix_reversed!(Dxreverse,x)
Dxreverse2 = allocate_float(x.n, x.n)
mul!(Dxreverse2,Dxreverse,Dxreverse)
D2xreverse = allocate_float(x.n, x.n)
cheb_second_derivative_matrix_reversed!(D2xreverse,x)

Dxreverse2[1,1] = 2.0*Dxreverse2[1,1]
Dxreverse2[end,end] = 2.0*Dxreverse2[end,end]
#println("x.grid \n",x.grid)
if x.n < 20
    print_matrix(Dxreverse,"\n Dxreverse \n",x.n,x.n)
    print_matrix(Dxreverse2,"\n Dxreverse*Dxreverse \n",x.n,x.n)
    print_matrix(D2xreverse,"\n D2xreverse \n",x.n,x.n)
    println("\n")
end

alpha = 512.0    
for ix in 1:x.n
#        f[ix] = sin(2.0*pi*x.grid[ix]/x.L)
#        df_exact[ix] = (2.0*pi/x.L)*cos(2.0*pi*x.grid[ix]/x.L)
#        df2_exact[ix] = -(2.0*pi/x.L)*(2.0*pi/x.L)*sin(2.0*pi*x.grid[ix]/x.L)

    f[ix] = exp(-alpha*(x.grid[ix])^2)
    df_exact[ix] = -2.0*alpha*x.grid[ix]*exp(-alpha*(x.grid[ix])^2)
    df2_exact[ix] = ((2.0*alpha*x.grid[ix])^2 - 2.0*alpha)*exp(-alpha*(x.grid[ix])^2)
end
#println("test f: \n",f)
# calculate d f / d x from matrix 
mul!(df,Dxreverse,f)
# calculate d^2 f / d x from second application of Dx matrix 
mul!(df2,Dxreverse2,f)
# calculate d^2 f / d x from applition of D2x matrix 
mul!(df2cheb,D2xreverse,f)
for ix in 1:x.n
    df_err[ix] = df[ix]-df_exact[ix]
    df2_err[ix] = df2[ix]-df2_exact[ix]
    df2cheb_err[ix] = df2cheb[ix]-df2_exact[ix]
end
println("Reversed - multiple elements")
#println("df \n",df)
#println("df_exact \n",df_exact)
#println("df_err \n",df_err)
#println("df2 \n",df2)
#println("df2_exact \n",df2_exact)
#println("df2_err \n",df2_err)
#println("df2cheb_err \n",df2cheb_err)

println("max(df_err) \n",maximum(abs.(df_err)))
println("max(df2_err) \n",maximum(abs.(df2_err)))
println("max(df2cheb_err) \n",maximum(abs.(df2cheb_err)))

### attempt at matrix inversion via LU decomposition
Dt = 0.1
Nu = 1.0
lu_obj, AA = diffusion_matrix(Dxreverse2,x.n,Dt,Nu,return_A=true)
#AA = allocate_float(x.n,x.n)
#for i in 1:x.n
#    for j in 1:x.n
#        AA[i,j] = - Dt*Nu*Dxreverse2[i,j]
#    end
#    AA[i,i] += 1.0
#end
#lu_obj = lu(AA)
if x.n < 20
    println("L : \n",lu_obj.L)
    println("U : \n",lu_obj.U)
    println("p vector : \n",lu_obj.p)
end
LUtest = true
AA_test_lhs = lu_obj.L*lu_obj.U 
AA_test_rhs = AA[lu_obj.p,:]
for i in 1:x.n
    for j in 1:x.n
        if abs.(AA_test_lhs[i,j]-AA_test_rhs[i,j]) > zero
            global LUtest = false
        end
    end
end
println("LU == AA : \n",LUtest)

#bb = ones(x.n) try this for bc = "" rather than bc = "zero"
bb = allocate_float(x.n)
yy = allocate_float(x.n)
#for i in 1:x.n
#    bb[i] = f[i]#exp(-(4.0*x.grid[i]/x.L)^2)
#end
#yy = lu_obj \ bb # solution to AA yy = bb 
#println("result", yy)
#println("check result", AA*yy, bb)
MMS_test = false 
evolution_test = false#true 
elliptic_solve_test = false#true
elliptic_solve_1D_infinite_domain_test = false#true
elliptic_2Dsolve_test = true
if MMS_test
    ntest = 5
    MMS_errors = allocate_float(ntest)
    Dt_list = allocate_float(ntest)
    fac_list = allocate_int(ntest)
    fac_list .= [1, 10, 100, 1000, 10000]
    #for itest in [1, 10, 100, 1000, 10000]
    for itest in 1:ntest
        fac = fac_list[itest]
        #println(fac)
        ntime = 1000*fac
        nwrite = 100*fac
        dt = 0.001/fac
        #println(ntime," ",dt)
        nu = 1.0
        LU_obj = diffusion_matrix(Dxreverse2,x.n,dt,nu)
        
        time = allocate_float(ntime)
        ff = allocate_float(x.n,ntime)
        ss = allocate_float(x.n) #source

        time[1] = 0.0
        ff[:,1] .= f[:] #initial condition
        for i in 1:ntime-1
            time[i+1] = (i+1)*dt
            bb .= ff[:,i]
            yy .= LU_obj\bb # implicit backward euler diffusion step
            @. ss = -nu*df2_exact # source term
             # explicit forward_euler_step with source
            @views forward_euler_step!(ff[:,i+1],yy,ss,dt,x.n)
        end

        ff_error = allocate_float(x.n)
        ff_error[:] .= abs.(ff[:,end] - ff[:,1])
        maxfferr = maximum(ff_error)
        #println("ff_error \n",ff_error)
        println("max(ff_error) \n",maxfferr)
        #println("t[end]: ",time[end])
        MMS_errors[itest] = maxfferr
        Dt_list[itest] = dt
    end 
    @views plot(Dt_list, [MMS_errors, 100.0*Dt_list], label=[L"max(\epsilon(f))" L"100\Delta t"], 
                 xlabel=L"\Delta t", ylabel="", xscale=:log10, yscale=:log10, shape =:circle)
    outfile = string("ff_err_vs_dt.pdf")
    savefig(outfile)
end

if evolution_test
    ntime = 100
    nwrite = 1
    dt = 0.001
    nu = 1.0
    LU_obj = diffusion_matrix(Dxreverse2,x.n,dt,nu)
    
    time = allocate_float(ntime)
    ff = allocate_float(x.n,ntime)
    ss = allocate_float(x.n) #source

    time[1] = 0.0
    ff[:,1] .= f[:] #initial condition
    for i in 1:ntime-1
        time[i+1] = (i+1)*dt
        bb .= ff[:,i]
        yy .= LU_obj\bb # implicit backward euler diffusion step
        @. ss = 0.0 # source term
         # explicit forward_euler_step with source
        @views forward_euler_step!(ff[:,i+1],yy,ss,dt,x.n)
    end

    ffmin = minimum(ff)
    ffmax = maximum(ff)
    anim = @animate for i in 1:nwrite:ntime
            @views plot(x.grid, ff[:,i], xlabel="x", ylabel="f", ylims = (ffmin,ffmax))
        end
    outfile = string("ff_vs_x.gif")
    gif(anim, outfile, fps=5)
end

if elliptic_solve_test
    println("elliptic solve test")
    ngrid = 25
    nelement_local = 50
    L = 8
    nelement_global = nelement_local
    radau = true #false
    if radau        
        input = OptionsDict("vperp" => OptionsDict("ngrid"=>ngrid,
                                                   "nelement"=>nelement_global,
                                                   "nelement_local"=>nelement_local, "L"=>L,
                                                   "discretization"=>discretization,
                                                   "finite_difference_option"=>fd_option,
                                                   "bc"=>"zero_upper"))
        # create the coordinate struct 'x'
        # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
        # errors due to communicators not being fully set up.
        y, y_spectral = define_coordinate(input, "vperp"; ignore_MPI=true)
        Dy = allocate_float(y.n, y.n)
        cheb_radau_derivative_matrix_reversed!(Dy,y,y_spectral)
    else #lobotto
        input = OptionsDict("vpa" => OptionsDict("ngrid"=>ngrid,
                                                 "nelement"=>nelement_global,
                                                 "nelement_local"=>nelement_local,
                                                 "L"=>L,
                                                 "discretization"=>discretization,
                                                 "finite_difference_option"=>fd_option,
                                                 "bc"=>"zero_upper"))
        # create the coordinate struct 'x'
        # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
        # errors due to communicators not being fully set up.
        y, y_spectral = define_coordinate(input, "vpa"; ignore_MPI=true)
        @. y.grid += y.L/2
        Dy = allocate_float(y.n, y.n)
        cheb_derivative_matrix_reversed!(Dy,y)
    end  
    
    yDy = allocate_float(y.n, y.n)
    for iy in 1:y.n
        @. yDy[iy,:] = y.grid[iy]*Dy[iy,:]
    end
    
    
    Dy_yDy = allocate_float(y.n, y.n)
    mul!(Dy_yDy,Dy,yDy)
    #Dy_yDy[1,1] = 2.0*Dy_yDy[1,1]
    #Dy_yDy[end,end] = 2.0*Dy_yDy[end,end]
    
    D2y = allocate_float(y.n, y.n)
    mul!(D2y,Dy,Dy)
    #Dy_yDy[1,1] = 2.0*Dy_yDy[1,1]
    D2y[end,end] = 2.0*D2y[end,end]
    yD2y = allocate_float(y.n, y.n)
    for iy in 1:y.n
        @. yD2y[iy,:] = y.grid[iy]*D2y[iy,:]
    end
    
    if y.n < 20
        print_matrix(Dy,"Dy",y.n,y.n)
        print_matrix(yDy,"yDy",y.n,y.n)
        print_matrix(Dy_yDy,"Dy_yDy",y.n,y.n)
        print_matrix(yD2y+Dy,"yD2y+Dy",y.n,y.n)
    end 
    Sy = allocate_float(y.n)
    Fy = allocate_float(y.n)
    Fy_exact = allocate_float(y.n)
    Fy_err = allocate_float(y.n)
    for iy in 1:y.n
        #Sy[iy] = (y.grid[iy] - 1.0)*exp(-y.grid[iy])
        #Fy_exact[iy] = exp(-y.grid[iy])
        Sy[iy] = 4.0*y.grid[iy]*(y.grid[iy]^2 - 1.0)*exp(-y.grid[iy]^2)
        Fy_exact[iy] = exp(-y.grid[iy]^2)
    end
    LL = allocate_float(y.n, y.n)
    #@. LL = yD2y + Dy
    for iy in 1:y.n 
        #@. LL[iy,:] = Dy_yDy[iy,:] #*(1.0/y.grid[iy])
        @. LL[iy,:] = yD2y[iy,:] + Dy[iy,:] #*(1.0/y.grid[iy])
    end
    Dirichlet = true
    if Dirichlet
        # fixed value at orgin -- doesn't work well 
        #@. LL[1,:] = 0.0
        #Sy[1] = Fy_exact[1]
        set_flux = false 
        if set_flux
            # set flux at origin 
            @. LL[1,:] = 0.0
            ilim = y.imax[1]  
            @. LL[1,:] = yDy[ilim,:]
            
            print_vector(Sy,"Sy before",y.n)
            integrand = allocate_float(ilim)
            @. integrand[1:ilim] = Sy[1:ilim]*y.wgts[1:ilim]/(2.0*y.grid[1:ilim])
            
            print_vector(integrand,"integrand",ilim)
            print_vector(y.wgts,"wgts",y.n)
            #@. integrand[1:ilim] = y.grid[1:ilim]*Sy[1:ilim]*y.wgts[1:ilim]
            flux = sum(integrand)
            Sy[1] = flux
        end  
        # zero at infinity  
        @. LL[end,:] = 0.0
        LL[end,end] = 1.0
        #LL[1,1] = 1.0
#            @. LL[1,:] = 2.0*D2y[1,:] 
        Sy[end] = Fy_exact[end]
        
        #print_matrix(LL,"LL",y.n,y.n)
        #print_vector(Sy,"Sy",y.n)
    end
    
    #lu_solver = false
    #gauss_seidel_solver = true
    #if lu_solver
        println("det: ", det(LL))
        println("condition number: ", cond(LL))
        LL_lu_obj = lu(sparse(LL))
        
        # do elliptic solve 
        Fy = LL_lu_obj\Sy
    #elseif gauss_seidel_solver
    #    niter=100
    #    @. Fy[:] = Fy_exact[:] # initial guess
    #    gauss_seidel!(Fy,sparse(LL),Sy,maxiter=niter)
    #else 
    #    println("no solution method prescribed")
    #end 
    @. Fy_err = abs(Fy - Fy_exact)        
    println("maximum(Fy_err)",maximum(Fy_err))
    #println("Fy_err",Fy_err)
    #println("Fy_exact",Fy_exact)
    #println("Fy",Fy)
    plot([y.grid,y.grid,y.grid], [Fy,Fy_exact,Fy_err], xlabel="y", ylabel="", label=["F" "F_exact" "F_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1D_elliptic_solve_test.pdf"
    savefig(outfile)
    plot([y.grid], [Fy_err], xlabel="x", ylabel="", label=["F_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1D_elliptic_solve_test_err.pdf"
    savefig(outfile)

end

if elliptic_solve_1D_infinite_domain_test
    println("elliptic solve 1D infinite domain test")
    ngrid = 17
    nelement_local = 50
    L = 25
    nelement_global = nelement_local
    input = OptionsDict("vpa" => OptionsDict("ngrid"=>ngrid,
                                             "nelement"=>nelement_global,
                                             "nelement_local"=>nelement_local,
                                             "L"=>L,
                                             "discretization"=>discretization,
                                             "finite_difference_option"=>fd_option,
                                             "bc"=>"zero"))
    # create the coordinate struct 'x'
    # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
    # errors due to communicators not being fully set up.
    x, x_spectral = define_coordinate(input, "vpa"; ignore_MPI=true)
    Dx = allocate_float(x.n, x.n)
    cheb_derivative_matrix_reversed!(Dx,x)
    
    D2x = allocate_float(x.n, x.n)
    mul!(D2x,Dx,Dx)
    Dirichlet= true
    if Dirichlet
        # Dirichlet BC?
        @. D2x[1,:] = 0.0
        @. D2x[end,:] = 0.0
        D2x[1,1] = 1.0    
        D2x[end,end] = 1.0    
    else 
        # FD-like zero - BC
        D2x[1,1] = 2.0*D2x[1,1]
        D2x[end,end] = 2.0*D2x[end,end]
    end 
    
    if x.n < 20
        print_matrix(Dx,"Dx",x.n,x.n)
        print_matrix(D2x,"D2x",x.n,x.n)
    end 
    LLx = allocate_float(x.n, x.n)
    @. LLx = D2x
    
    Sx = allocate_float(x.n)
    Fx = allocate_float(x.n)
    Fx_exact = allocate_float(x.n)
    Fx_err = allocate_float(x.n)
    for ix in 1:x.n
        Sx[ix] = (4.0*x.grid[ix]^2 - 2.0)*exp(-x.grid[ix]^2)
        Fx_exact[ix] = exp(-x.grid[ix]^2)
    end
    # do elliptic solve 
    if Dirichlet 
        Sx[1] = 0.0; Sx[end] = 0.0 #Dirichlet BC values
    end 
    
    println("condition number: ", cond(LLx))
    LLx_lu_obj = lu(sparse(LLx)) 
    lu_solver = true#false
    #iterative_solver= false#true
    if lu_solver
        Fx = LLx_lu_obj\Sx
    #elseif iterative_solver
    #    niter=1000
    #    @. Fx[:] = 1.0/(x.grid[:]^8 + 1.0) # initial guess Fx_exact[:]
    #    Fx[1] =0.0; Fx[end] =0.0
    #    #gauss_seidel!(Fx,sparse(LLx),Sx,maxiter=niter)
    #    #jacobi!(Fx,sparse(LLx),Sx,maxiter=niter)
    #    #idrs!(Fx,sparse(LLx),Sx;abstol=10^(-10))
    else 
        println("no solution method prescribed")
    end
    @. Fx_err = abs(Fx - Fx_exact)
    
    println("test 1: maximum(Fx_err)",maximum(Fx_err))
    #println("Fx_err",Fx_err)
    #println("Fx_exact",Fx_exact)
    #println("Fx",Fx)
    plot([x.grid,x.grid,x.grid], [Fx,Fx_exact,Fx_err], xlabel="x", ylabel="", label=["F" "F_exact" "F_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1D_infinite_domain_elliptic_solve_test.pdf"
    savefig(outfile)
    plot([x.grid], [Fx_err], xlabel="x", ylabel="", label=["F_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1D_infinite_domain_elliptic_solve_test_err.pdf"
    savefig(outfile)
    
    for ix in 1:x.n
        Sx[ix] = exp(-x.grid[ix]^2)
        Fx_exact[ix] = (sqrt(pi)/2.0)*x.grid[ix]*erf(x.grid[ix]) + exp(-x.grid[ix]^2)/2.0
    end
    if Dirichlet 
        Sx[1] = 0.0; Sx[end] = 0.0 #Dirichlet BC values
    end 
    
    if lu_solver
        Fx = LLx_lu_obj\Sx
    elseif iterative_solver
        niter=1000
        @. Fx[:] = 1.0/(x.grid[:]^8 + 1.0) # initial guess Fx_exact[:]
        Fx[1] =0.0; Fx[end] =0.0
        #gauss_seidel!(Fx,sparse(LLx),Sx,maxiter=niter)
        #jacobi!(Fx,sparse(LLx),Sx,maxiter=niter)
        idrs!(Fx,sparse(LLx),Sx)
    else 
        println("no solution method prescribed")
    end
    
    @. Fx += (sqrt(pi)/2.0)*x.grid[end]
    @. Fx_err = abs(Fx - Fx_exact)
    println("test 2: maximum(Fx_err)",maximum(Fx_err))
    plot([x.grid], [Fx], xlabel="x", ylabel="", label=["Fx"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1D_infinite_domain_elliptic_solve_gaussian_source.pdf"
    savefig(outfile)
    plot([x.grid], [Fx_err], xlabel="x", ylabel="", label=["F_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1D_infinite_domain_elliptic_solve_gaussian_source_err.pdf"
    savefig(outfile)

end

if elliptic_2Dsolve_test
    println("elliptic 2D solve test")
    x_ngrid = 17
    x_nelement_local = 4
    x_L = 12
    y_L = 6
    y_ngrid = 17
    y_nelement_local = 2
    
    x_nelement_global = x_nelement_local
    y_nelement_global = y_nelement_local
    # bc option
    dirichlet_zero = true#false#
    dirichlet_fixed_value = false#true#
    # test option
    secular_decay_test = false#true
    exponential_decay_test = true#false
    dHdvpa_test = false 
    d2Gdvpa2_test = false#true 
    # second derivative option 
    # default = false -> if true then use D2coord matrices based on D_elementwise^2
    second_derivative_elementwise = false#true 
    
    input = OptionsDict("vpa" => OptionsDict("ngrid"=>x_ngrid,
                                             "nelement"=>x_nelement_global,
                                             "nelement_local"=>x_nelement_local,
                                             "L"=>L,
                                             "discretization"=>discretization,
                                             "finite_difference_option"=>fd_option,
                                             "bc"=>"zero"))
    # create the coordinate struct 'x'
    # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
    # errors due to communicators not being fully set up.
    x, x_spectral = define_coordinate(input, "vpa"; ignore_MPI=true)
    
    Dx = allocate_float(x.n, x.n)
    cheb_derivative_matrix_reversed!(Dx,x)
    D2x = allocate_float(x.n, x.n)
    if second_derivative_elementwise
        cheb_second_derivative_matrix_reversed!(D2x,x)
    else
        mul!(D2x,Dx,Dx)
    end
    # set x bc on D2x
    if dirichlet_zero
        D2x[1,1] = 2.0*D2x[1,1]
        D2x[end,end] = 2.0*D2x[end,end]
    elseif dirichlet_fixed_value
        @. D2x[1,:] = 0.0 
        @. D2x[end,:] = 0.0 
        D2x[1,1] = 1.0
        D2x[end,end] = 1.0
    end 
    
    IIx = allocate_float(x.n,x.n)
    @. IIx = 0.0
    for ix in 1:x.n
        IIx[ix,ix] = 1.0 
    end 
    
    if x.n < 20
        print_matrix(Dx,"Dx",x.n,x.n)
        print_matrix(D2x,"D2x",x.n,x.n)
        print_matrix(IIx,"IIx",x.n,x.n)
    end 
    
    input = OptionsDict("vperp" => OptionsDict("ngrid"=>y_ngrid,
                                               "nelement"=>y_nelement_global,
                                               "nelement_local"=>y_nelement_local,
                                               "L"=>L,
                                               "discretization"=>discretization,
                                               "finite_difference_option"=>fd_option,
                                               "bc"=>"zero_upper"))
    # create the coordinate struct 'x'
    # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
    # errors due to communicators not being fully set up.
    y, y_spectral = define_coordinate(input, "vperp"; ignore_MPI=true)
    Dy = allocate_float(y.n, y.n)
    cheb_radau_derivative_matrix_reversed!(Dy,y,y_spectral)
    
    D2y = allocate_float(y.n, y.n)
    if second_derivative_elementwise
        cheb_radau_second_derivative_matrix_reversed!(D2y,y,y_spectral)
    else
        mul!(D2y,Dy,Dy)
    end
    if dirichlet_zero
        D2y[end,end] = 2.0*D2y[end,end]
    end         
    # y derivative operator 
    LLy = allocate_float(y.n,y.n)
    for iy in 1:y.n 
        @. LLy[iy,:] = D2y[iy,:] + (1.0/y.grid[iy])*Dy[iy,:]
    end
    if dirichlet_fixed_value
        @. LLy[end,:] = 0.0
        LLy[end,end] = 1.0
    end
    IIy = allocate_float(y.n,y.n)
    @. IIy = 0.0
    for iy in 1:y.n
        IIy[iy,iy] = 1.0 
    end      
    if y.n < 20
        print_matrix(Dy,"Dy",y.n,y.n)
        print_matrix(D2y,"D2y",y.n,y.n)
        print_matrix(LLy,"LLy",y.n,y.n)
        print_matrix(IIy,"IIy",y.n,y.n)
    end 
    println("Initialised 1D arrays")
    ### now form 2D matrix to invert and corresponding sources 
    
    # Array in 2D form 
    nx = x.n   
    ny = y.n 
    Sxy = allocate_float(nx, ny)
    Sxy_check = allocate_float(nx, ny)
    Sxy_check_err = allocate_float(nx, ny)
    Txy = allocate_float(nx, ny)
    Fxy = allocate_float(nx, ny)
    Fxy_exact = allocate_float(nx, ny)
    Fxy_err = allocate_float(nx, ny)
    #LLxy = allocate_float(nx, ny, nx, ny)
    # Array in compound 1D form 
    # ic = (ix-1) + nx*(iy-1) + 1
    # iy = mod(ic,nx) + 1
    # ix = rem(ic,nx)
    function icfunc(ix,iy,nx)
        return ix + nx*(iy-1)
    end
    function iyfunc(ic,nx)
        #return mod(ic,nx) + 1
        return floor(Int64,(ic-1)/nx) + 1
    end
    function ixfunc(ic,nx)
        ix = ic - nx*(iyfunc(ic,nx) - 1)
        #return rem(ic,nx)
        return ix
    end
    function kronecker_delta(i,j)
        delta = 0.0 
        if i == j 
            delta = 1.0
        end
        return delta
    end
    nc = nx*ny
    Fc = allocate_float(nc)
    Sc = allocate_float(nc)
    LLc = allocate_float(nc, nc)
    
    if exponential_decay_test
        for iy in 1:ny
            for ix in 1:nx
                # Exponential test inputs below 
                Sxy[ix,iy] = ((4.0*x.grid[ix]^2 - 2.0) + (4.0*y.grid[iy]^2 - 4.0))*exp(-y.grid[iy]^2-x.grid[ix]^2)
                Fxy_exact[ix,iy] = exp(-x.grid[ix]^2 - y.grid[iy]^2) 
            end
        end
    elseif secular_decay_test
        for iy in 1:ny
            for ix in 1:nx
                # secular test inputs below
                eta2 = x.grid[ix]^2 + y.grid[iy]^2
                zero = 1.0e-10
                if eta2 < zero
                    Sxy[ix,iy] = 0.0
                    Fxy_exact[ix,iy] = 0.5
                else
                    Sxy[ix,iy] = exp(-1.0/eta2)*(1.0/eta2^2 - 2.0/eta2^3)
                    Fxy_exact[ix,iy] = 0.5 - 0.5*exp(-1.0/eta2)
                end
            end
        end
    elseif dHdvpa_test
        for iy in 1:ny
            for ix in 1:nx
                # Rosenbluth dHdvpa test 
                Sxy[ix,iy] = -(4.0/sqrt(pi))*(-2.0*x.grid[ix]*exp(-y.grid[iy]^2-x.grid[ix]^2))
                Fxy_exact[ix,iy] = dH_Maxwellian_dvpa(x,y,ix,iy)
            end
        end
    elseif d2Gdvpa2_test
        for iy in 1:ny
            for ix in 1:nx
                # Rosenbluth d2Gdvpa2 test 
                Sxy[ix,iy] = 2.0*d2H_Maxwellian_dvpa2(x,y,ix,iy)
                Fxy_exact[ix,iy] = d2G_Maxwellian_dvpa2(x,y,ix,iy)
                Txy[ix,iy] = 2.0*dH_Maxwellian_dvpa(x,y,ix,iy)
            end
            @views derivative!(Sxy_check[:,iy],Txy[:,iy],x,x_spectral)
        end
        @. Sxy_check_err = abs(Sxy - Sxy_check)
        println("maximum(Sxy_check_err)",maximum(Sxy_check_err))
        #@views heatmap(y.grid, x.grid, Sxy[:,:], xlabel=L"y", ylabel=L"x", c = :deep, interpolation = :cubic,
        #    windowsize = (360,240), margin = 15pt)
        #    outfile = string("Sxy_exact_2D_solve.pdf")
        #    savefig(outfile)
        #@views heatmap(y.grid, x.grid, Sxy_check[:,:], xlabel=L"y", ylabel=L"x", c = :deep, interpolation = :cubic,
        #    windowsize = (360,240), margin = 15pt)
        #    outfile = string("Sxy_num_2D_solve.pdf")
        #    savefig(outfile)
    else 
        println("No Sxy or Fxy_exact specified")
    end
    
    if dirichlet_fixed_value
        # set boundary values 
        if dHdvpa_test
            for ix in 1:nx
                Sxy[ix,ny] = dHdvpa_inf(x,y,ix,ny)  
            end
            for iy in 1:ny
                Sxy[1,iy] = dHdvpa_inf(x,y,1,iy)  
                Sxy[nx,iy] = dHdvpa_inf(x,y,nx,iy)  
            end
        elseif d2Gdvpa2_test
            for ix in 1:nx
                Sxy[ix,ny] = d2Gdvpa2_inf(x,y,ix,ny)  
            end
            for iy in 1:ny
                Sxy[1,iy] = d2Gdvpa2_inf(x,y,1,iy)  
                Sxy[nx,iy] = d2Gdvpa2_inf(x,y,nx,iy)  
            end            
        end
        println("Check boundary specification")
        #println(Sxy[:,ny])
        #println(Fxy_exact[:,ny])
        println(abs.(Sxy[:,ny] .- Fxy_exact[:,ny]))
        println(abs.(Sxy[1,:] .- Fxy_exact[1,:]))
        println(abs.(Sxy[nx,:] .- Fxy_exact[nx,:]))
        #println(Sxy[1,:])
        #println(Fxy_exact[1,:])
        #println(Sxy[nx,:])
        #println(Fxy_exact[nx,:])
    end
    # assign values to arrays in compound coordinates
    @. LLc = 0.0
    for ic in 1:nc
        ix = ixfunc(ic,nx)
        iy = iyfunc(ic,nx)
        Sc[ic] = Sxy[ix,iy]
        for icp in 1:nc
            ixp = ixfunc(icp,nx)
            iyp = iyfunc(icp,nx)
            #println("ic: ",ic," ix: ", ix," iy: ",iy," icp: ",icp," ixp: ", ixp," iyp: ",iyp)
            LLc[icp,ic] = D2x[ixp,ix]*IIy[iyp,iy] + LLy[iyp,iy]*IIx[ixp,ix]
        end
    end
    #set fixed bc in LLc directly
    if dirichlet_fixed_value
        ix = 1; ixp = 1
        for iyp in 1:ny
            for iy in 1:ny
                ic = icfunc(ix,iy,nx) 
                icp = icfunc(ixp,iyp,nx)
                LLc[icp,ic] = IIy[iyp,iy]                
            end            
        end
        ix = nx; ixp = nx
        for iyp in 1:ny
            for iy in 1:ny
                ic = icfunc(ix,iy,nx) 
                icp = icfunc(ixp,iyp,nx)
                LLc[icp,ic] = IIy[iyp,iy]                
            end            
        end
        iy = ny; iyp = ny
        for ixp in 1:nx
            for ix in 1:nx
                ic = icfunc(ix,iy,nx) 
                icp = icfunc(ixp,iyp,nx)
                LLc[icp,ic] = IIx[ixp,ix]                
            end            
        end
    end
    println("Initialised 2D arrays")
    if nc < 30
        print_matrix(LLc,"LLc",nc,nc)
    end 
    println("condition number(LLc): ", cond(LLc))
    println("determinant(LLc): ", det(LLc))
    LLc_lu_obj = lu(LLc)
    println("Initialised 2D solve") 
    # do elliptic solve 
    Fc = LLc_lu_obj\Sc
    #reshape to 2D vector 
    for ic in 1:nc
        ix = ixfunc(ic,nx)
        iy = iyfunc(ic,nx)
        Fxy[ix,iy] = Fc[ic]
    end
    #if dHdvpa_test && dirichlet_zero
    #    for iy in 1:ny
    #        for ix in 1:nx
    #            Fxy[ix,iy] += dHdvpa_inf(x,y,ix,iy)
    #        end
    #    end
    #end
    println("Finished 2D solve")
    @. Fxy_err = abs(Fxy - Fxy_exact)
    
    println("maximum(Fxy_err)",maximum(Fxy_err))
    #println("Fxy_err",Fxy_err[1,:])
    #println("Fxy_exact",Fxy_exact[1,:])
    #println("Fxy",Fxy[1,:])
    @views heatmap(y.grid, x.grid, Fxy_exact[:,:], xlabel=L"y", ylabel=L"x", c = :deep, interpolation = :cubic,
            windowsize = (360,240), margin = 15pt)
            outfile = string("Fxy_exact_2D_solve.pdf")
            savefig(outfile)
    @views heatmap(y.grid, x.grid, Fxy[:,:], xlabel=L"y", ylabel=L"x", c = :deep, interpolation = :cubic,
            windowsize = (360,240), margin = 15pt)
            outfile = string("Fxy_num_2D_solve.pdf")
            savefig(outfile)
    @views heatmap(y.grid, x.grid, Fxy_err[:,:], xlabel=L"y", ylabel=L"x", c = :deep, interpolation = :cubic,
            windowsize = (360,240), margin = 15pt)
            outfile = string("Fxy_err_2D_solve.pdf")
            savefig(outfile)
end
