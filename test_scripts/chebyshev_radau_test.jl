export chebyshevradau_test

using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

import moment_kinetics
using moment_kinetics.chebyshev
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.calculus: derivative!


function print_matrix(matrix,name,n,m)
    println("\n ",name," \n")
    for i in 1:n
        for j in 1:m
            @printf("%.3f ", matrix[i,j])
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
"""
Test for derivative vector D0 that is used to compute 
the numerical derivative on the Chebyshev-Radau elements
at the lower endpoint of the domain (-1,1] in the normalised
coordinate x. Here in the tests the shifted coordinate y 
is used with the vperp label so that the grid runs from (0,L].
"""
function chebyshevradau_test(; ngrid=5, L_in=3.0)

    # elemental grid tests 
    #ngrid = 17
    #nelement = 4
    y_ngrid = ngrid #number of points per element 
    y_nelement_local = 1 # number of elements per rank
    y_nelement_global = y_nelement_local # total number of elements 
    y_L = L_in
    bc = "zero" 
    discretization = "gausslegendre_pseudospectral"
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = "fourth_order_centered"
    cheb_option = "matrix"
    adv_input = advection_input("default", 1.0, 0.0, 0.0)
    nrank = 1
    irank = 0#1
    comm = MPI.COMM_NULL
    element_spacing_option = "uniform"
    # create the 'input' struct containing input info needed to create a
    # coordinate
    y_name = "vperp" # to use radau grid
    y_input = grid_input(y_name, y_ngrid, y_nelement_global, y_nelement_local, 
            nrank, irank, y_L, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
    y, y_spectral = define_coordinate(y_input)
      
    Dmat = y_spectral.radau.Dmat
    print_matrix(Dmat,"Radau Dmat",y.ngrid,y.ngrid)
    D0 = y_spectral.radau.D0
    print_vector(D0,"Radau D0",y.ngrid)
    
    ff_err = Array{Float64,1}(undef,y.n)
    ff = Array{Float64,1}(undef,y.n)
    for iy in 1:y.n
        ff[iy] = exp(-4.0*y.grid[iy])
    end
    df_exact = -4.0
    df_num = sum(D0.*ff)/y.element_scale[1]
    df_err = abs(df_num - df_exact)
    println("f(y) = exp(-4 y) test")
    println("exact df: ",df_exact," num df: ",df_num," abs(err): ",df_err) 
    
    for iy in 1:y.n
        ff[iy] = exp(-y.grid[iy]^2)
    end
    df_exact = 0.0
    df_num = sum(D0.*ff)/y.element_scale[1]
    df_err = abs(df_num - df_exact)
    println("f(y) = exp(-y^2) test")
    println("exact df: ",df_exact," num df: ",df_num," abs(err): ",df_err) 
    
    for iy in 1:y.n
        ff[iy] = sin(y.grid[iy])
    end
    df_exact = 1.0
    df_num = sum(D0.*ff)/y.element_scale[1]
    df_err = abs(df_num - df_exact)
    println("f(y) = sin(y) test")
    println("exact df: ",df_exact," num df: ",df_num," abs(err): ",df_err) 
    
    for iy in 1:y.n
        ff[iy] = y.grid[iy] + (y.grid[iy])^2 + (y.grid[iy])^3
    end
    df_exact = 1.0
    df_num = sum(D0.*ff)/y.element_scale[1]
    df_err = abs(df_num - df_exact)
    println("f(y) = y + y^2 + y^3 test")
    println("exact df: ",df_exact," num df: ",df_num," abs(err): ",df_err) 
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    chebyshevradau_test()
end
