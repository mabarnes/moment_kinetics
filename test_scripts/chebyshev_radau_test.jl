export chebyshevradau_test

using Printf
using MPI

import moment_kinetics
using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.chebyshev
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.calculus: derivative!
using moment_kinetics.type_definitions: OptionsDict


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
function chebyshevradau_test(; ngrid=5, L_in=3.0, discretization="chebyshev_pseudospectral")

    # elemental grid tests 
    #ngrid = 17
    #nelement = 4
    y_ngrid = ngrid #number of points per element 
    y_nelement_local = 1 # number of elements per rank
    y_nelement_global = y_nelement_local # total number of elements 
    y_L = L_in
    bc = "zero" 
    fd_option = "fourth_order_centered"
    cheb_option = "matrix"
    nrank = 1
    irank = 0#1
    comm = MPI.COMM_NULL
    element_spacing_option = "uniform"
    # create the 'input' struct containing input info needed to create a
    # coordinate
    y_name = "vperp" # to use radau grid
    input = OptionsDict(y_name => OptionsDict("ngrid"=>y_ngrid, "nelement"=>y_nelement_global,
                                              "nelement_local"=>y_nelement_local, "L"=>y_L,
                                              "discretization"=>discretization,
                                              "finite_difference_option"=>fd_option,
                                              "cheb_option"=>cheb_option, "bc"=>bc,
                                              "element_spacing_option"=>element_spacing_option))
    # create the coordinate struct 'x'
    # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
    # errors due to communicators not being fully set up.
    y, y_spectral = define_coordinate(input, y_name; ignore_MPI=true)
      
    Dmat = y_spectral.radau.Dmat
    print_matrix(Dmat,"Radau Dmat",y.ngrid,y.ngrid)
    D0 = y_spectral.radau.D0
    print_vector(D0,"Radau D0",y.ngrid)
    
    ff_err = allocate_float(y.n)
    ff = allocate_float(y.n)
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
    f0 = -sum(D0[2:ngrid].*ff[2:ngrid])/D0[1]
    println("exact f[0]: ",ff[1]," num f[0]: ",f0," abs(err): ",abs(f0-ff[1]))
    
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
