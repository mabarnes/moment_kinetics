using FastGaussQuadrature
using LegendrePolynomials: Pl
using LinearAlgebra: mul!, lu, inv, cond
using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

import moment_kinetics
using moment_kinetics.gauss_legendre
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.calculus: derivative!, second_derivative!, laplacian_derivative!

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

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

    # gauss lobatto test 
    ngrid = 4
    nelement = 1
    x, w = gausslobatto(ngrid)
    println("Gauss Lobatto Legendre")
    println("x: ",x)
    println("w: ",w)
    Lx = 2.0
    xx = (Lx/2.0)*copy(x)
    ww = (Lx/2.0)*copy(w)
    
    f_exact = Array{Float64,1}(undef,ngrid)
    df_exact = Array{Float64,1}(undef,ngrid)
    df_num = Array{Float64,1}(undef,ngrid)
    df_err = Array{Float64,1}(undef,ngrid)
    d2f_exact = Array{Float64,1}(undef,ngrid)
    d2f_num = Array{Float64,1}(undef,ngrid)
    d2f_err = Array{Float64,1}(undef,ngrid)
    
    for ix in 1:ngrid
        #f_exact = exp(-x[ix]^2)
        #df_exact = -2.0*x[ix]*exp(-x[ix]^2)
        
        #f_exact[ix] = -2.0*xx[ix]*exp(-xx[ix]^2)
        #df_exact[ix] = (4.0*xx[ix]^2 - 2.0)*exp(-xx[ix]^2)
        #d2f_exact[ix] = (12.0 - 8.0*xx[ix]^2)*xx[ix]*exp(-xx[ix]^2)
        f_exact[ix] = (xx[ix]^2 - 1.0)^2
        df_exact[ix] = 4.0*xx[ix]*(xx[ix]^2 - 1.0)
        d2f_exact[ix] = 12.0*xx[ix]^2 - 4.0
    end
    F_exact = f_exact[end] - f_exact[1]
    # do a test integration
    F_num = sum(ww.*df_exact)
    F_err = abs(F_num - F_exact)
    #for ix in 1:ngrid
    #    F_num += w[ix]*df_exact[ix]
    #end
    println("F_err: ", F_err,  " F_exact: ",F_exact, " F_num: ", F_num)
    
    Dmat = Array{Float64,2}(undef,ngrid,ngrid)
    Dmat2 = Array{Float64,2}(undef,ngrid,ngrid)
    D2mat = Array{Float64,2}(undef,ngrid,ngrid)
    Dmat_test = Array{Float64,2}(undef,ngrid,ngrid)
    Dmat_err = Array{Float64,2}(undef,ngrid,ngrid)
    gausslobattolegendre_differentiation_matrix!(Dmat,x,ngrid,Lx,1)
    mul!(Dmat2,Dmat,Dmat)
    #print_matrix(Dmat,"Dmat",ngrid,ngrid)
    
    Mmat = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_mass_matrix!(Mmat,ngrid,x,w,Lx,nelement)
    print_matrix(Mmat,"Mmat",ngrid,ngrid)
    
    Mmat_1 = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendre_mass_matrix_1!(Mmat_1,ngrid,x,w,Lx,nelement)
    print_matrix(Mmat_1,"Mmat_1",ngrid,ngrid)
    
    IMmat = Array{Float64,2}(undef,ngrid,ngrid)
    II_test = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_inverse_mass_matrix!(IMmat,ngrid,x,w,Lx)
    print_matrix(IMmat,"IMmat",ngrid,ngrid)
    
    mul!(II_test,Mmat,IMmat)
    print_matrix(II_test,"II_test",ngrid,ngrid)
    print_matrix(inv(Mmat),"inv(Mmat)",ngrid,ngrid)
    
    Smat = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_S_matrix!(Smat,ngrid,Dmat,w,Lx)
    print_matrix(Smat,"Smat",ngrid,ngrid)
    Smat_1 = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendre_S_matrix_1!(Smat_1,ngrid,x,w,Lx,nelement)
    print_matrix(Smat_1,"Smat_1",ngrid,ngrid)
    
    M0 = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendre_weak_product_matrix!(M0,ngrid,x,w,Lx,nelement,"M0")
    print_matrix(M0,"M0",ngrid,ngrid)
    M1 = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendre_weak_product_matrix!(M1,ngrid,x,w,Lx,nelement,"M1")
    print_matrix(M1,"M1",ngrid,ngrid)
    S0 = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendre_weak_product_matrix!(S0,ngrid,x,w,Lx,nelement,"S0")
    print_matrix(S0,"S0",ngrid,ngrid)
    S1 = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendre_weak_product_matrix!(S1,ngrid,x,w,Lx,nelement,"S1")
    print_matrix(S1,"S1",ngrid,ngrid)
    
    
    #mul!(Dmat_test,IMmat,Smat)
    mul!(Dmat_test,inv(Mmat),Smat)
    #@. Dmat_test = Mmat\Smat
    @. Dmat_err = abs(Dmat_test - Dmat )
    println("max_Dmat_err: ",maximum(Dmat_err))
    #print_matrix(Dmat_test,"Dmat_test",ngrid,ngrid)
    
    Kmat = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_K_matrix!(Kmat,ngrid,Dmat,w,Lx,nelement)
    #print_matrix(Kmat,"Kmat",ngrid,ngrid)
    mul!(D2mat,IMmat,Kmat)
    #print_matrix(D2mat,"D2mat",ngrid,ngrid)
    
    
    mul!(df_num,Dmat,f_exact)
    @. df_err = abs(df_num - df_exact)
    mul!(d2f_num,D2mat,f_exact)
    @. d2f_err = abs(d2f_num - d2f_exact)
    println(df_num)
    println(df_exact)
    println(d2f_num)
    println(d2f_exact)
    max_df_err = maximum(df_err)
    max_d2f_err = maximum(d2f_err)
    println("max df_err (Dmat): ",max_df_err)
    println("max d2f_err (weak D2mat): ",max_d2f_err)
    # test by differentiating twice
    mul!(d2f_num,Dmat2,f_exact)
    @. d2f_err = abs(d2f_num - d2f_exact)
    max_d2f_err = maximum(d2f_err)
   
    println("max d2f_err (Dmat*Dmat): ",max_d2f_err)
    println(d2f_num)
    println(d2f_exact)
      
    
    # gauss radau test 
    ngrid = 3
    xradau, wradau = gaussradau(ngrid)
    println("Gauss Radau Legendre")
    println("xradau: ",xradau)
    println("wradau: ",wradau)
    x = -reverse(xradau)
    w = reverse(wradau)
    println("x: ",x)
    println("w: ",w)
    
    f_exact = Array{Float64,1}(undef,ngrid)
    df_exact = Array{Float64,1}(undef,ngrid)
    df_num = Array{Float64,1}(undef,ngrid)
    df_err = Array{Float64,1}(undef,ngrid)
    
    for ix in 1:ngrid
        #f_exact = exp(-x[ix]^2)
        #df_exact = -2.0*x[ix]*exp(-x[ix]^2)
        
        f_exact[ix] = -2.0*x[ix]*exp(-x[ix]^2)
        df_exact[ix] = (4.0*x[ix]^2 - 2.0)*exp(-x[ix]^2)
    end
    F_exact = f_exact[end] - f_exact[1]
    # do a test integration
    F_num = sum(w.*df_exact)
    F_err = abs(F_num - F_exact)
    #for ix in 1:ngrid
    #    F_num += w[ix]*df_exact[ix]
    #end
    println("F_err: ", F_err,  " F_exact: ",F_exact, " F_num: ", F_num)
    
    Dmat = Array{Float64,2}(undef,ngrid,ngrid)
    gaussradaulegendre_differentiation_matrix!(Dmat,xradau,ngrid,2.0,1)
    #print_matrix(Dmat,"Dmat",ngrid,ngrid)
    
    mul!(df_num,Dmat,f_exact)
    @. df_err = abs(df_num - df_exact)
    #println(df_num)
    #println(df_exact)
    max_df_err = maximum(df_err)
    println("max df_err: ",max_df_err)
    
    # elemental grid tests 
    ngrid = 17
    nelement = 2
    y_ngrid = ngrid #number of points per element 
    y_nelement_local = nelement # number of elements per rank
    y_nelement_global = y_nelement_local # total number of elements 
    y_L = 6.0 #physical box size in reference units 
    bc = "zero" 
    discretization = "gausslegendre_pseudospectral"
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = "fourth_order_centered"
    cheb_option = "matrix"
    adv_input = advection_input("default", 1.0, 0.0, 0.0)
    nrank = 1
    irank = 0#1
    comm = MPI.COMM_NULL
    # create the 'input' struct containing input info needed to create a
    # coordinate
    #y_name = "y"
    y_name = "vperp"
    y_input = grid_input(y_name, y_ngrid, y_nelement_global, y_nelement_local, 
        nrank, irank, y_L, discretization, fd_option, cheb_option, bc, adv_input,comm)
    
    # create the coordinate structs
    y = define_coordinate(y_input)
    y_spectral = setup_gausslegendre_pseudospectral(y)
    Mmat = Array{Float64,2}(undef,y.ngrid,y.ngrid)
    x, w = gausslobatto(y.ngrid)
    #print_vector(y.grid,"y.grid",y.n)
    #print_vector(y.wgts,"y.wgts",y.n)
    GaussLegendreLobatto_mass_matrix!(Mmat,y.ngrid,x,w,y.L,y.nelement_global)
    #print_matrix(Mmat,"Mmat",y.ngrid,y.ngrid)
    print_matrix(y_spectral.radau.M0,"local radau mass matrix M0",y.ngrid,y.ngrid)
    print_matrix(y_spectral.radau.M1,"local radau mass matrix M1",y.ngrid,y.ngrid)
    print_matrix(y_spectral.lobatto.M0,"local mass matrix M0",y.ngrid,y.ngrid)
    print_matrix(y_spectral.lobatto.M1,"local mass matrix M1",y.ngrid,y.ngrid)
    #print_matrix(y_spectral.mass_matrix,"global mass matrix",y.n,y.n)
    print_matrix(y_spectral.lobatto.S0,"local S0 matrix",y.ngrid,y.ngrid)
    print_matrix(y_spectral.lobatto.S1,"local S1 matrix",y.ngrid,y.ngrid)
    #print_matrix(y_spectral.S_matrix,"global S matrix",y.n,y.n)
    print_matrix(y_spectral.radau.K0,"local radau K matrix K0",y.ngrid,y.ngrid)
    print_matrix(y_spectral.radau.K1,"local radau K matrix K1",y.ngrid,y.ngrid)
    print_matrix(y_spectral.lobatto.K0,"local K matrix K0",y.ngrid,y.ngrid)
    print_matrix(y_spectral.lobatto.K1,"local K matrix K1",y.ngrid,y.ngrid)
    print_matrix(y_spectral.radau.P0,"local radau P matrix P0",y.ngrid,y.ngrid)
    print_matrix(y_spectral.lobatto.P0,"local P matrix P0",y.ngrid,y.ngrid)
    #print_matrix(y_spectral.K_matrix,"global K matrix",y.n,y.n)
    #print_matrix(y_spectral.L_matrix,"global L matrix",y.n,y.n)
    #@views y_spectral.K_matrix[1,:] *= (4.0/3.0)
    #print_matrix(y_spectral.K_matrix,"global K matrix (hacked) ",y.n,y.n)
    print_matrix(y_spectral.radau.Dmat,"local radau D matrix Dmat",y.ngrid,y.ngrid)
    print_vector(y_spectral.radau.D0,"local radau D matrix D0",y.ngrid)
    print_matrix(y_spectral.lobatto.Dmat,"local lobatto D matrix Dmat",y.ngrid,y.ngrid)
    print_vector(y_spectral.lobatto.D0,"local lobatto D matrix D0",y.ngrid)
    
    Dmat = Array{Float64,2}(undef,y.ngrid,y.ngrid)
    Dmat_test = Array{Float64,2}(undef,y.ngrid,y.ngrid)
    Dmat_err = Array{Float64,2}(undef,y.ngrid,y.ngrid)
    lu_M0 = lu(y_spectral.lobatto.M0)
    mul!(Dmat_test,inv(y_spectral.lobatto.M0), y_spectral.lobatto.S0)
    #Dmat_test = y_spectral.lobatto.M0 \ y_spectral.lobatto.S0
    #print_matrix(lu_M0 \ y_spectral.lobatto.S0, "local D matrix",y.ngrid,y.ngrid)
    #print_matrix(Dmat_test, "local D matrix",y.ngrid,y.ngrid)
    mul!(Dmat_err,y_spectral.lobatto.M0,Dmat_test)
    #print_matrix(Dmat_err, "local S matrix?",y.ngrid,y.ngrid)
    #print_matrix(y_spectral.lobatto.Dmat, "local D matrix (Dmat)",y.ngrid,y.ngrid)
    
    @. Dmat = y_spectral.lobatto.Dmat
    
    @. Dmat_err = Dmat_test - Dmat
    #println("max_Dmat_err: ",maximum(Dmat_err))
    
    #print_matrix(y_spectral.S_matrix,"global S matrix",y.n,y.n)
    #print_matrix(y_spectral.lobatto.Mmat,"local lobatto mass matrix",y.ngrid,y.ngrid)
    #print_matrix(y_spectral.radau.Mmat,"local radau mass matrix",y.ngrid,y.ngrid)
    #println("y.grid: ",y.grid)
    #println("y.wgts: ",y.wgts)
    x, w = gausslobatto(y.ngrid)
    println("Gauss Lobatto Legendre")
    #println("x: ",x)
    #println("w: ",w)
    
    f_exact = Array{Float64,1}(undef,y.n)
    df_exact = Array{Float64,1}(undef,y.n)
    df_num = Array{Float64,1}(undef,y.n)
    df_err = Array{Float64,1}(undef,y.n)
    g_exact = Array{Float64,1}(undef,y.n)
    h_exact = Array{Float64,1}(undef,y.n)
    divg_exact = Array{Float64,1}(undef,y.n)
    divg_num = Array{Float64,1}(undef,y.n)
    divg_err = Array{Float64,1}(undef,y.n)
    laph_exact = Array{Float64,1}(undef,y.n)
    laph_num = Array{Float64,1}(undef,y.n)
    laph_err = Array{Float64,1}(undef,y.n)
    d2f_exact = Array{Float64,1}(undef,y.n)
    d2f_num = Array{Float64,1}(undef,y.n)
    d2f_err = Array{Float64,1}(undef,y.n)
    b = Array{Float64,1}(undef,y.n)
    for iy in 1:y.n
        f_exact[iy] = exp(-y.grid[iy]^2)
        df_exact[iy] = -2.0*y.grid[iy]*exp(-y.grid[iy]^2)
        d2f_exact[iy] = (4.0*y.grid[iy]^2 - 2.0)*exp(-y.grid[iy]^2)
        g_exact[iy] = y.grid[iy]*exp(-y.grid[iy]^2)
        divg_exact[iy] = 2.0*(1.0-y.grid[iy]^2)*exp(-y.grid[iy]^2)
        h_exact[iy] = exp(-y.grid[iy]^2)
        laph_exact[iy] = 4.0*(y.grid[iy]^2 - 1.0)*exp(-y.grid[iy]^2)
        #h_exact[iy] = exp(-2.0*y.grid[iy]^2)
        #laph_exact[iy] = 8.0*(2.0*y.grid[iy]^2 - 1.0)*exp(-2.0*y.grid[iy]^2)
        #h_exact[iy] = exp(-y.grid[iy]^3)
        #laph_exact[iy] = 9.0*y.grid[iy]*(y.grid[iy]^3 - 1.0)*exp(-y.grid[iy]^3)
        #f_exact[iy] = -2.0*y.grid[iy]*exp(-y.grid[iy]^2)
        
    end
    if y.name == "y" 
        F_exact = sqrt(pi)
    elseif y.name == "vperp"
        F_exact = 1.0
    end
    # do a test integration
    #println(f_exact)
    F_num = sum(y.wgts.*f_exact)
    F_err = abs(F_num - F_exact)
    #for ix in 1:ngrid
    #    F_num += w[ix]*df_exact[ix]
    #end
    println("F_err: ", F_err,  " F_exact: ",F_exact, " F_num: ", F_num)
    
    derivative!(df_num, f_exact, y, y_spectral)
    @. df_err = df_num - df_exact
    println("max(df_err) (interpolation): ",maximum(df_err))
    derivative!(d2f_num, df_num, y, y_spectral)
    @. d2f_err = d2f_num - d2f_exact
    println("max(d2f_err) (double first derivative by interpolation): ",maximum(d2f_err))  
    if y.name == "y"
        mul!(b,y_spectral.S_matrix,f_exact)
        gausslegendre_mass_matrix_solve!(df_num,b,y_spectral)
        @. df_err = df_num - df_exact
        #println("df_num (weak form): ",df_num)
        #println("df_exact (weak form): ",df_exact)
        println("max(df_err) (weak form): ",maximum(df_err))
        second_derivative!(d2f_num, f_exact, y, y_spectral)
        #mul!(b,y_spectral.K_matrix,f_exact)
        #gausslegendre_mass_matrix_solve!(d2f_num,b,y_spectral)
        @. d2f_err = abs(d2f_num - d2f_exact) #(0.5*y.L/y.nelement_global)*
        #println(d2f_num)
        #println(d2f_exact)
        println("max(d2f_err) (weak form): ",maximum(d2f_err))
        plot([y.grid, y.grid], [d2f_num, d2f_exact], xlabel="vpa", label=["num" "exact"], ylabel="")
        outfile = "vpa_test.pdf"
        savefig(outfile)
        
    elseif y.name == "vperp"
        #println("condition: ",cond(y_spectral.mass_matrix)) 
        b = Array{Float64,1}(undef,y.n)
        mul!(b,y_spectral.S_matrix,g_exact)
        gausslegendre_mass_matrix_solve!(divg_num,b,y_spectral)
        @. divg_err = abs(divg_num - divg_exact)
        #println("divg_b (weak form): ",b)
        #println("divg_num (weak form): ",divg_num)
        #println("divg_exact (weak form): ",divg_exact)
        println("max(divg_err) (weak form): ",maximum(divg_err))
        
        second_derivative!(d2f_num, f_exact, y, y_spectral)
        #mul!(b,y_spectral.K_matrix,f_exact)
        #gausslegendre_mass_matrix_solve!(d2f_num,b,y_spectral)
        @. d2f_err = abs(d2f_num - d2f_exact) #(0.5*y.L/y.nelement_global)*
        println(d2f_num[1:10])
        println(d2f_exact[1:10])
        println(d2f_err[1:10])
        println("max(d2f_err) (weak form): ",maximum(d2f_err))
        plot([y.grid, y.grid], [d2f_num, d2f_exact], xlabel="vpa", label=["num" "exact"], ylabel="")
        outfile = "vperp_second_derivative_test.pdf"
        savefig(outfile)
         
        #mul!(b,y_spectral.L_matrix,h_exact)
        laplacian_derivative!(laph_num, h_exact, y, y_spectral)
        #gausslegendre_mass_matrix_solve!(laph_num,b,y_spectral)
        @. laph_err = abs(laph_num - laph_exact) #(0.5*y.L/y.nelement_global)*
        #println(b[1:10])
        println(laph_num[1:10])
        println(laph_exact[1:10])
        println(laph_err[1:10])
        println("max(laph_err) (weak form): ",maximum(laph_err))
        plot([y.grid, y.grid], [laph_num, laph_exact], xlabel="vperp", label=["num" "exact"], ylabel="")
        outfile = "vperp_laplacian_test.pdf"
        savefig(outfile)
        
        @. y.scratch = y.grid*g_exact
        derivative!(y.scratch2, y.scratch, y, y_spectral)
        @. divg_num = y.scratch2/y.grid
        @. divg_err = abs(divg_num - divg_exact)
        println("max(divg_err) (interpolation): ",maximum(divg_err))
        
        derivative!(y.scratch, h_exact, y, y_spectral)
        @. y.scratch2 = y.grid*y.scratch
        derivative!(y.scratch, y.scratch2, y, y_spectral)
        @. laph_num = y.scratch/y.grid
        @. laph_err = abs(laph_num - laph_exact)
        println("max(laph_err) (interpolation): ",maximum(laph_err))
        
    end
    
   
end
