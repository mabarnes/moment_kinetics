using FastGaussQuadrature
using LegendrePolynomials: Pl
using LinearAlgebra: mul!
using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

import moment_kinetics
using moment_kinetics.gausslegendre

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
    ngrid = 5
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
    GaussLegendreLobatto_mass_matrix!(Mmat,ngrid,x,w,Lx)
    #print_matrix(Mmat,"Mmat",ngrid,ngrid)
    
    IMmat = Array{Float64,2}(undef,ngrid,ngrid)
    II_test = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_inverse_mass_matrix!(IMmat,ngrid,x,w,Lx)
    #print_matrix(IMmat,"IMmat",ngrid,ngrid)
    
    mul!(II_test,Mmat,IMmat)
    #print_matrix(II_test,"II_test",ngrid,ngrid)
    
    Smat = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_S_matrix!(Smat,ngrid,Dmat,w,Lx)
    #print_matrix(Smat,"Smat",ngrid,ngrid)
    mul!(Dmat_test,IMmat,Smat)
    @. Dmat_err = abs(Dmat_test - Dmat )
    println("max_Dmat_err: ",maximum(Dmat_err))
    #print_matrix(Dmat_test,"Dmat_test",ngrid,ngrid)
    
    Kmat = Array{Float64,2}(undef,ngrid,ngrid)
    GaussLegendreLobatto_K_matrix!(Kmat,ngrid,Dmat,w,Lx)
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
    ngrid = 9
    x, w = gaussradau(ngrid)
    println("Gauss Radau Legendre")
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
    gaussradaulegendre_differentiation_matrix!(Dmat,x,ngrid,2.0,1)
    print_matrix(Dmat,"Dmat",ngrid,ngrid)
    
    mul!(df_num,Dmat,f_exact)
    @. df_err = abs(df_num - df_exact)
    #println(df_num)
    #println(df_exact)
    max_df_err = maximum(df_err)
    println("max df_err: ",max_df_err)
end
