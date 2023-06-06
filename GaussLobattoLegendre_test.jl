using FastGaussQuadrature
using LegendrePolynomials: Pl
using LinearAlgebra: mul!
using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

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
    """
    Formula for differentiation matrix taken from p196 of Chpt `The Spectral Elemtent Method' of 
    `Computational Seismology'. Heiner Igel First Edition. Published in 2017 by Oxford University Press.
    Or https://doc.nektar.info/tutorials/latest/fundamentals/differentiation/fundamentals-differentiationch2.html
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

    # gauss lobatto test 
    ngrid = 33
    x, w = gausslobatto(ngrid)
    println("Gauss Lobatto Legendre")
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
    gausslobattolegendre_differentiation_matrix!(Dmat,x,ngrid,2.0,1)
    print_matrix(Dmat,"Dmat",ngrid,ngrid)
    
    mul!(df_num,Dmat,f_exact)
    @. df_err = abs(df_num - df_exact)
    #println(df_num)
    #println(df_exact)
    max_df_err = maximum(df_err)
    println("max df_err: ",max_df_err)
    
    # gauss radau test 
    ngrid = 33
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
