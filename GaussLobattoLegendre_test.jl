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
