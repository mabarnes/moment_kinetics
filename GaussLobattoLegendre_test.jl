export gausslegendre_test

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

    function gausslegendre_test(; ngrid=17, nelement=4, L_in=6.0)
        
        # elemental grid tests 
        #ngrid = 17
        #nelement = 4
        y_ngrid = ngrid #number of points per element 
        y_nelement_local = nelement # number of elements per rank
        y_nelement_global = y_nelement_local # total number of elements 
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
        for y_name in ["vpa","vperp"]
            println("")
            println("$y_name test")
            println("")
            if y_name == "vperp"
                y_L = L_in #physical box size in reference units 
            else 
                y_L = 2*L_in
            end
            y_input = grid_input(y_name, y_ngrid, y_nelement_global, y_nelement_local, 
                nrank, irank, y_L, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
            
            # create the coordinate structs
            y, y_spectral = define_coordinate(y_input)
            #print_matrix(Mmat,"Mmat",y.n,y.n)
            #print_matrix(y_spectral.radau.M0,"local radau mass matrix M0",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.radau.M1,"local radau mass matrix M1",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.lobatto.M0,"local mass matrix M0",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.lobatto.M1,"local mass matrix M1",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.mass_matrix,"global mass matrix",y.n,y.n)
            #print_matrix(y_spectral.lobatto.S0,"local S0 matrix",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.lobatto.S1,"local S1 matrix",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.S_matrix,"global S matrix",y.n,y.n)
            #print_matrix(y_spectral.radau.K0,"local radau K matrix K0",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.radau.K1,"local radau K matrix K1",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.lobatto.K0,"local K matrix K0",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.lobatto.K1,"local K matrix K1",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.radau.P0,"local radau P matrix P0",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.lobatto.P0,"local P matrix P0",y.ngrid,y.ngrid)
            #print_matrix(y_spectral.K_matrix,"global K matrix",y.n,y.n)
            #print_matrix(y_spectral.L_matrix,"global L matrix",y.n,y.n)
            #@views y_spectral.K_matrix[1,:] *= (4.0/3.0)
            #print_matrix(y_spectral.K_matrix,"global K matrix (hacked) ",y.n,y.n)
            #print_matrix(y_spectral.radau.Dmat,"local radau D matrix Dmat",y.ngrid,y.ngrid)
            #print_vector(y_spectral.radau.D0,"local radau D matrix D0",y.ngrid)
            #print_matrix(y_spectral.lobatto.Dmat,"local lobatto D matrix Dmat",y.ngrid,y.ngrid)
            #print_vector(y_spectral.lobatto.D0,"local lobatto D matrix D0",y.ngrid)
            
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
            if y.name == "vpa" 
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
            if y.name == "vpa"
                mul!(b,y_spectral.S_matrix,f_exact)
                #b[1] -= f_exact[1]
                #b[y.n] += f_exact[y.n]
                gausslegendre_mass_matrix_solve!(df_num,b,y.name,y_spectral)
                @. df_err = df_num - df_exact
                #println("df_num (weak form): ",df_num)
                #println("df_exact (weak form): ",df_exact)
                println("max(df_err) (weak form): ",maximum(df_err))
                #second_derivative!(d2f_num, f_exact, y, y_spectral)
                mul!(b,y_spectral.K_matrix,f_exact)
                #b[1] -= sum(y_spectral.lobatto.Dmat[1,:].*f_exact[1:y.ngrid])/y.element_scale[1]
                #b[y.n] += sum(y_spectral.lobatto.Dmat[y.ngrid,:].*f_exact[y.n+1-y.ngrid:y.n])/y.element_scale[y.nelement_local]
                gausslegendre_mass_matrix_solve!(d2f_num,b,y.name,y_spectral)
                @. d2f_err = abs(d2f_num - d2f_exact) #(0.5*y.L/y.nelement_global)*
                #println(d2f_num)
                #println(d2f_exact)
                println("max(d2f_err) (weak form): ",maximum(d2f_err))
                plot([y.grid, y.grid], [d2f_num, d2f_exact], xlabel="vpa", label=["num" "exact"], ylabel="")
                outfile = "vpa_test.pdf"
                savefig(outfile)
                
            elseif y.name == "vperp"
                #println("condition: ",cond(y_spectral.mass_matrix)) 
                mul!(b,y_spectral.S_matrix,g_exact)
                #b[y.n] += y.grid[y.n]*g_exact[y.n]    
                gausslegendre_mass_matrix_solve!(divg_num,b,y.name,y_spectral)
                @. divg_err = abs(divg_num - divg_exact)
                #println("divg_b (weak form): ",b)
                #println("divg_num (weak form): ",divg_num)
                #println("divg_exact (weak form): ",divg_exact)
                println("max(divg_err) (weak form): ",maximum(divg_err))
                
                #second_derivative!(d2f_num, f_exact, y, y_spectral)
                mul!(b,y_spectral.K_matrix,f_exact)
                #b[y.n] += y.grid[y.n]*sum(y_spectral.lobatto.Dmat[y.ngrid,:].*f_exact[y.n+1-y.ngrid:y.n])/y.element_scale[y.nelement_local]
                gausslegendre_mass_matrix_solve!(d2f_num,b,y.name,y_spectral)
                @. d2f_err = abs(d2f_num - d2f_exact) #(0.5*y.L/y.nelement_global)*
                #println(d2f_num)
                #println(d2f_exact)
                #println(d2f_err[1:10])
                println("max(d2f_err) (weak form): ",maximum(d2f_err))
                plot([y.grid, y.grid], [d2f_num, d2f_exact], xlabel="vpa", label=["num" "exact"], ylabel="")
                outfile = "vperp_second_derivative_test.pdf"
                savefig(outfile)
                 
                mul!(b,y_spectral.L_matrix,h_exact)
                #b[y.n] += y.grid[y.n]*sum(y_spectral.lobatto.Dmat[y.ngrid,:].*h_exact[y.n+1-y.ngrid:y.n])/y.element_scale[y.nelement_local]
                #laplacian_derivative!(laph_num, h_exact, y, y_spectral)
                gausslegendre_mass_matrix_solve!(laph_num,b,y.name,y_spectral)
                @. laph_err = abs(laph_num - laph_exact) #(0.5*y.L/y.nelement_global)*
                #println(b[1:10])
                #println(laph_num)
                #println(laph_exact)
                #println(laph_err[1:10])
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
    end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    gausslegendre_test()
end
