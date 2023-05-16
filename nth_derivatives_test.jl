using MPI 
using Printf
using Plots
using LinearAlgebra: mul!

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral
	
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
    
    function Djk(x::Array{Float64,1},j::Int64,k::Int64,c_j::Float64,c_k::Float64)
        return  (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
    end
    
    function cheb_derivative_matrix_reversed!(D::Array{Float64,2},x) 
        D_elementwise = Array{Float64,2}(undef,x.ngrid,x.ngrid)
        cheb_derivative_matrix_elementwise_reversed!(D_elementwise,x.ngrid,x.L,x.nelement_global)    
        if x.ngrid < 8
            print_matrix(D_elementwise,"D_elementwise",x.ngrid,x.ngrid)
        end 
        assign_cheb_derivative_matrix!(D,D_elementwise,x)
    end
    
    function assign_cheb_derivative_matrix!(D::Array{Float64,2},D_elementwise::Array{Float64,2},x) 
        
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
    
    function cheb_derivative_matrix_elementwise_reversed!(D::Array{Float64,2},n::Int64,L::Float64,nelement::Int64) 
        
        #define Chebyshev points in reversed order x_j = { -1, ... , 1}
        x = Array{Float64,1}(undef,n)
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
    testopt_sine = "sinewavesegment"
    testopt_poly = "xpower5"
    testopt = testopt_poly
    #testopt = testopt_sine
    if testopt == testopt_poly
        L = 1.0 
    elseif testopt == testopt_sine
        L = 2.0*pi/100.0 #physical box size in reference units 
	end
	
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	etol = 1.0e-15
    outprefix = "derivative_test"
	###################
	## df/dx Nonperiodic (No) BC test
	###################
	
	# define inputs needed for the test
	ngrid = 33 #number of points per element 
	nelement_local = 1 # number of elements per rank
	nelement_global = nelement_local # total number of elements 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = "fourth_order_centered"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = MPI.COMM_NULL
	# create the 'input' struct containing input info needed to create a
	# coordinate
    input = grid_input("coord", ngrid, nelement_global, nelement_local, 
		nrank, irank, L, discretization, fd_option, bc, adv_input,comm)
	# create the coordinate struct 'x'
	println("testopt: ", testopt)
	println("made inputs: ngrid: ",ngrid, " nelement: ", nelement_local, " L: ",L)
	x = define_coordinate(input)
	println("made x")
	# create arrays needed for Chebyshev pseudospectral treatment in x
	# and create the plans for the forward and backward fast Chebyshev
	# transforms
	if discretization == "chebyshev_pseudospectral"
        spectral = setup_chebyshev_pseudospectral(x)
	else
        spectral = false
    end
    println("made spectral")
	# create array for the function f(x) to be differentiated/integrated
	f = Array{Float64,1}(undef, x.n)
	# create array for the derivative df/dx
	df = Array{Float64,1}(undef, x.n)
    df_exact = Array{Float64,1}(undef, x.n)
    df_err = Array{Float64,1}(undef, x.n)
    d2f = Array{Float64,1}(undef, x.n)
    d2f_exact = Array{Float64,1}(undef, x.n)
    d2f_err = Array{Float64,1}(undef, x.n)
    d3f = Array{Float64,1}(undef, x.n)
    d3f_exact = Array{Float64,1}(undef, x.n)
    d3f_err = Array{Float64,1}(undef, x.n)
    d4f = Array{Float64,1}(undef, x.n)
    d4f_exact = Array{Float64,1}(undef, x.n)
    d4f_err = Array{Float64,1}(undef, x.n)

    if testopt == testopt_poly
        p = 5.0
        for ix in 1:x.n
            f[ix] = x.grid[ix]^p
            df_exact[ix] = p*x.grid[ix]^(p-1.0)
            d2f_exact[ix] = p*(p-1.0)*x.grid[ix]^(p-2.0)
            d3f_exact[ix] = p*(p-1.0)*(p-2.0)*x.grid[ix]^(p-3.0)
            d4f_exact[ix] = p*(p-1.0)*(p-2.0)*(p-3.0)*x.grid[ix]^(p-4.0)
        end

    elseif testopt == testopt_sine
        ## sin(2pix/L) test 
        for ix in 1:x.n
            scale = 1.0
            arg = x.grid[ix]*scale
            f[ix] = sin(arg)
            df_exact[ix] = scale*cos(arg)    
            d2f_exact[ix] = -scale*scale*sin(arg)    
            d3f_exact[ix] = -scale*scale*scale*cos(arg)    
            d4f_exact[ix] = scale*scale*scale*scale*sin(arg)    
        end
    end
    # differentiate f
    derivative!(df, f, x, spectral)
    derivative!(d2f, df, x, spectral)
    derivative!(d3f, d2f, x, spectral)
    derivative!(d4f, d3f, x, spectral)

    @. df_err = abs(df - df_exact)
    @. d2f_err = abs(d2f - d2f_exact)
    @. d3f_err = abs(d3f - d3f_exact)
    @. d4f_err = abs(d4f - d4f_exact)
    println("FFT differentiation")
    println("max(df_err)",maximum(df_err))
    println("max(d2f_err)",maximum(d2f_err))
    println("max(d3f_err)",maximum(d3f_err))
    println("max(d4f_err)",maximum(d4f_err))
    
    plot_output = true
    if plot_output
        # plot df and f
        plot([x.grid,x.grid,x.grid], [df,df_exact,df_err], xlabel="x", ylabel="", label=["df_num" "df_exact" "df_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "1st_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [df_err], xlabel="x", ylabel="", label=["df_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "1st_derivative_err.pdf"
        savefig(outfile)
        
        plot([x.grid,x.grid,x.grid], [d2f,d2f_exact,d2f_err], xlabel="x", ylabel="", label=["d2f_num" "d2f_exact" "d2f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "2nd_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [d2f_err], xlabel="x", ylabel="", label=["d2f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "2nd_derivative_err.pdf"
        savefig(outfile)
        
        plot([x.grid,x.grid,x.grid], [d3f,d3f_exact,d3f_err], xlabel="x", ylabel="", label=["d3f_num" "d3f_exact" "d3f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "3rd_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [d3f_err], xlabel="x", ylabel="", label=["d3f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "3rd_derivative_err.pdf"
        savefig(outfile)
        
        plot([x.grid,x.grid,x.grid], [d4f,d4f_exact,d4f_err], xlabel="x", ylabel="", label=["d4f_num" "d4f_exact" "d4f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "4th_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [d4f_err], xlabel="x", ylabel="", label=["d4f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "4th_derivative_err.pdf"
        savefig(outfile)
    end
    
    # derivative matrix approach
    Dx = Array{Float64,2}(undef, x.n, x.n)
    cheb_derivative_matrix_reversed!(Dx,x)
    # differentiate f 
    mul!(df,Dx,f)
    mul!(d2f,Dx,df)
    mul!(d3f,Dx,d2f)
    mul!(d4f,Dx,d3f)
    @. df_err = abs(df - df_exact)
    @. d2f_err = abs(d2f - d2f_exact)
    @. d3f_err = abs(d3f - d3f_exact)
    @. d4f_err = abs(d4f - d4f_exact)
    println("matrix differentiation")
    println("max(df_err)",maximum(df_err))
    println("max(d2f_err)",maximum(d2f_err))
    println("max(d3f_err)",maximum(d3f_err))
    println("max(d4f_err)",maximum(d4f_err))
    
    plot_output = true
    if plot_output
        # plot df and f
        plot([x.grid,x.grid,x.grid], [df,df_exact,df_err], xlabel="x", ylabel="", label=["df_num" "df_exact" "df_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "1st_matrix_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [df_err], xlabel="x", ylabel="", label=["df_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "1st_matrix_derivative_err.pdf"
        savefig(outfile)
        
        plot([x.grid,x.grid,x.grid], [d2f,d2f_exact,d2f_err], xlabel="x", ylabel="", label=["d2f_num" "d2f_exact" "d2f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "2nd_matrix_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [d2f_err], xlabel="x", ylabel="", label=["d2f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "2nd_matrix_derivative_err.pdf"
        savefig(outfile)
        
        plot([x.grid,x.grid,x.grid], [d3f,d3f_exact,d3f_err], xlabel="x", ylabel="", label=["d3f_num" "d3f_exact" "d3f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "3rd_matrix_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [d3f_err], xlabel="x", ylabel="", label=["d3f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "3rd_matrix_derivative_err.pdf"
        savefig(outfile)
        
        plot([x.grid,x.grid,x.grid], [d4f,d4f_exact,d4f_err], xlabel="x", ylabel="", label=["d4f_num" "d4f_exact" "d4f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "4th_matrix_derivative_test.pdf"
        savefig(outfile)
        plot([x.grid], [d4f_err], xlabel="x", ylabel="", label=["d4f_err"],
             shape =:circle, markersize = 5, linewidth=2)
        outfile = "4th_matrix_derivative_err.pdf"
        savefig(outfile)
    end
    
end
	
