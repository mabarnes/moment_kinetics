using MPI 

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral
	#import MPI 
	using Plots
    
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	etol = 1.0e-15
    outprefix = "derivative_test"
	###################
	## df/dx Nonperiodic (No) BC test
	###################
	
	# define inputs needed for the test
	ngrid = 33 #number of points per element 
	nelement_local = 100 # number of elements per rank
	nelement_global = nelement_local # total number of elements 
	L = 1.0 #physical box size in reference units 
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
	println("made inputs")
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

    ## sin(2pix/L) test 
    for ix in 1:x.n
        scale = 2.0*pi/x.L
        arg = x.grid[ix]*scale
        f[ix] = sin(arg)
        df_exact[ix] = scale*cos(arg)    
        d2f_exact[ix] = -scale*scale*sin(arg)    
        d3f_exact[ix] = -scale*scale*scale*cos(arg)    
        d4f_exact[ix] = scale*scale*scale*scale*sin(arg)    
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
    println("max(df_err)",maximum(df_err))
    println("max(d2f_err)",maximum(d2f_err))
    println("max(d3f_err)",maximum(d3f_err))
    println("max(d4f_err)",maximum(d4f_err))
    
    # plot df and f
    plot([x.grid,x.grid,x.grid], [df,df_exact,df_err], xlabel="x", ylabel="", label=["df_num" "df_exact" "df_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "1st_derivative_test.pdf"
    savefig(outfile)
    
    plot([x.grid,x.grid,x.grid], [d2f,d2f_exact,d2f_err], xlabel="x", ylabel="", label=["d2f_num" "d2f_exact" "d2f_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "2nd_derivative_test.pdf"
    savefig(outfile)
    
    plot([x.grid,x.grid,x.grid], [d3f,d3f_exact,d3f_err], xlabel="x", ylabel="", label=["d3f_num" "d3f_exact" "d3f_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "3rd_derivative_test.pdf"
    savefig(outfile)
    
    plot([x.grid,x.grid,x.grid], [d4f,d4f_exact,d4f_err], xlabel="x", ylabel="", label=["d4f_num" "d4f_exact" "d4f_err"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = "4th_derivative_test.pdf"
    savefig(outfile)
    
end
	
