if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral #derivative_handle_wall_bc!
	using Plots
    
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	etol = 1.0e-15
    outprefix = "derivative_test"
	###################
	## df/dx Nonperiodic (No) BC test
	###################
	
	# define inputs needed for the test
	ngrid = 17 #number of points per element 
	nelement_local = 4 # number of elements per rank
	nelement_global = nelement_local # total number of elements 
	L = 1.0 #physical box size in reference units 
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = "fourth_order_centered"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = false
	# create the 'input' struct containing input info needed to create a
	# coordinate
    x_input = grid_input("coord", ngrid, nelement_global, nelement_local, 
		nrank, irank, L, discretization, fd_option, "", adv_input,comm)
	z_input = grid_input("coord", ngrid, nelement_global, nelement_local, 
		nrank, irank, L, discretization, fd_option, "wall", adv_input,comm)
	# create the coordinate struct 'x'
	println("made inputs")
	x = define_coordinate(x_input)
	z = define_coordinate(z_input)
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
	df_opt = Array{Float64,1}(undef, x.n)
    df_exact = Array{Float64,1}(undef, x.n)
    vz = Array{Float64,1}(undef,x.n)
    

    xcut = 0.125
    teststring = ".x2."*string(xcut)
    ## x^2 test 
    for ix in 1:x.n
        vz[ix] = x.grid[ix] - xcut 
        if x.grid[ix] > xcut
            f[ix] = (x.grid[ix] - xcut)^2
            df_exact[ix] = 2.0*(x.grid[ix] - xcut)
        else
            f[ix] = 0.0
            df_exact[ix] = 0.0
        end
    end
        
    # differentiate f using standard Chebyshev method 
    derivative!(df, f, x, spectral)
    # differentiate f using Chebyshev with spline for subgrid differentiation
    iz = 1 # a wall boundary point
    derivative!(df_opt, f, x, spectral, iz, z, vz)
    
    # plot df, df_opt & df_exact 
    plot([x.grid,x.grid,x.grid], [vz, f,df_opt,df_exact], xlabel="x", ylabel="", label=["vz" "f" "df_opt" "df_exact"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = outprefix*teststring*".onlyspline.pdf"
    savefig(outfile)
    plot([x.grid,x.grid,x.grid], [f,df,df_opt,df_exact], xlabel="x", ylabel="", label=["f" "df_cheb" "df_opt" "df_exact"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = outprefix*teststring*".withspline.pdf"
    savefig(outfile)
    plot([x.grid,x.grid,x.grid], [f,df,df_exact], xlabel="x", ylabel="", label=["f" "df_cheb" "df_exact"],
         shape =:circle, markersize = 5, linewidth=2)
    outfile = outprefix*teststring*".pdf"
    savefig(outfile)

end
	
