if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral
	import MPI 
	using Plots
	
	# define inputs needed for the xy calculus test
	ngrid = 20 #number of points per element 
	
	x_nelement_global = 2 # number of elements 
	y_nelement_global = 2 
	
	MPI.Init()
	comm = MPI.COMM_WORLD
	nrank = MPI.Comm_size(comm) # number of ranks 
	irank = MPI.Comm_rank(comm) # rank of this process
	#println("Hello world, I am $(irank) of $(nrank)")
	MPI.Barrier(comm)
	
	#require nrank to be a factor of y_nelement if nrank < y_nelement*x_nelement
	#require nrank to be a factor of y_nelement*x_nelement if nrank > y_nelement*x_nelement
	
	y_nelement_local = max(floor(Int,y_nelement_global/nrank),1)
	if y_nelement_local == 1
		x_nelement_local = max(floor(Int,y_nelement_global*x_nelement_global/nrank),1)
	else 
		x_nelement_local = x_nelement_global
	end
	
	if irank == 0
		println("ngrid = ",ngrid," x_nelement_local = ",x_nelement_local,
			" x_nelement_global = ",x_nelement_global," y_nelement_local = ",
			y_nelement_local," y_nelement_global = ",y_nelement_global,
			" nrank = ",nrank)
	end
	
	discretization = "chebyshev_pseudospectral"
	etol = 1.0e-15
	
	y_nblocks = y_nelement_global/y_nelement_local
	x_nblocks = x_nelement_global/x_nelement_local
	# let least parallelised dimension have MPI.COMM_WORLD
	# in this example, this is the x coordinate 
	# for each x block, we need a sub communicator for y 
	nrank_per_block = floor(Int,nrank/x_nblocks)
	color = floor(Int,irank/nrank_per_block)
	irank_sub = mod(irank,nrank_per_block)
	comm_sub = MPI.Comm_split(comm,color,irank_sub)
	
	L = 6.0 #physical box size in reference units 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = ""
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	# create the 'input' struct containing input info needed to create a
	# coordinate
	y_input = grid_input("coord", ngrid, y_nelement_global, y_nelement_local, 
		nrank_per_block, irank_sub, L, discretization, fd_option, bc, adv_input,comm_sub)
	x_input = grid_input("coord", ngrid, x_nelement_global, x_nelement_local, 
		nrank, irank, L, discretization, fd_option, bc, adv_input,comm)
	# create the coordinate struct 'x'
	#println("made inputs")
	y = define_coordinate(y_input)
	x = define_coordinate(x_input)
	#println("made x")
	# create arrays needed for Chebyshev pseudospectral treatment in x
	# and create the plans for the forward and backward fast Chebyshev
	# transforms
	y_spectral = setup_chebyshev_pseudospectral(y)
	x_spectral = setup_chebyshev_pseudospectral(x)
	
	# create a 2D array for the function f(x,y) to be differentiated/integrated
	f = Array{Float64,2}(undef, x.n, y.n)
	g = Array{Float64,2}(undef, x.n, y.n)
	h = Array{Float64,2}(undef, x.n, y.n)
	x_for_plot = Array{Float64,2}(undef, x.n, nrank)
	g_for_plot = Array{Float64,2}(undef, x.n, nrank)
	df_for_plot = Array{Float64,2}(undef, x.n, nrank)
	# create array for the derivative df/dx
	dfdx = Array{Float64,2}(undef, x.n, y.n)
	dfdy = Array{Float64,2}(undef, x.n, y.n)
	# initialize f
	
	println(x.grid,y.grid)
	for iy ∈ 1:y.n
		for ix ∈ 1:x.n
			f[ix,iy] =  sinpi(2.0*x.grid[ix]/x.L)*sinpi(2.0*y.grid[iy]/y.L) 
			g[ix,iy] =  (2.0*pi/x.L)*cospi(2.0*x.grid[ix]/x.L)*sinpi(2.0*y.grid[iy]/y.L)
			h[ix,iy] =  (2.0*pi/y.L)*sinpi(2.0*y.grid[iy]/y.L)*cospi(2.0*x.grid[ix]/x.L) 
		end
	end
	# differentiate f w.r.t x
	for iy in 1:y.n
		@views derivative!(dfdx[:,iy], f[:,iy], x, x_spectral)
	end
	# differentiate f w.r.t y
	for ix in 1:x.n
		@views derivative!(dfdy[ix,:], f[ix,:], y, y_spectral)
	end
	# plot df g h per process
	outprefix = "run_MPI_test2D.plot."
	for iy in 1:y.n
	plot([x.grid,x.grid], [g[:,iy],dfdx[:,iy]], xlabel="x", ylabel="", label=["g" "df"],
         shape =:circle, markersize = 5, linewidth=2)
	end
	outfile = outprefix*string(irank)*".pdf"
	savefig(outfile)
	
	#intdf = integral(df, x.wgts)
	#println(intdf)
	
	#intdf_out = MPI.Reduce(intdf,+,comm)
	
	# Test that error intdf is less than the specified error tolerance etol
	#@test abs(intdf) < etol
	
	#if(irank == 0)
		#println( "abs(intdf_out) = ", abs(intdf_out), ": etol = ",etol)
	#end
	
	
	
	MPI.Finalize()
end
	