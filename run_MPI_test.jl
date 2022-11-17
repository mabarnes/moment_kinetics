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
	MPI.Init()
	comm = MPI.COMM_WORLD
	nrank = MPI.Comm_size(comm) # number of ranks 
	irank = MPI.Comm_rank(comm) # rank of this process
	println("Hello world, I am $(irank) of $(nrank)")
	MPI.Barrier(comm)
	println("comm: ",comm)
	discretization = "chebyshev_pseudospectral"

	etol = 1.0e-15
	# define inputs needed for the test
	ngrid = 100 #number of points per element 
	nelement_local = 5 # number of elements per rank
	nelement_global = nelement_local*nrank # total number of elements 
	println("ngrid = ",ngrid," nelement_local = ",nelement_local,
	 " nelement_global = ",nelement_global," nrank = ",nrank)
	L = 6.0 #physical box size in reference units 
	bc = "periodic"
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = ""
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
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
	spectral = setup_chebyshev_pseudospectral(x)
	println("made spectral")
	# create array for the function f(x) to be differentiated/integrated
	f = Array{Float64,1}(undef, x.n)
	g = Array{Float64,1}(undef, x.n)
	x_for_plot = Array{Float64,2}(undef, x.n, nrank)
	g_for_plot = Array{Float64,2}(undef, x.n, nrank)
	df_for_plot = Array{Float64,2}(undef, x.n, nrank)
	# create array for the derivative df/dx
	df = Array{Float64,1}(undef, x.n)
	# initialize f
	for ix ∈ 1:x.n
		#f[ix] = ( (cospi(2.0*x.grid[ix]/x.L)+sinpi(2.0*x.grid[ix]/x.L))
		#		  * exp(-x.grid[ix]^2) )
	    f[ix] =  cospi(2.0*x.grid[ix]/x.L)
	    g[ix] =  -sinpi(2.0*x.grid[ix]/x.L)
	end
	# differentiate f
	derivative!(df, f, x, spectral)
	#println(g)
	#println(df)
	# plot df and g per process
	outprefix = "run_MPI_test.plot."
	plot([x.grid,x.grid], [g,df], xlabel="x", ylabel="", label=["g" "df"],
         shape =:circle, markersize = 5, linewidth=2)
	outfile = outprefix*string(irank)*".pdf"
	savefig(outfile)
	#println(outfile)
	# plot df and g on rank 0
	x_for_plot .= 0.0
	g_for_plot .= 0.0
	df_for_plot .= 0.0
	for ix ∈ 1:x.n
		x_for_plot[ix,irank+1] = x.grid[ix]
		g_for_plot[ix,irank+1] = g[ix]
		df_for_plot[ix,irank+1] = df[ix]
	end
	MPI.Reduce!(x_for_plot,.+,comm)
	MPI.Reduce!(g_for_plot,.+,comm)
	MPI.Reduce!(df_for_plot,.+,comm)
	if irank == 0
		outprefix = "run_MPI_test.plot."
		xlist = [x_for_plot[:,1]]
		ylist = [g_for_plot[:,1]]
		labels = Matrix{String}(undef, 1, 2*nrank)
		#labels = ["g"]
		labels[1] = "g"
		for iproc in 2:nrank
			push!(xlist,x_for_plot[:,iproc])
			push!(ylist,g_for_plot[:,iproc])
			labels[iproc] ="g"
		end
		push!(xlist,x_for_plot[:,1])
		push!(ylist,df_for_plot[:,1])
		labels[1+nrank]="df"
		for iproc in 2:nrank
			push!(xlist,x_for_plot[:,iproc])
			push!(ylist,df_for_plot[:,iproc])
			#push!(labels,"df")
			labels[iproc+nrank]="df"
		end
		println(labels)
		plot(xlist, ylist, xlabel="x", ylabel="", label=labels, markersize = 1, linewidth=1)
		outfile = outprefix*"global.pdf"
		savefig(outfile)
		println(outfile)	
	end
	# integrate df/dx
	intdf = integral(df, x.wgts)
	println(intdf)
	intdf_out = MPI.Reduce(intdf,+,comm)
	# Test that error intdf is less than the specified error tolerance etol
	#@test abs(intdf) < etol
	if(irank == 0)
		println( "abs(intdf_out) = ", abs(intdf_out), ": etol = ",etol)
	end
	MPI.Finalize()
end 