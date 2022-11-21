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
	# nrank must be y_nblocks*x_nblocks, i.e.,
	# mpirun -n nrank run_MPI_test2D.jl
	x_ngrid = 10 #number of points per element 
	x_nelement_local  = 1 
	x_nelement_global = 2 # number of elements 
	y_ngrid = 12
	y_nelement_local  = 1 
	y_nelement_global = 3 
	
	y_nblocks = floor(Int,x_nelement_global/x_nelement_local)
	x_nblocks = floor(Int,y_nelement_global/y_nelement_local)
	
	MPI.Init()
	comm = MPI.COMM_WORLD
	nrank = MPI.Comm_size(comm) # number of ranks 
	irank = MPI.Comm_rank(comm) # rank of this process
	MPI.Barrier(comm)
	
	
	if irank == 0
		println("x_ngrid = ",x_ngrid," x_nelement_local = ",x_nelement_local,
			" x_nelement_global = ",x_nelement_global)
		println("y_ngrid = ",y_ngrid," y_nelement_local = ",
			y_nelement_local," y_nelement_global = ",y_nelement_global)
		println("nrank: ",nrank)
		println("y_nblocks: ",y_nblocks)
		println("x_nblocks: ",x_nblocks)
	
		
	end
	
	discretization = "chebyshev_pseudospectral"
	etol = 1.0e-15
	
	y_nrank_per_block = floor(Int,nrank/y_nblocks)
	y_iblock = mod(irank,y_nblocks) # irank - > y_iblock 
	y_irank_sub = floor(Int,irank/y_nblocks) # irank -> y_irank_sub
	# irank = y_iblock + x_nrank_per_block * y_irank_sub
	# useful information for debugging
	#println("y_nrank_per_block: ",y_nrank_per_block)
	#println("y_iblock: ",y_iblock)
	#println("y_irank_sub: ",y_irank_sub)
	
	x_nrank_per_block = floor(Int,nrank/x_nblocks)
	x_iblock = y_irank_sub # irank - > x_iblock 
	x_irank_sub = y_iblock # irank -> x_irank_sub
	# irank = x_iblock * x_nrank_per_block + x_irank_sub 
	# useful information for debugging
	#println("x_nrank_per_block: ",x_nrank_per_block)
	#println("x_iblock: ",x_iblock)
	#println("x_irank_sub: ",x_irank_sub)	
	
	# MPI.Comm_split(comm,color,key)
	# comm -> communicator to be split
	# color -> label of group of processes
	# key -> label of process in group
	y_comm_sub = MPI.Comm_split(comm,y_iblock,y_irank_sub)
	x_comm_sub = MPI.Comm_split(comm,x_iblock,x_irank_sub)
	
	L = 6.0 #physical box size in reference units 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	fd_option = ""
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	# create the 'input' struct containing input info needed to create a coordinate
	y_input = grid_input("coord", y_ngrid, y_nelement_global, y_nelement_local, 
		y_nrank_per_block, y_irank_sub, L, discretization, fd_option, bc, adv_input,y_comm_sub)
	x_input = grid_input("coord", x_ngrid, x_nelement_global, x_nelement_local, 
		x_nrank_per_block, x_irank_sub, L, discretization, fd_option, bc, adv_input,x_comm_sub)
	# create the coordinate struct 'x'
	y = define_coordinate(y_input)
	x = define_coordinate(x_input)
	# create arrays needed for Chebyshev pseudospectral treatment in x
	y_spectral = setup_chebyshev_pseudospectral(y)
	x_spectral = setup_chebyshev_pseudospectral(x)
	
	# create a 2D array for the function f(x,y) to be differentiated/integrated
	k = Array{Float64,2}(undef, x.n, y.n)
	f = Array{Float64,2}(undef, x.n, y.n)
	g = Array{Float64,2}(undef, x.n, y.n)
	h = Array{Float64,2}(undef, x.n, y.n)
	# create array for the derivative df/dx
	dfdx = Array{Float64,2}(undef, x.n, y.n)
	dkdy = Array{Float64,2}(undef, x.n, y.n)
	
	y_for_plot = Array{Float64,2}(undef, y.n, nrank)
	x_for_plot = Array{Float64,2}(undef, x.n, nrank)
	h_for_plot = Array{Float64,3}(undef, x.n, y.n, nrank)
	g_for_plot = Array{Float64,3}(undef, x.n, y.n, nrank)
	dfdx_for_plot = Array{Float64,3}(undef, x.n, y.n, nrank)
	dkdy_for_plot = Array{Float64,3}(undef, x.n, y.n, nrank)
	
	# initialize f
	
	#println("x",x.grid)
	#println("y",y.grid)
	for iy ∈ 1:y.n
		for ix ∈ 1:x.n
			k[ix,iy] =  sinpi(2.0*y.grid[iy]/y.L) #*sinpi(2.0*y.grid[iy]/y.L) 
			f[ix,iy] =  sinpi(2.0*x.grid[ix]/x.L) #*sinpi(2.0*y.grid[iy]/y.L) 
			g[ix,iy] =  (2.0*pi/x.L)*cospi(2.0*x.grid[ix]/x.L) #*sinpi(2.0*y.grid[iy]/y.L)
			h[ix,iy] =  (2.0*pi/y.L)*cospi(2.0*y.grid[iy]/y.L) #*cospi(2.0*x.grid[ix]/x.L) 
		end
	end
	# differentiate f w.r.t x
	for iy in 1:y.n
		@views derivative!(dfdx[:,iy], f[:,iy], x, x_spectral)
	end
	# differentiate f w.r.t y
	for ix in 1:x.n
		@views derivative!(dkdy[ix,:], k[ix,:], y, y_spectral)
	end
	
	# Test that error intdf is less than the specified error tolerance etol
	#@test abs(intdf) < etol
	# here we do a 1D integral in the x and y dimensions separately
	
	for iy in 1:1
		intdf = integral(dfdx[:,iy], x.wgts)
		intdf_out = MPI.Reduce(intdf,+,x.comm)
		if(x_irank_sub == 0 && x_iblock == 0)
			println( "abs(intdf_out) = ", abs(intdf_out), ": etol = ",etol)
		end
	end

	for ix in 1:1
		intdk = integral(dkdy[ix,:], y.wgts)
		intdk_out = MPI.Reduce(intdk,+,y.comm)
		if(y_irank_sub == 0 && y_iblock == 0)
			println( "abs(intdk_out) = ", abs(intdk_out), ": etol = ",etol)
		end
	end
	#println(intdf)
	

	
	#if(irank == 0)
		#println( "abs(intdf_out) = ", abs(intdf_out), ": etol = ",etol)
	#end
	
	
	# plot df g h per process
	outprefix = "run_MPI_test2D.plot."
	for iy in 1:1
		plot([x.grid,x.grid], [g[:,iy],dfdx[:,iy]], xlabel="x", ylabel="", label=["g" "df/dx"],
			 line = (2, [:solid :dash]), markersize = 2, linewidth=1)
		outfile = outprefix*"iy."*string(iy)*"."*string(irank)*".pdf"
		savefig(outfile)
	end
	for ix in 1:1
	plot([y.grid,y.grid], [h[ix,:],dkdy[ix,:]], xlabel="y", ylabel="", label=["h" "dk/dy"],
         line = (2, [:solid :dash]), markersize = 2, linewidth=1)
	outfile = outprefix*"ix."*string(ix)*"."*string(irank)*".pdf"
		savefig(outfile)
	end
	
	# get data onto irank = 0 for plotting
	
	y_for_plot .= 0.0
	x_for_plot .= 0.0
	h_for_plot .= 0.0
	g_for_plot .= 0.0
	dfdx_for_plot .= 0.0
	dkdy_for_plot .= 0.0
	
	y_for_plot[:,irank+1] .= y.grid[:]
	x_for_plot[:,irank+1] .= x.grid[:]
	h_for_plot[:,:,irank+1] .= h[:,:]
	g_for_plot[:,:,irank+1] .= g[:,:]
	dfdx_for_plot[:,:,irank+1] .= dfdx[:,:]
	dkdy_for_plot[:,:,irank+1] .= dkdy[:,:]
	
	MPI.Reduce!(y_for_plot,.+,comm)
	MPI.Reduce!(x_for_plot,.+,comm)
	MPI.Reduce!(g_for_plot,.+,comm)
	MPI.Reduce!(h_for_plot,.+,comm)
	MPI.Reduce!(dfdx_for_plot,.+,comm)
	MPI.Reduce!(dkdy_for_plot,.+,comm)
	
	#plot the data after the reduction operation	
	if irank == 0
		for iy in 1:1
			# plot all x blocks with iy = 1
			for x_iblockprim = 0:x_nblocks-1  
				xlist = []
				func_list = []
				labels = Matrix{String}(undef, 1, 2*x_nrank_per_block)
				
				for iproc in 0:x_nrank_per_block-1
					irankprim = x_iblockprim * x_nrank_per_block + iproc
					push!(xlist,x_for_plot[:,irankprim+1])
					push!(func_list,g_for_plot[:,iy,irankprim+1])
					labels[iproc+1] ="g"
				end
				
				for iproc in 0:x_nrank_per_block-1
					irankprim = x_iblockprim * x_nrank_per_block + iproc
					push!(xlist,x_for_plot[:,irankprim+1])
					push!(func_list,dfdx_for_plot[:,iy,irankprim+1])
					labels[iproc+1+x_nrank_per_block] ="df/dx"
				end
				#println(xlist)
				#println(func_list)
				#println(labels)
				plot(xlist, func_list, xlabel="x", ylabel="", label=labels, markersize = 1, linewidth=1)
				outfile = "run_MPI_test2D.plot.iy."*string(iy)*".x_iblock."*string(x_iblockprim)*".global.pdf"
				savefig(outfile)
				println(outfile)	
			end
		end
		
		for ix in 1:1
			# plot all y blocks with ix = 1 
			for y_iblockprim = 0:y_nblocks-1  
				xlist = []
				func_list = []
				labels = Matrix{String}(undef, 1, 2*y_nrank_per_block)
				
				for iproc in 0:y_nrank_per_block-1
					irankprim = y_iblockprim + x_nrank_per_block * iproc
					push!(xlist,y_for_plot[:,irankprim+1])
					push!(func_list,h_for_plot[ix,:,irankprim+1])
					labels[iproc+1] ="h"
				end
				
				for iproc in 0:y_nrank_per_block-1
					irankprim = y_iblockprim + x_nrank_per_block * iproc
					push!(xlist,y_for_plot[:,irankprim+1])
					push!(func_list,dkdy_for_plot[ix,:,irankprim+1])
					labels[iproc+1+y_nrank_per_block] ="dk/dy"
				end
				#println(xlist)
				#println(func_list)
				#println(labels)
				plot(xlist, func_list, xlabel="y", ylabel="", label=labels, markersize = 1, linewidth=1)
				outfile = "run_MPI_test2D.plot.ix."*string(ix)*".y_iblock."*string(y_iblockprim)*".global.pdf"
				savefig(outfile)
				println(outfile)	
			end
		end
	end	
	
	MPI.Finalize()
end
	