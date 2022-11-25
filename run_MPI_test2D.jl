if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.calculus: derivative!, integral
	#using coordinates: coordinate_info
	import MPI 
	using Plots
	
	function reconcile_element_boundaries_MPI!(df1d::Array{Float64,Ndims},
	dfdx_lower_endpoints::Array{Float64,N}, dfdx_upper_endpoints::Array{Float64,N},
	send_buffer::Array{Float64,N}, receive_buffer::Array{Float64,N}, coord) where {Ndims,N}
		
		#counter to test if endpoint data assigned
		assignment_counter = 0
		
		# now deal with endpoints that are stored across ranks
		comm = coord.comm
		nrank = coord.nrank 
		irank = coord.irank 
		#send_buffer = coord.send_buffer
		#receive_buffer = coord.receive_buffer
		# sending pattern is cyclic. First we send data form irank -> irank + 1
		# to fix the lower endpoints, then we send data from irank -> irank - 1
		# to fix upper endpoints. Special exception for the periodic points.
		# receive_buffer[1] is for data received, send_buffer[1] is data to be sent
		
		send_buffer .= dfdx_upper_endpoints #highest end point on THIS rank
		# pass data from irank -> irank + 1, receive data from irank - 1
		idst = mod(irank+1,nrank) # destination rank for sent data
		isrc = mod(irank-1,nrank) # source rank for received data
		#MRH what value should tag take here and below? Esp if nrank >= 32
		rreq = MPI.Irecv!(receive_buffer, comm; source=isrc, tag=isrc+32)
		sreq = MPI.Isend(send_buffer, comm; dest=idst, tag=irank+32)
		#print("$irank: Sending   $irank -> $idst = $send_buffer\n")
		stats = MPI.Waitall([rreq, sreq])
		#print("$irank: Received $isrc -> $irank = $receive_buffer\n")
		MPI.Barrier(comm)
		
		# no update receive buffer, taking into account the reconciliation
		if irank == 0
			if coord.bc == "periodic"
				#update the extreme lower endpoint with data from irank = nrank -1	
				receive_buffer .= 0.5*(receive_buffer .+ dfdx_lower_endpoints)
			else #directly use value from Cheb
				receive_buffer .= dfdx_lower_endpoints
			end
		else # enforce continuity at lower endpoint
			receive_buffer .= 0.5*(receive_buffer .+ dfdx_lower_endpoints)
		end
		
		#now update the df1d array -- using a slice appropriate to the dimension reconciled
		# test against coord name -- make sure to use exact string delimiters e.g. "x" not 'x'
		# test against Ndims (autodetermined) to choose which array slices to use in assigning endpoints
		#println("coord.name: ",coord.name," Ndims: ",Ndims)
		if coord.name == "x" && Ndims==2
			df1d[1,:] .= receive_buffer[:]
			assignment_counter += 1
		elseif coord.name == "x" && Ndims==3
			df1d[1,:,:] .= receive_buffer[:,:]
			assignment_counter += 1
		elseif coord.name == "y" && Ndims==2
			df1d[:,1] .= receive_buffer[:]
			assignment_counter += 1
		elseif coord.name == "y" && Ndims==3
			df1d[:,1,:] .= receive_buffer[:,:]
			assignment_counter += 1
		end
		
		send_buffer .= dfdx_lower_endpoints #lowest end point on THIS rank
		# pass data from irank -> irank - 1, receive data from irank + 1
		idst = mod(irank-1,nrank) # destination rank for sent data
		isrc = mod(irank+1,nrank) # source rank for received data
		#MRH what value should tag take here and below? Esp if nrank >= 32
		rreq = MPI.Irecv!(receive_buffer, comm; source=isrc, tag=isrc+32)
		sreq = MPI.Isend(send_buffer, comm; dest=idst, tag=irank+32)
		#print("$irank: Sending   $irank -> $idst = $send_buffer\n")
		stats = MPI.Waitall([rreq, sreq])
		#print("$irank: Received $isrc -> $irank = $receive_buffer\n")
		MPI.Barrier(comm)
		
		if irank == nrank-1
			if coord.bc == "periodic"
				#update the extreme upper endpoint with data from irank = 0
				receive_buffer .= 0.5*(receive_buffer .+ dfdx_upper_endpoints)
			else #directly use value from Cheb
				receive_buffer .= dfdx_upper_endpoints
			end
		else # enforce continuity at upper endpoint
			receive_buffer .= 0.5*(receive_buffer .+ dfdx_upper_endpoints)
		end
	
		#now update the df1d array -- using a slice appropriate to the dimension reconciled
		# test against coord name -- make sure to use exact string delimiters e.g. "x" not 'x'
		# test against Ndims (autodetermined) to choose which array slices to use in assigning endpoints
		#println("coord.name: ",coord.name," Ndims: ",Ndims)
		if coord.name=="x" && Ndims ==2
			df1d[end,:] .= receive_buffer[:]
			assignment_counter += 1
		elseif coord.name=="x" && Ndims ==3
			df1d[end,:,:] .= receive_buffer[:,:]
			assignment_counter += 1
		elseif coord.name=="y" && Ndims ==2
			df1d[:,end] .= receive_buffer[:]
			assignment_counter += 1
		elseif coord.name=="y" && Ndims ==3
			df1d[:,end,:] .= receive_buffer[:,:]
			assignment_counter += 1
		end
	
		if  !(assignment_counter == 2)
			println("ERROR: failure to assign endpoints in reconcile_element_boundaries_MPI!: coord.name: ",coord.name," Ndims: ",Ndims)
		end
	end
	
	#3D version
	function derivative_x!(dfdx::Array{Float64,3},f::Array{Float64,3},
		dfdx_lower_endpoints::Array{Float64,2}, dfdx_upper_endpoints::Array{Float64,2},
		x_send_buffer::Array{Float64,2},x_receive_buffer::Array{Float64,2},
		x_spectral,x,y,z)
	
		# differentiate f w.r.t x
		for iz in 1:z.n
			for iy in 1:y.n
				@views derivative!(dfdx[:,iy,iz], f[:,iy,iz], x, x_spectral)
				# get external endpoints to reconcile via MPI
				dfdx_lower_endpoints[iy,iz] = x.scratch_2d[1,1]
				dfdx_upper_endpoints[iy,iz] = x.scratch_2d[end,end] 
			end
		end
		# now reconcile element boundaries across
		# processes with large message involving all y 
		if x.nelement_local < x.nelement_global
			reconcile_element_boundaries_MPI!(dfdx,
			 dfdx_lower_endpoints,dfdx_upper_endpoints,
			 x_send_buffer, x_receive_buffer, x)
		end
		
	end
	
	#2D version
	function derivative_x!(dfdx::Array{Float64,2},f::Array{Float64,2},
		dfdx_lower_endpoints::Array{Float64,1}, dfdx_upper_endpoints::Array{Float64,1},
		x_send_buffer::Array{Float64,1},x_receive_buffer::Array{Float64,1},
		x_spectral,x,y)
	
		# differentiate f w.r.t x
		for iy in 1:y.n
			@views derivative!(dfdx[:,iy], f[:,iy], x, x_spectral)
			# get external endpoints to reconcile via MPI
			dfdx_lower_endpoints[iy] = x.scratch_2d[1,1]
			dfdx_upper_endpoints[iy] = x.scratch_2d[end,end] 
		end
		# now reconcile element boundaries across
		# processes with large message involving all y 
		if x.nelement_local < x.nelement_global
			reconcile_element_boundaries_MPI!(dfdx,
			 dfdx_lower_endpoints,dfdx_upper_endpoints,
			 x_send_buffer, x_receive_buffer, x)
		end
		
	end
	
	#3D version
	function derivative_y!(dfdy::Array{Float64,3},f::Array{Float64,3},
		dfdy_lower_endpoints::Array{Float64,2}, dfdy_upper_endpoints::Array{Float64,2},
		y_send_buffer::Array{Float64,2},y_receive_buffer::Array{Float64,2},
		y_spectral,x,y,z)
	
		# differentiate f w.r.t y
		for iz in 1:z.n
			for ix in 1:x.n
				@views derivative!(dfdy[ix,:,iz], f[ix,:,iz], y, y_spectral)
				# get external endpoints to reconcile via MPI
				dfdy_lower_endpoints[ix,iz] = y.scratch_2d[1,1]
				dfdy_upper_endpoints[ix,iz] = y.scratch_2d[end,end] 
			end
		end
		# now reconcile element boundaries across
		# processes with large message involving all y 
		if y.nelement_local < y.nelement_global
			reconcile_element_boundaries_MPI!(dfdy,
			 dfdy_lower_endpoints,dfdy_upper_endpoints,
			 y_send_buffer, y_receive_buffer, y)
		end
	end
	
	#2D version
	function derivative_y!(dfdy::Array{Float64,2},f::Array{Float64,2},
		dfdy_lower_endpoints::Array{Float64,1}, dfdy_upper_endpoints::Array{Float64,1},
		y_send_buffer::Array{Float64,1},y_receive_buffer::Array{Float64,1},
		y_spectral,x,y)
	
		# differentiate f w.r.t y
		for ix in 1:x.n
			@views derivative!(dfdy[ix,:], f[ix,:], y, y_spectral)
			# get external endpoints to reconcile via MPI
			dfdy_lower_endpoints[ix] = y.scratch_2d[1,1]
			dfdy_upper_endpoints[ix] = y.scratch_2d[end,end] 
		end
		# now reconcile element boundaries across
		# processes with large message involving all y 
		if y.nelement_local < y.nelement_global
			reconcile_element_boundaries_MPI!(dfdy,
			 dfdy_lower_endpoints,dfdy_upper_endpoints,
			 y_send_buffer, y_receive_buffer, y)
		end
	end
	
	# define inputs needed for the xy calculus test
	# nrank must be nrank = y_nblocks*x_nblocks, i.e.,
	# mpirun -n nrank run_MPI_test2D.jl
	x_ngrid = 4 #number of points per element 
	x_nelement_local  = 1 
	x_nelement_global = 5 # number of elements 
	y_ngrid = 4
	y_nelement_local  = 1
	y_nelement_global = 7
	
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
	y_input = grid_input("y", y_ngrid, y_nelement_global, y_nelement_local, 
		y_nrank_per_block, y_irank_sub, L, discretization, fd_option, bc, adv_input,y_comm_sub)
	x_input = grid_input("x", x_ngrid, x_nelement_global, x_nelement_local, 
		x_nrank_per_block, x_irank_sub, L, discretization, fd_option, bc, adv_input,x_comm_sub)
	# z dimension kept entirely local
	z_ngrid = 7
	z_nelement_local = 1
	z_nelement_global = 1
	z_nrank_per_block = 0 # dummy value
	z_irank_sub = 0 #dummy value
	z_comm_sub = false #dummy value
	z_input = grid_input("z", z_ngrid, x_nelement_global, x_nelement_local, 
		x_nrank_per_block, x_irank_sub, L, discretization, fd_option, bc, adv_input,z_comm_sub)
	z = define_coordinate(z_input)
	
	# create the coordinate struct 'x'
	y = define_coordinate(y_input)
	x = define_coordinate(x_input)
	# create arrays needed for Chebyshev pseudospectral treatment in x
	y_spectral = setup_chebyshev_pseudospectral(y)
	x_spectral = setup_chebyshev_pseudospectral(x)
	
	
	# create a 3D array for the function f(x,y) to be differentiated/integrated
	df3Ddx = Array{Float64,3}(undef, x.n, y.n, z.n)
	f3D = Array{Float64,3}(undef, x.n, y.n, z.n)
	g3D = Array{Float64,3}(undef, x.n, y.n, z.n)
	dk3Ddy = Array{Float64,3}(undef, x.n, y.n, z.n)
	k3D = Array{Float64,3}(undef, x.n, y.n, z.n)
	h3D = Array{Float64,3}(undef, x.n, y.n, z.n)
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
	for iz ∈ 1:z.n
		for iy ∈ 1:y.n
			for ix ∈ 1:x.n
				f3D[ix,iy,iz] =  sinpi(2.0*x.grid[ix]/x.L) #*sinpi(2.0*y.grid[iy]/y.L) 
				g3D[ix,iy,iz] =  (2.0*pi/x.L)*cospi(2.0*x.grid[ix]/x.L) #*sinpi(2.0*y.grid[iy]/y.L)
				k3D[ix,iy,iz] =  sinpi(2.0*y.grid[iy]/y.L) #*sinpi(2.0*y.grid[iy]/y.L) 
				h3D[ix,iy,iz] =  (2.0*pi/y.L)*cospi(2.0*y.grid[iy]/y.L) #*cospi(2.0*x.grid[ix]/x.L) 
			end
		end
	end
	
	for iy ∈ 1:y.n
		for ix ∈ 1:x.n
			k[ix,iy] =  sinpi(2.0*y.grid[iy]/y.L) #*sinpi(2.0*y.grid[iy]/y.L) 
			f[ix,iy] =  sinpi(2.0*x.grid[ix]/x.L) #*sinpi(2.0*y.grid[iy]/y.L) 
			g[ix,iy] =  (2.0*pi/x.L)*cospi(2.0*x.grid[ix]/x.L) #*sinpi(2.0*y.grid[iy]/y.L)
			h[ix,iy] =  (2.0*pi/y.L)*cospi(2.0*y.grid[iy]/y.L) #*cospi(2.0*x.grid[ix]/x.L) 
		end
	end
	
	x_send_buffer = Array{Float64,1}(undef,y.n)
	x_receive_buffer = Array{Float64,1}(undef,y.n)
	dfdx_lower_endpoints = Array{Float64,1}(undef,y.n)
	dfdx_upper_endpoints = Array{Float64,1}(undef,y.n)
	# differentiate f w.r.t x
	derivative_x!(dfdx,f,dfdx_lower_endpoints,dfdx_upper_endpoints,x_send_buffer,x_receive_buffer,x_spectral,x,y)

	x3D_send_buffer = Array{Float64,2}(undef,y.n,z.n)
	x3D_receive_buffer = Array{Float64,2}(undef,y.n,z.n)
	df3Ddx_lower_endpoints = Array{Float64,2}(undef,y.n,z.n)
	df3Ddx_upper_endpoints = Array{Float64,2}(undef,y.n,z.n)
	# differentiate f w.r.t x
	derivative_x!(df3Ddx,f3D,df3Ddx_lower_endpoints,df3Ddx_upper_endpoints,x3D_send_buffer,x3D_receive_buffer,x_spectral,x,y,z)
		
	y_send_buffer = Array{Float64,1}(undef,x.n)
	y_receive_buffer = Array{Float64,1}(undef,x.n)
	dkdy_lower_endpoints = Array{Float64,1}(undef,x.n)
	dkdy_upper_endpoints = Array{Float64,1}(undef,x.n)
	# differentiate k w.r.t y
	derivative_y!(dkdy,k,dkdy_lower_endpoints,dkdy_upper_endpoints,y_send_buffer,y_receive_buffer,y_spectral,x,y)
	
	y3D_send_buffer = Array{Float64,2}(undef,x.n,z.n)
	y3D_receive_buffer = Array{Float64,2}(undef,x.n,z.n)
	dk3Ddy_lower_endpoints = Array{Float64,2}(undef,x.n,z.n)
	dk3Ddy_upper_endpoints = Array{Float64,2}(undef,x.n,z.n)
	# differentiate f w.r.t x
	derivative_y!(dk3Ddy,k3D,dk3Ddy_lower_endpoints,dk3Ddy_upper_endpoints,y3D_send_buffer,y3D_receive_buffer,y_spectral,x,y,z)
	
	# Test that error intdf is less than the specified error tolerance etol
	#@test abs(intdf) < etol
	# here we do a 1D integral in the x and y dimensions separately
	
	for iz in 1:1
		for iy in 1:1
			intdf3D = integral(df3Ddx[:,iy,iz], x.wgts)
			intdf3D_out = MPI.Reduce(intdf3D,+,x.comm)
			if(x_irank_sub == 0 && x_iblock == 0)
				println( "abs(intdf3D_out) = ", abs(intdf3D_out), ": etol = ",etol)
			end
		end
	end
	
	for iy in 1:1
		intdf = integral(dfdx[:,iy], x.wgts)
		intdf_out = MPI.Reduce(intdf,+,x.comm)
		if(x_irank_sub == 0 && x_iblock == 0)
			println( "abs(intdf_out) = ", abs(intdf_out), ": etol = ",etol)
		end
	end

	for iz in 1:1
		for ix in 1:1
			intdk3D = integral(dkdy[ix,:,iz], y.wgts)
			intdk3D_out = MPI.Reduce(intdk3D,+,y.comm)
			if(y_irank_sub == 0 && y_iblock == 0)
				println( "abs(intdk3D_out) = ", abs(intdk3D_out), ": etol = ",etol)
			end
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
	