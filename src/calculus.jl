"""
"""
module calculus

export derivative!
export reconcile_element_boundaries_MPI!
export integral


using ..chebyshev: chebyshev_info, chebyshev_derivative!
using ..finite_differences: derivative_finite_difference!
using ..type_definitions: mk_float, mk_int
using MPI 
using ..communication: block_rank

"""
Chebyshev transform f to get Chebyshev spectral coefficients and use them to calculate f'
"""
function derivative!(df, f, coord, adv_fac, spectral::chebyshev_info)
    # get the derivative at each grid point within each element and store in df
    chebyshev_derivative!(coord.scratch_2d, f, spectral, coord)
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the derivative from the upwind element.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord, adv_fac)
end

"""
Chebyshev transform f to get Chebyshev spectral coefficients and use them to calculate f'
"""
function derivative!(df, f, coord, spectral::chebyshev_info)
    # get the derivative at each grid point within each element and store in df
    chebyshev_derivative!(coord.scratch_2d, f, spectral, coord)
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the average of the derivatives from neighboring elements.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord)
end

"""
calculate the derivative of f using finite differences, with particular scheme
specifiied by coord.fd_option; stored in df
"""
function derivative!(df, f, coord, adv_fac, not_spectral::Bool)
    # get the derivative at each grid point within each element and store in df
    derivative_finite_difference!(coord.scratch_2d, f, coord.cell_width, adv_fac,
        coord.bc, coord.fd_option, coord.igrid, coord.ielement)
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the derivative from the upwind element.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord, adv_fac)
end

"""
calculate the derivative of f using centered differences; stored in df
"""
function derivative!(df, f, coord, not_spectral::Bool)
    # get the derivative at each grid point within each element and store in df
    derivative_finite_difference!(coord.scratch_2d, f, coord.cell_width,
        coord.bc, "fourth_order_centered", coord.igrid, coord.ielement)
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the average of the derivatives from neighboring elements.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord)
end

"""
"""
function derivative_elements_to_full_grid!(df1d, df2d, coord, adv_fac::AbstractArray{mk_float,1})
    # no changes need to be made for the derivative at points away from element boundaries
    elements_to_full_grid_interior_pts!(df1d, df2d, coord)
    # resolve the multi-valued nature of the derivative at element boundaries
    # by using the derivative from the upwind element
    reconcile_element_boundaries_upwind!(df1d, df2d, coord, adv_fac)
    return nothing
end

"""
"""
function derivative_elements_to_full_grid!(df1d, df2d, coord)
    # no changes need to be made for the derivative at points away from element boundaries
    elements_to_full_grid_interior_pts!(df1d, df2d, coord)
    # resolve the multi-valued nature of the derivative at element boundaries
    # by using the derivative from the upwind element
    reconcile_element_boundaries_centered!(df1d, df2d, coord)
    return nothing
end

"""
maps the derivative at points away from element boundaries
from the grid/element representation to the full grid representation
"""
function elements_to_full_grid_interior_pts!(df1d, df2d, coord)
    # for efficiency, define ngm1 to be ngrid-1, as it will be used repeatedly
    ngm1 = coord.ngrid-1
    # treat the first element
    df1d[2:ngm1] .= @view df2d[2:ngm1,1]
    # deal with any additional elements
    if coord.nelement_local > 1
        for ielem ∈ 2:coord.nelement_local
            @. df1d[coord.imin[ielem]:coord.imax[ielem]-1] = @view df2d[2:ngm1,ielem]
        end
    end
    return nothing
end

"""
if at the boundary point within the element, must carefully
choose which value of df to use; this is because
df is multi-valued at the overlapping point at the boundary
between neighboring elements.
here we choose to use the value of df from the upwind element.
"""
function reconcile_element_boundaries_upwind!(df1d, df2d, coord, adv_fac::AbstractArray{mk_float,1})
    # note that the first ngrid points are classified as belonging to the first element
    # and the next ngrid-1 points belonging to second element, etc.

    # first deal with domain boundaries
    if coord.bc == "periodic" && coord.nelement_global == coord.nelement_local
        # consider left domain boundary
        if adv_fac[1] > 0.0
            # adv_fac > 0 corresponds to negative advection speed, so
            # use derivative information from upwind element at larger coordinate value
            df1d[1] = df2d[1,1]
        elseif adv_fac[1] < 0.0
            # adv_fac < 0 corresponds to positive advection speed, so
            # use derivative information from upwind element at smaller coordinate value
            df1d[1] = df2d[coord.ngrid,coord.nelement_local]
        else
            # adv_fac = 0, so no upwinding required;
            # use average value
            df1d[1] = 0.5*(df2d[1,1]+df2d[coord.ngrid,coord.nelement_local])
        end
        # consider right domain boundary
        if adv_fac[coord.n] > 0.0
            # adv_fac > 0 corresponds to negative advection speed, so
            # use derivative information from upwind element at larger coordinate value
            df1d[coord.n] = df2d[1,1]
        elseif adv_fac[coord.ngrid] < 0.0
            # adv_fac < 0 corresponds to positive advection speed, so
            # use derivative information from upwind element at smaller coordinate value
            df1d[coord.n] = df2d[coord.ngrid,coord.nelement_local]
        else
            # adv_fac = 0, so no upwinding required;
            # use average value
            df1d[coord.n] = 0.5*(df2d[1,1]+df2d[coord.ngrid,coord.nelement_local])
        end
    else
        df1d[1] = df2d[1,1]
        df1d[coord.n] = df2d[coord.ngrid,coord.nelement_local]
    end
    # next consider remaining elements, if any.
    # only need to consider interior element boundaries
    if coord.nelement_local > 1
        for ielem ∈ 2:coord.nelement_local
            im1 = ielem-1
            # consider left element boundary
            if adv_fac[coord.imax[im1]] > 0.0
                # adv_fac > 0 corresponds to negative advection speed, so
                # use derivative information from upwind element at larger coordinate value
                df1d[coord.imax[im1]] = df2d[1,ielem]
            elseif adv_fac[coord.imax[im1]] < 0.0
                # adv_fac < 0 corresponds to positive advection speed, so
                # use derivative information from upwind element at smaller coordinate value
                df1d[coord.imax[im1]] = df2d[coord.ngrid,im1]
            else
                # adv_fac = 0, so no upwinding required;
                # use average value
                df1d[coord.imax[im1]] = 0.5*(df2d[1,ielem]+df2d[coord.ngrid,im1])
            end
        end
    end
    return nothing
end

"""
if at the boundary point within the element, must carefully
choose which value of df to use; this is because
df is multi-valued at the overlapping point at the boundary
between neighboring elements.
here we choose to use the averaged value across elements.
"""

function reconcile_element_boundaries_centered!(df1d, df2d, coord)
    # note that the first ngrid points are classified as belonging to the first element
    # and the next ngrid-1 points belonging to second element, etc.
	# first deal with domain boundaries
	if coord.bc == "periodic" && coord.nelement_local == coord.nelement_global
		# consider left domain boundary
		df1d[1] = 0.5*(df2d[1,1]+df2d[coord.ngrid,coord.nelement_local])
		# consider right domain boundary
		df1d[coord.n] = df1d[1]
	else 
	# put endpoints into 1D array to be reconciled
	# across processes at a higher scope -> larger message sizes possible
		df1d[1] = df2d[1,1]
		df1d[coord.n] = df2d[coord.ngrid,coord.nelement_local]
	end
	# next consider remaining elements, if any.
	# only need to consider interior element boundaries
	if coord.nelement_local > 1
		for ielem ∈ 2:coord.nelement_local
			im1 = ielem-1
			# consider left element boundary
			df1d[coord.imax[im1]] = 0.5*(df2d[1,ielem]+df2d[coord.ngrid,im1])
		end
	end
	return nothing
end

"""
extension of the above function to distributed memory MPI
function allows for arbitray array sizes ONLY IF the 
if statements doing the final endpoint assignments are 
updated to include each physical dimension required
in the main code
"""

function assign_endpoint!(df1d::Array{mk_float,Ndims},
 receive_buffer::Array{mk_float,Mdims},key::String,coord) where {Ndims,Mdims}
	if key == "lower"
		j = 1
	elseif key == "upper"
		j = coord.n
	else
		println("ERROR: invalid key in assign_endpoint!")
	end
    # test against coord name -- make sure to use exact string delimiters e.g. "x" not 'x'
	# test against Ndims (autodetermined) to choose which array slices to use in assigning endpoints
	#println("DEBUG MESSAGE: coord.name: ",coord.name," Ndims: ",Ndims," key: ",key)
	if coord.name == "z" && Ndims==2
		df1d[j,:] .= receive_buffer[:]
		#println("ASSIGNING DATA")
	elseif coord.name == "z" && Ndims==4
		df1d[:,:,j,:] .= receive_buffer[:,:,:]
		#println("ASSIGNING DATA")
    elseif coord.name == "z" && Ndims==5
		df1d[:,:,:,j,:] .= receive_buffer[:,:,:,:]
		#println("ASSIGNING DATA")
	elseif coord.name == "r" && Ndims==2
		df1d[:,j] .= receive_buffer[:]
		#println("ASSIGNING DATA")
	elseif coord.name == "r" && Ndims==4
		df1d[:,:,:,j] .= receive_buffer[:,:,:]
		#println("ASSIGNING DATA")
	elseif coord.name == "r" && Ndims==5
		df1d[:,:,:,:,j] .= receive_buffer[:,:,:,:]
		#println("ASSIGNING DATA")
	else
        println("ERROR: failure to assign endpoints in reconcile_element_boundaries_MPI! (centered): coord.name: ",coord.name," Ndims: ",Ndims," key: ",key)
    end
end

function reconcile_element_boundaries_MPI!(df1d::Array{mk_float,Ndims},
	dfdx_lower_endpoints::Array{mk_float,Mdims}, dfdx_upper_endpoints::Array{mk_float,Mdims},
	send_buffer::Array{mk_float,Mdims}, receive_buffer::Array{mk_float,Mdims}, coord) where {Ndims,Mdims}
	
    if block_rank[] == 0 # lead process on this shared-memory block
        
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
        assign_endpoint!(df1d,receive_buffer,"lower",coord)
        
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
        assign_endpoint!(df1d,receive_buffer,"upper",coord)
        
    end
end
	
function apply_adv_fac!(buffer::Array{mk_float,Ndims},adv_fac::Array{mk_float,Ndims},endpoints::Array{mk_float,Ndims},sgn::mk_int) where Ndims
		#buffer contains off-process endpoint
		#adv_fac < 0 is positive advection speed
		#adv_fac > 0 is negative advection speed
		#endpoint is local on-process endpoint
		#sgn = 1 for send irank -> irank + 1
		#sgn = -1 for send irank + 1 -> irank
		#loop over all indices in array
		for i in eachindex(buffer,adv_fac,endpoints)
			if sgn*adv_fac[i] > 0.0 
			# replace buffer value with endpoint value 
				buffer[i] = endpoints[i]
			elseif sgn*adv_fac[i] < 0.0
				break #do nothing
			else #average values 
				buffer[i] = 0.5*(buffer[i] + endpoints[i])
			end
		end
		
	end
	
function reconcile_element_boundaries_MPI!(df1d::Array{mk_float,Ndims}, 
	adv_fac_lower_endpoints::Array{mk_float,Mdims}, adv_fac_upper_endpoints::Array{mk_float,Mdims},
	dfdx_lower_endpoints::Array{mk_float,Mdims}, dfdx_upper_endpoints::Array{mk_float,Mdims},
	send_buffer::Array{mk_float,Mdims}, receive_buffer::Array{mk_float,Mdims}, coord) where {Ndims,Mdims}
	
    if block_rank[] == 0 # lead process on this shared-memory block
        
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
                # depending on adv_fac, update the extreme lower endpoint with data from irank = nrank -1	
                apply_adv_fac!(receive_buffer,adv_fac_lower_endpoints,dfdx_lower_endpoints,1)
            else # directly use value from Cheb at extreme lower point 
                receive_buffer .= dfdx_lower_endpoints
            end
        else # depending on adv_fac, update the lower endpoint with data from irank = nrank -1	
            apply_adv_fac!(receive_buffer,adv_fac_lower_endpoints,dfdx_lower_endpoints,1)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        assign_endpoint!(df1d,receive_buffer,"lower",coord)
        
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
                # depending on adv_fac, update the extreme upper endpoint with data from irank = 0
                apply_adv_fac!(receive_buffer,adv_fac_upper_endpoints,dfdx_upper_endpoints,-1)
            else #directly use value from Cheb
                receive_buffer .= dfdx_upper_endpoints
            end
        else # enforce continuity at upper endpoint
            apply_adv_fac!(receive_buffer,adv_fac_upper_endpoints,dfdx_upper_endpoints,-1)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        assign_endpoint!(df1d,receive_buffer,"upper",coord)

    end
end
	

"""
Computes the integral of the integrand, using the input wgts
"""
function integral(integrand, wgts)
    # n is the number of grid points
    n = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n == length(integrand) || throw(BoundsError(integrand))
    @boundscheck n == length(wgts) || throw(BoundsError(wgts))
    @inbounds for i ∈ 1:n
        integral += integrand[i]*wgts[i]
    end
    return integral
end

"""
Computes the integral of the integrand multiplied by v, using the input wgts
"""
function integral(integrand, v, wgts)
    # n is the number of grid points
    n = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n == length(integrand) || throw(BoundsError(integrand))
    @boundscheck n == length(v) || throw(BoundsError(v))
    @boundscheck n == length(wgts) || throw(BoundsError(wgts))
    @inbounds for i ∈ 1:n
        integral += integrand[i] * v[i] * wgts[i]
    end
    return integral
end

"""
Computes the integral of the integrand multiplied by v^n, using the input wgts
"""
function integral(integrand, v, n, wgts)
    # n is the number of grid points
    n_v = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n_v == length(integrand) || throw(BoundsError(integrand))
    @boundscheck n_v == length(v) || throw(BoundsError(v))
    @boundscheck n_v == length(wgts) || throw(BoundsError(wgts))
    @inbounds for i ∈ 1:n_v
        integral += integrand[i] * v[i] ^ n * wgts[i]
    end
    return integral
end


"""
2D velocity integration routines
"""

"""
Computes the integral of the 2D integrand, using the input wgts
"""
function integral(integrand, vx, px, wgtsx, vy, py, wgtsy)
    # nx is the number of grid points
    nx = length(wgtsx)
    ny = length(wgtsy)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck nx == size(integrand,1) || throw(BoundsError(integrand))
    @boundscheck ny == size(integrand,2) || throw(BoundsError(integrand))
    @boundscheck nx == length(vx) || throw(BoundsError(vx))
    @boundscheck ny == length(vy) || throw(BoundsError(vy))
#    @boundscheck ny == length(wgtsy) || throw(BoundsError(wtgsy))
#    @boundscheck nx == length(wgtsx) || throw(BoundsError(wtgsx))
   
    @inbounds for j ∈ 1:ny
        @inbounds for i ∈ 1:nx
            integral += integrand[i,j] * (vx[i] ^ px) * (vy[j] ^ py) * wgtsx[i] * wgtsy[j]
        end
    end
    return integral
end


"""
3D velocity integration routines
"""

"""
Computes the integral of the 3D integrand, using the input wgts
"""
function integral(integrand, vx, px, wgtsx, vy, py, wgtsy, vz, pz, wgtsz)
    # nx is the number of grid points
    nx = length(wgtsx)
    ny = length(wgtsy)
    nz = length(wgtsz)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck nx == size(integrand,1) || throw(BoundsError(integrand))
    @boundscheck ny == size(integrand,2) || throw(BoundsError(integrand))
    @boundscheck nz == size(integrand,3) || throw(BoundsError(integrand))
    @boundscheck nx == length(vx) || throw(BoundsError(vx))
    @boundscheck ny == length(vy) || throw(BoundsError(vy))
    @boundscheck nz == length(vz) || throw(BoundsError(vz))
   
    @inbounds for k ∈ 1:nz
        @inbounds for j ∈ 1:ny
            @inbounds for i ∈ 1:nx
                integral += integrand[i,j,k] * (vx[i] ^ px) * (vy[j] ^ py) * (vz[k] ^ pz) * wgtsx[i] * wgtsy[j] * wgtsz[k]
            end
        end
    end
    return integral
end


end
