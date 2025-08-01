"""
"""
module calculus

# Import moment_kinetics so that we can refer to it in docstrings
import moment_kinetics

export derivative!, second_derivative!, laplacian_derivative!
export elementwise_indefinite_integration!
export reconcile_element_boundaries_MPI!
export integral
export indefinite_integral!

using ..moment_kinetics_structs: discretization_info, null_spatial_dimension_info,
                                 null_velocity_dimension_info, weak_discretization_info
using ..timer_utils
using ..type_definitions: mk_float, mk_int
using MPI
using ..communication: block_rank
using ..communication: @_block_synchronize
using ..looping

using LinearAlgebra

"""
    elementwise_derivative!(coord, f, adv_fac, spectral)
    elementwise_derivative!(coord, f, spectral)

Generic function for element-by-element derivatives

First signature, with `adv_fac`, calculates an upwind derivative, the second signature
calculates a derivative without upwinding information.

Result is stored in coord.scratch_2d.
"""
function elementwise_derivative! end

"""
"""
function elementwise_indefinite_integration! end

"""
    indefinite_integral!(pf, f, coord, spectral)

Indefinite line integral. 

This function is designed to work on local-in-memory data only,
with distributed-memory MPI not implemented here.
A function which integrates along a line which is distributed
in memory exists in [`moment_kinetics.em_fields`](@ref) as 
`calculate_phi_from_Epar!()`. The distributed-memory functionality could
be ported to a generic function, similiar to how the derivative!
functions are generalised in [`moment_kinetics.derivatives`](@ref).
"""
function indefinite_integral!(pf, f, coord, spectral::discretization_info)
    # get the indefinite integral at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_indefinite_integration!(coord, f, spectral)
    # map the integral from the elemental grid to the full grid;
    # taking care to match integral constants at the boundaries
    # we assume that the lower limit of the indefinite integral
    # is the lowest value in coord.grid
    indefinite_integral_elements_to_full_grid!(pf, coord)
end

"""
"""
function indefinite_integral_elements_to_full_grid!(pf, coord)
    pf2d = coord.scratch_2d
    nelement_local = coord.nelement_local
    ngrid = coord.ngrid
    igrid_full = coord.igrid_full   
    j = 1 # the first element
    for i in 1:ngrid
        pf[i] = pf2d[i,j]
    end
    for j in 2:nelement_local
        ilast = igrid_full[ngrid,j-1] # the grid index of the last point in the previous element
        for k in 2:coord.ngrid
            i = coord.igrid_full[k,j] # the index in the full grid
            pf[i] = pf2d[k,j] + pf[ilast]
        end
    end
    return nothing
end

"""
    derivative!(df, f, coord, adv_fac, spectral)

Upwinding derivative.
"""
function derivative!(df, f, coord, adv_fac, spectral::discretization_info)
    # get the derivative at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_derivative!(coord, f, adv_fac, spectral)
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the derivative from the upwind element.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord, adv_fac)
end

"""
    derivative!(df, f, coord, spectral)

Non-upwinding derivative.
"""
function derivative!(df, f, coord, spectral)
    # get the derivative at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_derivative!(coord, f, spectral)
    # map the derivative from the elemental grid to the full grid;
    # at element boundaries, use the average of the derivatives from neighboring elements.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord)
end

# Special versions for 'null' coordinates with only one point
function derivative!(df, f, coord, spectral::Union{null_spatial_dimension_info,
                                                   null_velocity_dimension_info})
    df .= 0.0
    return nothing
end

function second_derivative!(d2f, f, coord, spectral)
    # computes d^2f / d(coord)^2
    # For spectral element methods, calculate second derivative by applying first
    # derivative twice, with special treatment for element boundaries

    # First derivative
    elementwise_derivative!(coord, f, spectral)
    derivative_elements_to_full_grid!(coord.scratch3, coord.scratch_2d, coord)
    # MPI reconcile code here if used with z or r coords

    # Save elementwise first derivative result
    coord.scratch2_2d .= coord.scratch_2d

    # Second derivative for element interiors
    elementwise_derivative!(coord, coord.scratch3, spectral)
    derivative_elements_to_full_grid!(d2f, coord.scratch_2d, coord)
    # MPI reconcile code here if used with z or r coords

    # Add contribution to penalise discontinuous first derivatives at element
    # boundaries. For smooth functions this would do nothing so should not affect
    # convergence of the second derivative. Aims to stabilise numerical instability when
    # spike develops at an element boundary. The coefficient is an arbitrary choice, it
    # should probably be large enough for stability but as small as possible.
    #
    # Arbitrary numerical coefficient
    C = 1.0
    function penalise_discontinuous_first_derivative!(d2f, imin, imax, df)
        # Left element boundary
        d2f[imin] += C * df[1]

        # Right element boundary
        d2f[imax] -= C * df[end]

        return nothing
    end
    @views penalise_discontinuous_first_derivative!(d2f, 1, coord.imax[1],
                                                    coord.scratch2_2d[:,1])
    for ielement ∈ 2:coord.nelement_local
        @views penalise_discontinuous_first_derivative!(d2f, coord.imin[ielement]-1,
                                                        coord.imax[ielement],
                                                        coord.scratch2_2d[:,ielement])
    end

    if coord.periodic
        # Need to get first derivatives from opposite ends of grid
        if coord.nelement_local != coord.nelement_global
            error("Distributed memory MPI not yet supported here")
        end
        d2f[1] -= C * coord.scratch2_2d[end,end]
        # With the first derivative from the opposite end of the grid, d2f[end] here
        # should be equal to d2f[1] up to rounding errors...
        @boundscheck isapprox(d2f[end] + C * coord.scratch2_2d[1,1], d2f[1]; atol=1.0e-14)
        # ...but because arithmetic operations were in a different order, there may be
        # rounding errors, so set the two ends exactly equal to ensure consistency for the
        # rest of the code - we assume that duplicate versions of the 'same point' on
        # element boundaries (due to periodic bc or distributed-MPI block boundaries) are
        # exactly equal.
        d2f[end] = d2f[1]
    else
        # For stability don't contribute to evolution at boundaries, in case these
        # points are not set by a boundary condition.
        # Full grid may be across processes and bc only applied to extreme ends of the
        # domain.
        if coord.irank == 0
            d2f[1] = 0.0
        end
        if coord.irank == coord.nrank - 1
            d2f[end] = 0.0
        end
    end
    return nothing
end

function laplacian_derivative!(d2f, f, coord, spectral)
    # computes (1/coord) d / coord ( coord d f / d(coord)) for vperp coordinate
    # For spectral element methods, calculate second derivative by applying first
    # derivative twice, with special treatment for element boundaries

    # First derivative
    elementwise_derivative!(coord, f, spectral)
    derivative_elements_to_full_grid!(coord.scratch3, coord.scratch_2d, coord)
    if coord.name == "vperp"
        # include the Jacobian
        @. coord.scratch3 *= coord.grid
    end
    # MPI reconcile code here if used with z or r coords

    # Save elementwise first derivative result
    coord.scratch2_2d .= coord.scratch_2d

    # Second derivative for element interiors
    elementwise_derivative!(coord, coord.scratch3, spectral)
    derivative_elements_to_full_grid!(d2f, coord.scratch_2d, coord)
    if coord.name == "vperp"
        # include the Jacobian
        @. d2f /= coord.grid
    end
    # MPI reconcile code here if used with z or r coords

    # Add contribution to penalise discontinuous first derivatives at element
    # boundaries. For smooth functions this would do nothing so should not affect
    # convergence of the second derivative. Aims to stabilise numerical instability when
    # spike develops at an element boundary. The coefficient is an arbitrary choice, it
    # should probably be large enough for stability but as small as possible.
    #
    # Arbitrary numerical coefficient
    C = 1.0
    function penalise_discontinuous_first_derivative!(d2f, imin, imax, df)
        # Left element boundary
        d2f[imin] += C * df[1]

        # Right element boundary
        d2f[imax] -= C * df[end]

        return nothing
    end
    @views penalise_discontinuous_first_derivative!(d2f, 1, coord.imax[1],
                                                    coord.scratch2_2d[:,1])
    for ielement ∈ 2:coord.nelement_local
        @views penalise_discontinuous_first_derivative!(d2f, coord.imin[ielement]-1,
                                                        coord.imax[ielement],
                                                        coord.scratch2_2d[:,ielement])
    end

    if coord.bc ∈ ("zero", "both_zero", "zero-no-regularity")
        # For stability don't contribute to evolution at boundaries, in case these
        # points are not set by a boundary condition.
        # Full grid may be across processes and bc only applied to extreme ends of the
        # domain.
        if coord.irank == coord.nrank - 1
            d2f[end] = 0.0
        end
    else
        error("Unsupported bc '$(coord.bc)'")
    end
    return nothing
end
"""
    mass_matrix_solve!(f, b, spectral::weak_discretization_info)

Solve
```math
M.f = b
```
for \$a\$, where \$M\$ is the mass matrix of a weak-form finite element method and \$b\$
is an input.
"""
function mass_matrix_solve! end

function second_derivative!(d2f, f, coord, spectral::weak_discretization_info)
    # obtain the RHS of numerical weak-form of the equation 
    # g = d^2 f / d coord^2, which is 
    # M * g = K * f, with M the mass matrix and K an appropriate stiffness matrix
    # by multiplying by basis functions and integrating by parts    
    mul!(coord.scratch3, spectral.K_matrix, f)

    # solve weak form matrix problem M * g = K * f to obtain g = d^2 f / d coord^2
    if coord.nrank > 1
        error("mass_matrix_solve!() does not support a "
              * "distributed coordinate")
    end
    # Do mass-matrix solve into a buffer array, to ensure that the output array is always
    # a contiguous array, not a view into another array that might have a stride bigger
    # than 1.
    mass_matrix_solve!(coord.scratch4, coord.scratch3, spectral)
    d2f .= coord.scratch4

    if coord.periodic
        # d2f[end] here should be equal to d2f[1] up to rounding errors...
        @boundscheck isapprox(d2f[end], d2f[1]; atol=1.0e-14)
        # ...but in the matrix operations arithmetic operations are not necessarily in
        # exactly the same order, there may be rounding errors, so set the two ends
        # exactly equal to ensure consistency for the rest of the code - we assume that
        # duplicate versions of the 'same point' on element boundaries (due to periodic bc
        # or distributed-MPI block boundaries) are exactly equal.
        d2f[end] = d2f[1]
    end
end

function laplacian_derivative!(d2f, f, coord, spectral::weak_discretization_info)
    # for coord.name 'vperp' obtain the RHS of numerical weak-form of the equation 
    # g = (1/coord) d/d coord ( coord  d f / d coord ), which is 
    # M * g = K * f, with M the mass matrix, and K an appropriate stiffness matrix,
    # by multiplying by basis functions and integrating by parts.
    # for all other coord.name, do exactly the same as second_derivative! above.
    mul!(coord.scratch3, spectral.L_matrix, f)

    if coord.periodic && coord.name == "vperp"
        error("laplacian_derivative!() cannot handle periodic boundaries for vperp")
    elseif coord.periodic
        if coord.nrank > 1
            error("laplacian_derivative!() cannot handle periodic boundaries for a "
                  * "distributed coordinate")
        end

        coord.scratch3[1] = 0.5 * (coord.scratch3[1] + coord.scratch3[end])
        coord.scratch3[end] = coord.scratch3[1]
    end

    # solve weak form matrix problem M * g = K * f to obtain g = d^2 f / d coord^2
    if coord.nrank > 1
        error("mass_matrix_solve!() does not support a "
              * "distributed coordinate")
    end
    mass_matrix_solve!(d2f, coord.scratch3, spectral)
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
    for i ∈ 2:ngm1
        df1d[i] = df2d[i,1]
    end
    # deal with any additional elements
    if coord.nelement_local > 1
        for ielem ∈ 2:coord.nelement_local
            for i ∈ 0:ngm1-2
                df1d[coord.imin[ielem]+i] = df2d[i+2,ielem]
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
here we choose to use the value of df from the upwind element.
"""
function reconcile_element_boundaries_upwind!(df1d, df2d, coord, adv_fac::AbstractArray{mk_float,1})
    # note that the first ngrid points are classified as belonging to the first element
    # and the next ngrid-1 points belonging to second element, etc.

    # first deal with domain boundaries
    if coord.periodic && coord.nelement_global == coord.nelement_local
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
        elseif adv_fac[coord.n] < 0.0
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
	if coord.periodic && coord.nelement_local == coord.nelement_global
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

function assign_endpoint!(df1d::AbstractArray{mk_float,Ndims},
 receive_buffer::AbstractArray{mk_float,Mdims},key::String,coord) where {Ndims,Mdims}
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
    if coord.name == "z" && Ndims==1
        df1d[j] = receive_buffer[]
        #println("ASSIGNING DATA")
    elseif coord.name == "z" && Ndims==2
        df1d[j,:] .= receive_buffer[:]
        #println("ASSIGNING DATA")
    elseif coord.name == "z" && Ndims==3
        df1d[j,:,:] .= receive_buffer[:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "z" && Ndims==4
        df1d[:,:,j,:] .= receive_buffer[:,:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "z" && Ndims==5
        df1d[:,:,j,:,:] .= receive_buffer[:,:,:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "z" && Ndims==6
        df1d[:,:,:,j,:,:] .= receive_buffer[:,:,:,:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "r" && Ndims==1
        df1d[j] = receive_buffer[]
        #println("ASSIGNING DATA")
    elseif coord.name == "r" && Ndims==2
        df1d[:,j] .= receive_buffer[:]
        #println("ASSIGNING DATA")
    elseif coord.name == "r" && Ndims==3
        df1d[:,j,:] .= receive_buffer[:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "r" && Ndims==4
        df1d[:,:,:,j] .= receive_buffer[:,:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "r" && Ndims==5
        df1d[:,:,:,j,:] .= receive_buffer[:,:,:,:]
        #println("ASSIGNING DATA")
    elseif coord.name == "r" && Ndims==6
        df1d[:,:,:,:,j,:] .= receive_buffer[:,:,:,:,:]
        #println("ASSIGNING DATA")
    else
        error("ERROR: failure to assign endpoints in reconcile_element_boundaries_MPI! (centered): coord.name: ",coord.name," Ndims: ",Ndims," key: ",key)
    end
end

@timeit_debug global_timer reconcile_element_boundaries_MPI!(
                  df1d::AbstractArray{mk_float,Ndims},
                  dfdx_lower_endpoints::AbstractArray{mk_float,Mdims},
                  dfdx_upper_endpoints::AbstractArray{mk_float,Mdims},
                  receive_buffer1::AbstractArray{mk_float,Mdims},
                  receive_buffer2::AbstractArray{mk_float,Mdims},
                  coord) where {Ndims,Mdims} = begin
	
    # synchronize buffers
    # -- this all-to-all block communicate here requires that this function is NOT called from within a parallelised loop
    # -- or from a @serial_region or from an if statment isolating a single rank on a block
    @_block_synchronize()
    @serial_region begin

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

        # pass data from irank -> irank + 1, receive data from irank - 1
        rreq1 = MPI.Irecv!(receive_buffer1, comm; source=coord.prevrank, tag=1)
        sreq1 = MPI.Isend(dfdx_upper_endpoints, comm; dest=coord.nextrank, tag=1)

        # pass data from irank -> irank - 1, receive data from irank + 1
        rreq2 = MPI.Irecv!(receive_buffer2, comm; source=coord.nextrank, tag=2)
        sreq2 = MPI.Isend(dfdx_lower_endpoints, comm; dest=coord.prevrank, tag=2)
        stats = MPI.Waitall([rreq1, sreq1, rreq2, sreq2])

        # now update receive buffers, taking into account the reconciliation
        if irank == 0
            if coord.periodic
                #update the extreme lower endpoint with data from irank = nrank -1	
                receive_buffer1 .= 0.5*(receive_buffer1 .+ dfdx_lower_endpoints)
            else #directly use value from Cheb
                receive_buffer1 .= dfdx_lower_endpoints
            end
        else # enforce continuity at lower endpoint
            receive_buffer1 .= 0.5*(receive_buffer1 .+ dfdx_lower_endpoints)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        assign_endpoint!(df1d,receive_buffer1,"lower",coord)

        if irank == nrank-1
            if coord.periodic
                #update the extreme upper endpoint with data from irank = 0
                receive_buffer2 .= 0.5*(receive_buffer2 .+ dfdx_upper_endpoints)
            else #directly use value from Cheb
                receive_buffer2 .= dfdx_upper_endpoints
            end
        else # enforce continuity at upper endpoint
            receive_buffer2 .= 0.5*(receive_buffer2 .+ dfdx_upper_endpoints)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        assign_endpoint!(df1d,receive_buffer2,"upper",coord)

    end
    # synchronize buffers
    @_block_synchronize()
end

function apply_adv_fac!(buffer::AbstractArray{mk_float,Ndims},adv_fac::AbstractArray{mk_float,Ndims},endpoints::AbstractArray{mk_float,Ndims},sgn::mk_int) where Ndims
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
				#do nothing
			else #average values
				buffer[i] = 0.5*(buffer[i] + endpoints[i])
			end
		end
		
	end
	
@timeit_debug global_timer reconcile_element_boundaries_MPI!(
                  df1d::AbstractArray{mk_float,Ndims},
                  adv_fac_lower_endpoints::AbstractArray{mk_float,Mdims},
                  adv_fac_upper_endpoints::AbstractArray{mk_float,Mdims},
                  dfdx_lower_endpoints::AbstractArray{mk_float,Mdims},
                  dfdx_upper_endpoints::AbstractArray{mk_float,Mdims},
                  receive_buffer1::AbstractArray{mk_float,Mdims},
                  receive_buffer2::AbstractArray{mk_float,Mdims},
                  coord) where {Ndims,Mdims} = begin
	
    # synchronize buffers
    # -- this all-to-all block communicate here requires that this function is NOT called from within a parallelised loop
    # -- or from a @serial_region or from an if statment isolating a single rank on a block
    @_block_synchronize()
    #if block_rank[] == 0 # lead process on this shared-memory block
    @serial_region begin
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

        # send highest end point on THIS rank
        # pass data from irank -> irank + 1, receive data from irank - 1
        rreq1 = MPI.Irecv!(receive_buffer1, comm; source=coord.prevrank, tag=1)
        sreq1 = MPI.Isend(dfdx_upper_endpoints, comm; dest=coord.nextrank, tag=1)

        # send lowest end point on THIS rank
        # pass data from irank -> irank - 1, receive data from irank + 1
        rreq2 = MPI.Irecv!(receive_buffer2, comm; source=coord.nextrank, tag=2)
        sreq2 = MPI.Isend(dfdx_lower_endpoints, comm; dest=coord.prevrank, tag=2)
        stats = MPI.Waitall([rreq1, sreq1, rreq2, sreq2])

        # now update receive buffers, taking into account the reconciliation
        if irank == 0
            if coord.periodic
                # depending on adv_fac, update the extreme lower endpoint with data from irank = nrank -1	
                apply_adv_fac!(receive_buffer1,adv_fac_lower_endpoints,dfdx_lower_endpoints,1)
            else # directly use value from Cheb at extreme lower point
                receive_buffer1 .= dfdx_lower_endpoints
            end
        else # depending on adv_fac, update the lower endpoint with data from irank = nrank -1	
            apply_adv_fac!(receive_buffer1,adv_fac_lower_endpoints,dfdx_lower_endpoints,1)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        assign_endpoint!(df1d,receive_buffer1,"lower",coord)

        if irank == nrank-1
            if coord.periodic
                # depending on adv_fac, update the extreme upper endpoint with data from irank = 0
                apply_adv_fac!(receive_buffer2,adv_fac_upper_endpoints,dfdx_upper_endpoints,-1)
            else #directly use value from Cheb
                receive_buffer2 .= dfdx_upper_endpoints
            end
        else # enforce continuity at upper endpoint
            apply_adv_fac!(receive_buffer2,adv_fac_upper_endpoints,dfdx_upper_endpoints,-1)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        assign_endpoint!(df1d,receive_buffer2,"upper",coord)

    end
    # synchronize buffers
    @_block_synchronize()
end

# Special version for pdf_electron with no r-dimension, which has the same number of
# dimensions as an ion/neutral moment variable, but different dimensions.
@timeit_debug global_timer reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
                  df1d::AbstractArray{mk_float,3},
                  dfdx_lower_endpoints::AbstractArray{mk_float,2},
                  dfdx_upper_endpoints::AbstractArray{mk_float,2},
                  receive_buffer1::AbstractArray{mk_float,2},
                  receive_buffer2::AbstractArray{mk_float,2}, coord) = begin
	
    # synchronize buffers
    # -- this all-to-all block communicate here requires that this function is NOT called from within a parallelised loop
    # -- or from a @serial_region or from an if statment isolating a single rank on a block
    @_block_synchronize()
    #if block_rank[] == 0 # lead process on this shared-memory block
    @serial_region begin

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

        # pass data from irank -> irank + 1, receive data from irank - 1
        rreq1 = MPI.Irecv!(receive_buffer1, comm; source=coord.prevrank, tag=1)
        sreq1 = MPI.Isend(dfdx_upper_endpoints, comm; dest=coord.nextrank, tag=1)

        # pass data from irank -> irank - 1, receive data from irank + 1
        rreq2 = MPI.Irecv!(receive_buffer2, comm; source=coord.nextrank, tag=2)
        sreq2 = MPI.Isend(dfdx_lower_endpoints, comm; dest=coord.prevrank, tag=2)
        stats = MPI.Waitall([rreq1, sreq1, rreq2, sreq2])

        # now update receive buffers, taking into account the reconciliation
        if irank == 0
            if coord.periodic
                #update the extreme lower endpoint with data from irank = nrank -1	
                receive_buffer1 .= 0.5*(receive_buffer1 .+ dfdx_lower_endpoints)
            else #directly use value from Cheb
                receive_buffer1 .= dfdx_lower_endpoints
            end
        else # enforce continuity at lower endpoint
            receive_buffer1 .= 0.5*(receive_buffer1 .+ dfdx_lower_endpoints)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        @views df1d[:,:,1] .= receive_buffer1

        if irank == nrank-1
            if coord.periodic
                #update the extreme upper endpoint with data from irank = 0
                receive_buffer2 .= 0.5*(receive_buffer2 .+ dfdx_upper_endpoints)
            else #directly use value from Cheb
                receive_buffer2 .= dfdx_upper_endpoints
            end
        else # enforce continuity at upper endpoint
            receive_buffer2 .= 0.5*(receive_buffer2 .+ dfdx_upper_endpoints)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        @views df1d[:,:,end] .= receive_buffer2

    end
    # synchronize buffers
    @_block_synchronize()
end

# Special version for pdf_electron with no r-dimension, which has the same number of
# dimensions as an ion/neutral moment variable, but different dimensions.
@timeit_debug global_timer reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
                  df1d::AbstractArray{mk_float,3},
                  adv_fac_lower_endpoints::AbstractArray{mk_float,2},
                  adv_fac_upper_endpoints::AbstractArray{mk_float,2},
                  dfdx_lower_endpoints::AbstractArray{mk_float,2},
                  dfdx_upper_endpoints::AbstractArray{mk_float,2},
                  receive_buffer1::AbstractArray{mk_float,2},
                  receive_buffer2::AbstractArray{mk_float,2}, coord) = begin
	
    # synchronize buffers
    # -- this all-to-all block communicate here requires that this function is NOT called from within a parallelised loop
    # -- or from a @serial_region or from an if statment isolating a single rank on a block
    @_block_synchronize()
    #if block_rank[] == 0 # lead process on this shared-memory block
    @serial_region begin
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

        # send highest end point on THIS rank
        # pass data from irank -> irank + 1, receive data from irank - 1
        rreq1 = MPI.Irecv!(receive_buffer1, comm; source=coord.prevrank, tag=1)
        sreq1 = MPI.Isend(dfdx_upper_endpoints, comm; dest=coord.nextrank, tag=1)

        # send lowest end point on THIS rank
        # pass data from irank -> irank - 1, receive data from irank + 1
        rreq2 = MPI.Irecv!(receive_buffer2, comm; source=coord.nextrank, tag=2)
        sreq2 = MPI.Isend(dfdx_lower_endpoints, comm; dest=coord.prevrank, tag=2)
        stats = MPI.Waitall([rreq1, sreq1, rreq2, sreq2])

        # now update receive buffers, taking into account the reconciliation
        if irank == 0
            if coord.periodic
                # depending on adv_fac, update the extreme lower endpoint with data from irank = nrank -1	
                apply_adv_fac!(receive_buffer1,adv_fac_lower_endpoints,dfdx_lower_endpoints,1)
            else # directly use value from Cheb at extreme lower point
                receive_buffer1 .= dfdx_lower_endpoints
            end
        else # depending on adv_fac, update the lower endpoint with data from irank = nrank -1	
            apply_adv_fac!(receive_buffer1,adv_fac_lower_endpoints,dfdx_lower_endpoints,1)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        @views df1d[:,:,1] .= receive_buffer1

        if irank == nrank-1
            if coord.periodic
                # depending on adv_fac, update the extreme upper endpoint with data from irank = 0
                apply_adv_fac!(receive_buffer2,adv_fac_upper_endpoints,dfdx_upper_endpoints,-1)
            else #directly use value from Cheb
                receive_buffer2 .= dfdx_upper_endpoints
            end
        else # enforce continuity at upper endpoint
            apply_adv_fac!(receive_buffer2,adv_fac_upper_endpoints,dfdx_upper_endpoints,-1)
        end
        #now update the df1d array -- using a slice appropriate to the dimension reconciled
        @views df1d[:,:,end] .= receive_buffer2

    end
    # synchronize buffers
    @_block_synchronize()
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
Compute the 1D integral `∫dv prefactor(v)*integrand`

In this variant `v` should be a `coordinate` object.
"""
function integral(prefactor::Function, integrand, v)
    @boundscheck v.n == length(integrand) || throw(BoundsError(integrand))
    v_grid = v.grid
    wgts = v.wgts
    integral = 0.0
    @inbounds for i ∈ eachindex(v_grid)
        integral += prefactor(v[i]) * integrand[i] * wgts[i]
    end
    return integral
end

"""
Compute the 2D integral `∫d^2vperp.dvpa prefactor(vperp,vpa)*integrand`

In this variant `vperp` and `vpa` should be `coordinate` objects.

Note that vperp_wgts contains the extra factor of vperp required for the
Jacobian.
"""
function integral(prefactor::Function, integrand, vperp, vpa)
    @boundscheck (vpa.n, vperp.n) == size(integrand) || throw(BoundsError(integrand))
    vperp_grid = vperp.grid
    vperp_wgts = vperp.wgts
    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts
    integral = 0.0
    for ivperp ∈ eachindex(vperp_grid), ivpa ∈ eachindex(vpa_grid)
        integral += prefactor(vperp_grid[ivperp], vpa_grid[ivpa]) *
                    integrand[ivpa, ivperp] * vperp_wgts[ivperp] * vpa_wgts[ivpa]
    end
    return integral
end

"""
Compute the 3D integral `∫dvzeta.dvr.dvz prefactor(vzeta,vr,vz)*integrand`

In this variant `vzeta`, `vr`, and `vz` should be `coordinate` objects.
"""
function integral(prefactor::Function, integrand, vzeta, vr, vz)
    @boundscheck (vz.n, vr.n, vzeta.n) == size(integrand) || throw(BoundsError(integrand))
    vzeta_grid = vzeta.grid
    vzeta_wgts = vzeta.wgts
    vr_grid = vr.grid
    vr_wgts = vr.wgts
    vz_grid = vz.grid
    vz_wgts = vz.wgts
    integral = 0.0
    for ivzeta ∈ eachindex(vzeta_grid), ivr ∈ eachindex(vr_grid), ivz ∈ eachindex(vz_grid)
        integral += prefactor(vzeta_grid[ivzeta], vr_grid[ivr], vz_grid[ivz]) *
                    integrand[ivz, ivr, ivzeta] * vzeta_wgts[ivzeta] * vr_wgts[ivr] *
                    vz_wgts[ivz]
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


# Indefinite integral routines

"""
A function that takes the indefinite integral in each element of `coord.grid`,
leaving the result (element-wise) in `coord.scratch_2d`.
"""
function elementwise_indefinite_integration!(coord, ff, spectral::discretization_info)
    # the primitive of f
    pf = coord.scratch_2d
    # define local variable nelement for convenience
    nelement = coord.nelement_local
    # check array bounds
    @boundscheck nelement == size(pf,2) && coord.ngrid == size(pf,1) || throw(BoundsError(pf))

    # variable k will be used to avoid double counting of overlapping point
    k = 0
    j = 1 # the first element
    imin = coord.imin[j]-k
    # imax is the maximum index on the full grid for this (jth) element
    imax = coord.imax[j]
    if coord.radau_first_element && coord.irank == 0 # differentiate this element with the Radau scheme
        @views mul!(pf[:,j],spectral.radau.indefinite_integration_matrix[:,:],ff[imin:imax])
    else #differentiate using the Lobatto scheme
        @views mul!(pf[:,j],spectral.lobatto.indefinite_integration_matrix[:,:],ff[imin:imax])
    end
    # transform back to the physical coordinate scale
    for i in 1:coord.ngrid
        pf[i,j] *= coord.element_scale[j]
    end
    # calculate the derivative on each element
    @inbounds for j ∈ 2:nelement
        k = 1
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        @views mul!(pf[:,j],spectral.lobatto.indefinite_integration_matrix[:,:],ff[imin:imax])
        # transform back to the physical coordinate scale
        for i in 1:coord.ngrid
            pf[i,j] *= coord.element_scale[j]
        end
    end
    return nothing
end

end
