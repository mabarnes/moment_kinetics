module vpa_advection

export vpa_advection!
export update_speed_vpa!

using moment_kinetics_input: advection_speed, advection_speed_option
using semi_lagrange: find_approximate_characteristic!
using time_advance: advance_f_local!
using source_terms: update_boundary_indices!
using em_fields: update_phi!
using chebyshev: update_fcheby!
using chebyshev: update_df_chebyshev!
using initial_conditions: enforce_vpa_boundary_condition!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function vpa_advection!(ff, ff_scratch, phi, moments, SL, source, vpa, z,
	use_semi_lagrange, dt, vpa_spectral, z_spectral)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
	@boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
	@boundscheck size(ff_scratch,3) == 3 || throw(BoundsError(ff_scratch))
    # get the updated speed along the vpa direction
    update_speed_vpa!(source, phi, moments, ff, vpa, z, z_spectral)
	# update the upwind/downwind boundary indices and upwind_increment
	update_boundary_indices!(source)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for iz ∈ 1:z.n
            find_approximate_characteristic!(SL[iz], source[iz], vpa, dt)
        end
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
	ff_scratch[:,:,1] .= ff
    for j ∈ 1:jend
        for iz ∈ 1:z.n
			@views advance_f_local!(ff_scratch[iz,:,j+1], ff_scratch[iz,:,j],
				ff[iz,:], SL[iz], source[iz], vpa, dt, vpa_spectral, j)
		end
		enforce_vpa_boundary_condition!(view(ff_scratch,:,:,j+1), vpa.bc, source)
        moments.dens_updated = false ; moments.ppar_updated = false
        if j != jend
			# calculate the advection speed corresponding to current f
			update_speed_vpa!(source, phi, moments, view(ff_scratch,:,:,j+1), vpa, z, z_spectral)
			# update the upwind/downwind boundary indices and upwind_increment
			# NB: not sure if this will work properly with SL method at the moment
			# NB: if the speed is actually time-dependent
			update_boundary_indices!(source)
        end
    end
    @inbounds @fastmath begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:z.n
                ff[iz,ivpa] = 0.5*(ff_scratch[iz,ivpa,2] + ff_scratch[iz,ivpa,3])
            end
        end
    end
end
# for use with finite difference scheme
function vpa_advection!(ff, ff_scratch, phi, moments, SL, source, vpa, z,
	use_semi_lagrange, dt)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(f,1) == z.n || throw(BoundsError(f))
    @boundscheck size(f,2) == vpa.n || throw(BoundsError(f))
	@boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
	@boundscheck size(ff_scratch,3) == 3 || throw(BoundsError(ff_scratch))
    # get the updated speed along the vpa direction
	update_speed_vpa!(source, phi, moments, ff, vpa, z)
	# update the upwind/downwind boundary indices and upwind_increment
	update_boundary_indices!(source)
    # if using interpolation-free Semi-Lagrange,
    # follow characteristics backwards in time from level m+1 to level m
    # to get departure points.  then find index of grid point nearest
    # the departure point at time level m and use this to define
    # an approximate characteristic
    if use_semi_lagrange
        for iz ∈ 1:z.n
            find_approximate_characteristic!(SL[iz], source[iz], vpa, dt)
        end
    end
    # Heun's method (RK2) for explicit time advance
    jend = 2
	ff_scratch[:,:,1] .= ff
    for j ∈ 1:jend
        for iz ∈ 1:z.n
			@views advance_f_local!(ff_scratch[iz,:,j+1], ff_scratch[iz,:,j],
				ff[iz,:], SL[iz], source[iz], vpa, dt, j)
        end
		enforce_vpa_boundary_condition!(view(ff_scratch,:,:,j+1), vpa.bc, source)
        moments.dens_updated = false ; moments.ppar_updated = false
        # calculate the advection speed corresponding to current f
		if j != jend
			update_speed_vpa!(source, phi, moments, view(ff_scratch,:,:,j+1), vpa, z)
			# update the upwind/downwind boundary indices and upwind_increment
			update_boundary_indices!(source)
		end
    end
    @inbounds @fastmath begin
        for ivpa ∈ 1:vpa.n
            for iz ∈ 1:nz
                ff[iz,ivpa] = 0.5*(ff_scratch[iz,ivpa,2] + ff_scratch[iz,ivpa,3])
            end
        end
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(source, phi, moments, ff, vpa, z, z_spectral)
    @boundscheck z.n == size(source,1) || throw(BoundsError(source))
    @boundscheck vpa.n == size(source[1].speed,1) || throw(BoundsError(speed))
    if advection_speed_option == "default"
		# dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(source, phi, moments, ff, vpa, z, z_spectral)
		#update_speed_constant!(source, vpa.n, z.n)
    elseif advection_speed_option == "constant"
		# dvpa/dt = constant
		update_speed_constant!(source, vpa.n, z.n)
    elseif advection_speed_option == "linear"
		# dvpa/dt = constant ⋅ (vpa + L_vpa/2)
		update_speed_linear!(source, vpa, z.n)
    end
    return nothing
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(source, phi, moments, ff, vpa, z)
    @boundscheck z.n == size(source,1) || throw(BoundsError(source))
    @boundscheck vpa.n == size(source[1].speed,1) || throw(BoundsError(speed))
    if advection_speed_option == "default"
		# dvpa/dt = Ze/m ⋅ E_parallel
		println("speed default option not setup for finite_differences")
        #update_speed_default!(source, phi, moments, ff, vpa, z)
		update_speed_constant!(source, vpa.n, z.n)
    elseif advection_speed_option == "constant"
		# dvpa/dt = constant
		update_speed_constant!(source, vpa.n, z.n)
    elseif advection_speed_option == "linear"
		# dvpa/dt = constant ⋅ (vpa + L_vpa/2)
		update_speed_linear!(source, vpa, z.n)
    end
    return nothing
end
# update the advection speed dvpa/dt = Ze/m E_parallel
function update_speed_default!(source, phi, moments, ff, vpa, z, z_spectral)
	update_phi!(phi, moments, ff, vpa, z.n)
	# get the Chebyshev coefficients for phi and store in z_spectral.f
	update_fcheby!(z_spectral, phi, z)
	# dphi/dz is calculated and stored in z.scratch
	update_df_chebyshev!(z.scratch, z_spectral, z)
	@inbounds @fastmath begin
		for iz ∈ 1:z.n
			for ivpa ∈ 1:vpa.n
				# NB: the default option with dvpa/dt = Epar not yet working
				source[iz].speed[ivpa] = advection_speed #-z.scratch[iz]
			end
		end
	end
end
# update the advection speed dvpa/dt = constant
function update_speed_constant!(source, nvpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:nvpa
				source[iz].speed[ivpa] = advection_speed
			end
		end
	end
end
# update the advection speed dvpa/dt = const*(vpa + L/2)
function update_speed_linear(source, vpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:vpa.n
				source[iz].speed[ivpa] = advection_speed*(vpa.grid[ivpa]+0.5*vpa.L)
			end
		end
	end
end

end
