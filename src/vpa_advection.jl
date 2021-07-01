function vpa_advection!(f_out, fvec_in, ff, fields, moments, SL, advect,
	vpa, z, use_semi_lagrange, dt, t, vpa_spectral, z_spectral, composition, CX_frequency, istage)

	# only have a parallel acceleration term for neutrals if using the peculiar velocity
	# wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.
	# define the number of evolved species accordingly
	if moments.evolve_upar
		nspecies_accelerated = composition.n_species
	else
		nspecies_accelerated = composition.n_ion_species
	end
	# calculate the advection speed corresponding to current f
	update_speed_vpa!(advect, fields, fvec_in, moments, vpa, z, composition, CX_frequency, t, z_spectral)
	for is ∈ 1:nspecies_accelerated
		# update the upwind/downwind boundary indices and upwind_increment
		# NB: not sure if this will work properly with SL method at the moment
		# NB: if the speed is actually time-dependent
		update_boundary_indices!(view(advect,:,is))
		# if using interpolation-free Semi-Lagrange,
		# follow characteristics backwards in time from level m+1 to level m
		# to get departure points.  then find index of grid point nearest
		# the departure point at time level m and use this to define
		# an approximate characteristic
		if use_semi_lagrange
			for iz ∈ 1:z.n
				find_approximate_characteristic!(SL[iz], advect[iz,is], vpa, dt)
			end
		end
		for iz ∈ 1:z.n
			@views advance_f_local!(f_out[iz,:,is], fvec_in.pdf[iz,:,is],
				ff[iz,:,is], SL[iz], advect[iz,is], vpa, dt, istage, vpa_spectral,
				use_semi_lagrange)
		end
		#@views enforce_vpa_boundary_condition!(f_out[:,:,is], vpa.bc, advect[:,is])
	end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(advect, fields, fvec, moments, vpa, z, composition, CX_frequency, t, z_spectral)
    @boundscheck z.n == size(advect,1) || throw(BoundsError(advect))
	#@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
	@boundscheck composition.n_species == size(advect,2) || throw(BoundsError(advect))
	@boundscheck vpa.n == size(advect[1,1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
		# dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(advect, fields, fvec, moments, vpa, z, composition, CX_frequency, t, z_spectral)
    elseif vpa.advection.option == "constant"
		# dvpa/dt = constant
		for is ∈ 1:composition.n_ion_species
			update_speed_constant!(view(advect,:,is), vpa, z.n)
		end
    elseif vpa.advection.option == "linear"
		# dvpa/dt = constant ⋅ (vpa + L_vpa/2)
		for is ∈ 1:composition.n_ion_species
			update_speed_linear!(view(advect,:,is), vpa, z.n)
		end
	end
	#@inbounds begin
		#for is ∈ 1:composition.n_ion_species
		for is ∈ 1:composition.n_species
			for iz ∈ 1:z.n
				@. advect[iz,is].modified_speed = advect[iz,is].speed
			end
		end
	#end
    return nothing
end
function update_speed_default!(advect, fields, fvec, moments, vpa, z, composition, CX_frequency, t, z_spectral)
	if moments.evolve_ppar
		for is ∈ 1:composition.n_species
			# get d(ppar)/dz
			derivative!(z.scratch, view(fvec.ppar,:,is), z, z_spectral)
			# update parallel acceleration to account for parallel derivative of parallel pressure
			# NB: no vpa-dependence so compute for first entry in vpa and copy into remaining entries
			for iz ∈ 1:z.n
				advect[iz,is].speed[1] = z.scratch[iz]/(fvec.density[iz,is]*moments.vth[iz,is])
				for ivpa ∈ 2:vpa.n
					advect[iz,is].speed[ivpa] = advect[iz,is].speed[1]
				end
			end
			# calculate d(qpar)/dz
			derivative!(z.scratch, view(moments.qpar,:,is), z, z_spectral)
			# update parallel acceleration to account for (wpar/2*ppar)*dqpar/dz
			for iz ∈ 1:z.n
				for ivpa ∈ 1:vpa.n
					advect[iz,is].speed[ivpa] += 0.5*vpa.grid[ivpa]*z.scratch[iz]/fvec.ppar[iz,is]
				end
			end
			# calculate d(vth)/dz
			derivative!(z.scratch, view(moments.vth,:,is), z, z_spectral)
			# update parallel acceleration to account for -wpar^2 * d(vth)/dz term
			for iz ∈ 1:z.n
				for ivpa ∈ 1:vpa.n
					advect[iz,is].speed[ivpa] -= vpa.grid[ivpa]^2*z.scratch[iz]
				end
			end
		end
		# add in contributions from charge exchange collisions
		if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
			for is ∈ 1:composition.n_ion_species
				for isn ∈ 1:composition.n_neutral_species
					isp = composition.n_ion_species + isn
					for iz ∈ 1:z.n
						@. advect[iz,is].speed += CX_frequency *
							(0.5*vpa.grid/fvec.ppar[iz,is] * (fvec.density[iz,isp]*fvec.ppar[iz,is]
							- fvec.density[iz,is]*fvec.ppar[iz,isp])
							- fvec.density[iz,isp] * (fvec.upar[iz,isp]-fvec.upar[iz,is])/moments.vth[iz,is])
					end
				end
			end
			for isn ∈ 1:composition.n_neutral_species
				is = isn + composition.n_ion_species
				for isp ∈ 1:composition.n_ion_species
					for iz ∈ 1:z.n
						@. advect[iz,is].speed += CX_frequency *
							(0.5*vpa.grid/fvec.ppar[iz,is] * (fvec.density[iz,isp]*fvec.ppar[iz,is]
							- fvec.density[iz,is]*fvec.ppar[iz,isp])
							- fvec.density[iz,isp] * (fvec.upar[iz,isp]-fvec.upar[iz,is])/moments.vth[iz,is])
					end
				end
			end
		end
	elseif moments.evolve_upar
		for is ∈ 1:composition.n_species
			# get d(ppar)/dz
			derivative!(z.scratch, view(fvec.ppar,:,is), z, z_spectral)
			# update parallel acceleration to account for parallel derivative of parallel pressure
			# NB: no vpa-dependence so compute for first entry in vpa and copy into remaining entries
			for iz ∈ 1:z.n
				advect[iz,is].speed[1] = z.scratch[iz]/fvec.density[iz,is]
				for ivpa ∈ 2:vpa.n
					advect[iz,is].speed[ivpa] = advect[iz,is].speed[1]
				end
			end
			# calculate d(upar)/dz
			derivative!(z.scratch, view(fvec.upar,:,is), z, z_spectral)
			# update parallel acceleration to account for -wpar*dupar/dz
			for iz ∈ 1:z.n
				for ivpa ∈ 1:vpa.n
					advect[iz,is].speed[ivpa] -= vpa.grid[ivpa]*z.scratch[iz]
				end
			end
		end
		# if neutrals present and charge exchange frequency non-zero,
		# account for collisional friction between ions and neutrals
		if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
			# include contribution to ion acceleration due to collisional friction with neutrals
			for is ∈ 1:composition.n_ion_species
				for isp ∈ 1:composition.n_neutral_species
					# get the absolute species index for the neutral species
					isn = composition.n_ion_species + isp
					for iz ∈ 1:z.n
						tmp = -CX_frequency*fvec.density[iz,isn]*(fvec.upar[iz,isn]-fvec.upar[iz,is])
						for ivpa ∈ 1:vpa.n
							advect[iz,is].speed[ivpa] += tmp
						end
					end
				end
			end
			# include contribution to neutral acceleration due to collisional friction with ions
			for isp ∈ 1:composition.n_neutral_species
				for isi ∈ 1:composition.n_ion_species
					# get the absolute species index for the neutral species
					is = composition.n_ion_species + isp
					for iz ∈ 1:z.n
						tmp = -CX_frequency*fvec.density[iz,isi]*(fvec.upar[iz,isi]-fvec.upar[iz,is])
						for ivpa ∈ 1:vpa.n
							advect[iz,is].speed[ivpa] += tmp
						end
					end
				end
			end
		end
	else
		# update the electrostatic potential phi
		update_phi!(fields, fvec, vpa, z.n, composition)
		# calculate the derivative of phi with respect to z;
		# the value at element boundaries is taken to be the average of the values
		# at neighbouring elements
		derivative!(z.scratch, fields.phi, z, z_spectral)
		# advection velocity in vpa is -dphi/dz = -z.scratch
		@inbounds @fastmath begin
			for is ∈ 1:composition.n_ion_species
				for iz ∈ 1:z.n
					for ivpa ∈ 1:vpa.n
						advect[iz,is].speed[ivpa] = -0.5*z.scratch[iz]
					end
				end
			end
		end
	end
end
# update the advection speed dvpa/dt = constant
function update_speed_constant!(advect, vpa, nz)
	#@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:vpa.n
				advect[iz].speed[ivpa] = vpa.advection.constant_speed
			end
		end
	#end
end
# update the advection speed dvpa/dt = const*(vpa + L/2)
function update_speed_linear(advect, vpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:vpa.n
				advect[iz].speed[ivpa] = vpa.advection.constant_speed*(vpa.grid[ivpa]+0.5*vpa.L)
			end
		end
	end
end
