module charge_exchange

export charge_exchange_collisions!

using velocity_moments: update_density!

function charge_exchange_collisions!(ff, ff_scratch, moments, composition,
	vpa, charge_exchange_frequency, nz, dt)
	# define n_ion_species and n_neutral_species for convenience
	n_species = composition.n_species
	n_ion_species = composition.n_ion_species
	n_neutral_species = composition.n_neutral_species
	# check array bounds to avoid cost of doing so in loops below
	@boundscheck nz == size(ff_scratch,1) || throw(BoundsError(ff_scratch))
	@boundscheck vpa.n == size(ff_scratch,2) || throw(BoundsError(ff_scratch))
	@boundscheck n_species == size(ff_scratch,3) || throw(BoundsError(ff_scratch))
	@boundscheck 3 == size(ff_scratch,4) || throw(BoundsError(ff_scratch))
	@boundscheck nz == size(moments.dens,1) || throw(BoundsError(moments.dens))
	@boundscheck n_species == size(moments.dens,2) || throw(BoundsError(moments.dens))
	# Heun's method (RK2) for explicit time advance
    jend = 2
	@inbounds for j ∈ 1:jend+1
		ff_scratch[:,:,:,j] .= ff
	end
    for j ∈ 1:jend
		# make sure all densities needed for the charge exchange collisions are up-to-date
		for is ∈ 1:n_ion_species+n_neutral_species
			if moments.dens_updated[is] == false
				@views update_density!(moments.dens[:,is], vpa.scratch, ff_scratch[:,:,is,j], vpa, nz)
				moments.dens_updated[is] = true
			end
		end
		#ff_scratch[:,:,:,j+1] .= 0.0
		# apply CX collisions to all ion species
		@inbounds for is ∈ 1:n_ion_species
			# for each ion species, obtain affect of charge exchange collisions
			# with all of the neutral species
			for isp ∈ 1:n_neutral_species
				#cxfac = dt*charge_exchange_frequency[is,isp]
				cxfac = dt*charge_exchange_frequency
				for ivpa ∈ 1:vpa.n
					for iz ∈ 1:nz
						ff_scratch[iz,ivpa,is,j+1] += dt*charge_exchange_frequency *(
							ff_scratch[iz,ivpa,isp+n_ion_species,j]*moments.dens[iz,is]
							- ff_scratch[iz,ivpa,is,j]*moments.dens[iz,isp+n_ion_species])
					end
				end
			end
			moments.dens_updated[is] = false ; moments.ppar_updated[is] = false
#			println("CX_ions: ", " is: ", is, "  scratch: ", sum(ff_scratch[:,:,is,j+1]),
#				"  cxfac: ", dt*charge_exchange_frequency, "  scratch_old: ", sum(ff_scratch[:,:,:,j]),
#				"  dens: ", sum(moments.dens))
		end

		# apply CX collisions to all neutral species
		@inbounds for is ∈ 1:n_neutral_species
			# for each neutral species, obtain affect of charge exchange collisions
			# with all of the ion species
			for isp ∈ 1:n_ion_species
				#cxfac = dt*charge_exchange_frequency[isp,is]
				cxfac = dt*charge_exchange_frequency
				for ivpa ∈ 1:vpa.n
					for iz ∈ 1:nz
						ff_scratch[iz,ivpa,is+n_ion_species,j+1] += dt*charge_exchange_frequency*(
							ff_scratch[iz,ivpa,isp,j]*moments.dens[iz,is+n_ion_species]
							- ff_scratch[iz,ivpa,is+n_ion_species,j]*moments.dens[iz,isp])
					end
				end
			end
			moments.dens_updated[is] = false ; moments.ppar_updated[is] = false
#			println("CX_neutrals: ", " is: ", is, "  scratch: ", sum(ff_scratch[:,:,is+n_ion_species,j+1]))
		end
	end
	@inbounds @fastmath begin
		for is ∈ 1:n_ion_species+n_neutral_species
			for ivpa ∈ 1:vpa.n
				for iz ∈ 1:nz
					ff[iz,ivpa,is] = 0.5*(ff_scratch[iz,ivpa,is,2] + ff_scratch[iz,ivpa,is,3])
				end
			end
		end
	end
end

end
