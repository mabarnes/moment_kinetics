module charge_exchange

export charge_exchange_collisions!

using velocity_moments: update_density!, reset_moments_status!

function charge_exchange_collisions!(ff, ff_scratch, moments, composition,
	vpa, charge_exchange_frequency, nz, dt, n_rk_stages)
	# define n_ion_species and n_neutral_species for convenience
	n_species = composition.n_species
	n_ion_species = composition.n_ion_species
	n_neutral_species = composition.n_neutral_species
	# check array bounds to avoid cost of doing so in loops below
	@boundscheck nz == size(ff_scratch,1) || throw(BoundsError(ff_scratch))
	@boundscheck vpa.n == size(ff_scratch,2) || throw(BoundsError(ff_scratch))
	@boundscheck n_species == size(ff_scratch,3) || throw(BoundsError(ff_scratch))
	@boundscheck n_rk_stages+1 == size(ff_scratch,4) || throw(BoundsError(ff_scratch))
	@boundscheck nz == size(moments.dens,1) || throw(BoundsError(moments.dens))
	@boundscheck n_species == size(moments.dens,2) || throw(BoundsError(moments.dens))
	# SSP RK for explicit time advance
	@inbounds for istage ∈ 1:n_rk_stages+1
		ff_scratch[:,:,:,istage] .= ff
	end
    for istage ∈ 1:n_rk_stages
		@views charge_exchange_single_stage!(ff_scratch[:,:,:,istage+1],
			ff_scratch[:,:,:,istage], ff, moments, n_ion_species,
			n_neutral_species, vpa, charge_exchange_frequency, nz, dt)
		reset_moments_status!(moments)
	end
end
function charge_exchange_single_stage!(f_out, f_in, ff, moments, n_ion_species,
	n_neutral_species, vpa, charge_exchange_frequency, nz, dt)
	# make sure all densities needed for the charge exchange collisions are up-to-date
	for is ∈ 1:n_ion_species+n_neutral_species
		if moments.dens_updated[is] == false
			@views update_density!(moments.dens[:,is], vpa.scratch, f_in[:,:,is], vpa, nz)
			moments.dens_updated[is] = true
		end
	end
	# apply CX collisions to all ion species
	@inbounds for is ∈ 1:n_ion_species
		# for each ion species, obtain affect of charge exchange collisions
		# with all of the neutral species
		for isp ∈ 1:n_neutral_species
			#cxfac = dt*charge_exchange_frequency[is,isp]
			#cxfac = dt*charge_exchange_frequency
			for ivpa ∈ 1:vpa.n
				for iz ∈ 1:nz
					f_out[iz,ivpa,is] += dt*charge_exchange_frequency *(
						f_in[iz,ivpa,isp+n_ion_species]*moments.dens[iz,is]
						- f_in[iz,ivpa,is]*moments.dens[iz,isp+n_ion_species])
				end
			end
		end
	end

	# apply CX collisions to all neutral species
	@inbounds for is ∈ 1:n_neutral_species
		# for each neutral species, obtain affect of charge exchange collisions
		# with all of the ion species
		for isp ∈ 1:n_ion_species
			#cxfac = dt*charge_exchange_frequency
			for ivpa ∈ 1:vpa.n
				for iz ∈ 1:nz
					f_out[iz,ivpa,is+n_ion_species] += dt*charge_exchange_frequency*(
						f_in[iz,ivpa,isp]*moments.dens[iz,is+n_ion_species]
						- f_in[iz,ivpa,is+n_ion_species]*moments.dens[iz,isp])
				end
			end
		end
	end
end

end
