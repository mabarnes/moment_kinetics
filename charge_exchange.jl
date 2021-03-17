module charge_exchange

export charge_exchange_collisions!

using velocity_moments: update_density!

function charge_exchange_collisions!(f_out, fvec_in, moments, n_ion_species,
	n_neutral_species, vpa, charge_exchange_frequency, nz, dt)
	# make sure all densities needed for the charge exchange collisions are up-to-date
	# if density evolved via continuity, then use value passed in;
	# else, must update the density via velocity integration of the pdf
	if moments.evolve_density
		dens = fvec_in.density
	else
		for is ∈ 1:n_ion_species+n_neutral_species
			if moments.dens_updated[is] == false
				@views update_density!(moments.dens[:,is], vpa.scratch, fvec_in.pdf[:,:,is], vpa, nz)
				moments.dens_updated[is] = true
			end
		end
		dens = moments.dens
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
						fvec_in.pdf[iz,ivpa,isp+n_ion_species]*dens[iz,is]
						#fvec_in.pdf[iz,ivpa,isp+n_ion_species]*fvec_in.density[iz,is]
						- fvec_in.pdf[iz,ivpa,is]*dens[iz,isp+n_ion_species])
						#- fvec_in.pdf[iz,ivpa,is]*fvec_in.density[iz,isp+n_ion_species])
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
						fvec_in.pdf[iz,ivpa,isp]*dens[iz,is+n_ion_species]
						#fvec_in.pdf[iz,ivpa,isp]*fvec_in.density[iz,is+n_ion_species]
						- fvec_in.pdf[iz,ivpa,is+n_ion_species]*dens[iz,isp])
						#- fvec_in.pdf[iz,ivpa,is+n_ion_species]*fvec_in.density[iz,isp])
				end
			end
		end
	end
end

end
