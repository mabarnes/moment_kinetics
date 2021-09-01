module ionization

export ionization_collisions!

function ionization_collisions!(f_out, fvec_in, evolve_density, n_ion_species,
	n_neutral_species, vpa, collisions, nz, dt)

    if evolve_density
        error("Ionization collisions not currently supported for anything other than the standard drift kinetic equation: Aborting.")
	elseif collisions.constant_ionization_rate
		width = 0.5
		for ivpa ∈ 1:vpa.n
			@. f_out[ivpa,:,1] += dt*collisions.ionization/width*exp(-(vpa.grid[ivpa]/width)^2)
		end
	else
        # apply ionization collisions to all ion species
		for is ∈ 1:n_ion_species
			# for each ion species, obtain affect of charge exchange collisions
			# with all of the neutral species
			for isn ∈ 1:n_neutral_species
				isp = isn + n_ion_species
				for iz ∈ 1:nz
					for ivpa ∈ 1:vpa.n
						#NB: used quasineutrality to replace electron density with ion density
						f_out[ivpa,iz,is] += dt*collisions.ionization*fvec_in.pdf[ivpa,iz,isp]*fvec_in.density[iz,is]
					end
				end
			end
		end
		# apply ionization collisions to all neutral species
		for isn ∈ 1:n_neutral_species
			is = isn + n_ion_species
			# for each neutral species, obtain affect of ionization collisions
			# with all of the ion species
			for isp ∈ 1:n_ion_species
				for iz ∈ 1:nz
					for ivpa ∈ 1:vpa.n
						f_out[ivpa,iz,is] -= dt*collisions.ionization*fvec_in.pdf[ivpa,iz,is]*fvec_in.density[iz,isp]
					end
				end
			end
		end
    end
end

end
