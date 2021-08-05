module ionization

export ionization_collisions!

function ionization_collisions!(f_out, fvec_in, evolve_density, n_ion_species,
	n_neutral_species, nvpa, ionization_frequency, nz, dt)

    if evolve_density
        println("Ionization collisions not currently supported for anything other than the standard drift kinetic equation: Aborting.")
        exit()
    else
        # apply ionization collisions to all ion species
		for is ∈ 1:n_ion_species
			# for each ion species, obtain affect of charge exchange collisions
			# with all of the neutral species
			for isn ∈ 1:n_neutral_species
				isp = isn + n_ion_species
				for ivpa ∈ 1:nvpa
					for iz ∈ 1:nz
						#NB: used quasineutrality to replace electron density with ion density
						f_out[iz,ivpa,is] += dt*ionization_frequency*fvec_in.pdf[iz,ivpa,isp]*fvec_in.density[iz,is]
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
				for ivpa ∈ 1:nvpa
					for iz ∈ 1:nz
						f_out[iz,ivpa,is] -= dt*ionization_frequency*fvec_in.pdf[iz,ivpa,is]*fvec_in.density[iz,isp]
					end
				end
			end
		end
    end
end

end
