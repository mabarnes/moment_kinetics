module ionization

export ionization_collisions!

function ionization_collisions!(f_out, fvec_in, evolve_density, n_ion_species,
	n_neutral_species, vpa, collisions, nz, dt)

    if evolve_density
		# apply ionization collisions to all ion species
		for isi ∈ 1:n_ion_species
			# for each ion species, obtain effect of ionization collisions
			# with all of the neutral species
			for is ∈ 1:n_neutral_species
				isn = is + n_ion_species
				for iz ∈ 1:nz
					@. f_out[:,iz,isi] += dt*collisions.ionization*fvec_in.density[iz,isn]*fvec_in.pdf[:,iz,isn]
				end
			end
		end
		# apply ionization collisions to all neutral species
		for is ∈ 1:n_neutral_species
			isn = is + n_ion_species
			# for each neutral species, obtain effect of ionization collisions
			# with all of the ion species
			for isi ∈ 1:n_ion_species
				for iz ∈ 1:nz
					@. f_out[:,iz,isn] -= dt*collisions.ionization*fvec_in.density[iz,isi]*fvec_in.pdf[:,iz,isn]
				end
			end
		end
	elseif collisions.constant_ionization_rate
                # Oddly the test in test/harrisonthompson.jl matches the analitical
                # solution (which assumes width=0.0) better with width=0.5 than with,
                # e.g., width=0.15. Possibly narrower widths would require more vpa
                # resolution, which then causes crashes due to overshoots giving
                # negative f??
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
