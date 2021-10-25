module ionization

export ionization_collisions!

using ..interpolation: interpolate_to_grid_vpa

function ionization_collisions!(f_out, fvec_in, moments, n_ion_species,
								n_neutral_species, vpa, spectral, collisions, nz, dt)

    if moments.evolve_density
		# apply ionization collisions to all ion species
		for isi ∈ 1:n_ion_species
			# for each ion species, obtain effect of ionization collisions
			# with all of the neutral species
			for is ∈ 1:n_neutral_species
				isn = is + n_ion_species
				for iz ∈ 1:nz
					if moments.evolve_upar
						# wpa = vpa - upa_s if upar evolved but no ppar
	                    # if ppar also evolved, wpa = (vpa - upa_s)/vths
						if moments.evolve_ppar
							wpa_shift_norm = moments.vth[iz,isn]
						else
							wpa_shift_norm = 1.0
						end
						# calculate the shift in the wpa grid to the desired (ion) wpa locations
						wpa_shift = (fvec_in.upar[iz,isi] - fvec_in.upar[iz,isn])/wpa_shift_norm
						# construct the wpa grid on which to interpolate the neutral pdf
                        @. vpa.scratch = vpa.grid + wpa_shift
						# interpolate to the new grid (passed in as vpa.scratch)
                        # and return interpolated values in vpa.scratch
                        vpa.scratch .= interpolate_to_grid_vpa(vpa.scratch, view(fvec_in.pdf,:,iz,isn), vpa, spectral)
					else
						@. vpa.scratch = fvec_in.pdf[:,iz,isn]
					end
					@. f_out[:,iz,isi] += dt*collisions.ionization*fvec_in.density[iz,isn]*vpa.scratch
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
