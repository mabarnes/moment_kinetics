module ionization

export ionization_collisions!

using ..looping

function ionization_collisions!(f_out, fvec_in, moments, n_ion_species,
        n_neutral_species, vpa, z, spectral, composition, collisions, nz, dt)

    if moments.evolve_density
		for is ∈ composition.species_local_range
			# apply ionization collisions to all ion species
			if is ∈ composition.ion_species_range
				# identify index 'is' as an ion species index
				isi = is
				# for each ion species, obtain the effect of ionization collisions
				# with all of the neutral species
				for isn ∈ composition.neutral_species_range
					for iz ∈ z.outer_loop_range
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
			if is ∈ composition.neutral_species_range
				# identify this 'is' index as a neutral species index
				isn = is
				# for each neutral species, obtain effect of ionization collisions
				# with all of the ion species
				for isi ∈ composition.ion_species_range
					for iz ∈ z.outer_loop_range
						for ivpa ∈ 1:vpa.n
							f_out[ivpa,iz,isn] -= dt*collisions.ionization*fvec_in.density[iz,isi]*fvec_in.pdf[ivpa,iz,isn]
						end
					end
				end
			end
		end
    elseif collisions.constant_ionization_rate
        # Oddly the test in test/harrisonthompson.jl matches the analytical
        # solution (which assumes width=0.0) better with width=0.5 than with,
        # e.g., width=0.15. Possibly narrower widths would require more vpa
        # resolution, which then causes crashes due to overshoots giving
        # negative f??
        width = 0.5
        @s_z_loop_s is begin
            if is ∈ composition.ion_species_range
                @s_z_loop_z iz begin
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] += dt*collisions.ionization/width*exp(-(vpa.grid[ivpa]/width)^2)
                    end
                end
            end
        end
    else
        @s_z_loop_s is begin
            # apply ionization collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    @s_z_loop_z iz begin
                        for ivpa ∈ 1:vpa.n
                            #NB: used quasineutrality to replace electron density with ion density
                            f_out[ivpa,iz,is] += dt*collisions.ionization*fvec_in.pdf[ivpa,iz,isp]*fvec_in.density[iz,is]
                        end
                    end
                end
            end
            # apply ionization collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of ionization collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    @s_z_loop_z iz begin
                        for ivpa ∈ 1:vpa.n
                            f_out[ivpa,iz,is] -= dt*collisions.ionization*fvec_in.pdf[ivpa,iz,is]*fvec_in.density[iz,isp]
                        end
                    end
                end
            end
        end
    end
end

end
