module ionization

export ionization_collisions!

using ..communication: block_synchronize

function ionization_collisions!(f_out, fvec_in, moments, n_ion_species,
        n_neutral_species, vpa, z, composition, collisions, nz, dt)

    if moments.evolve_density
        error("Ionization collisions not currently supported for anything other than the standard drift kinetic equation: Aborting.")
    elseif collisions.constant_ionization_rate
        # Oddly the test in test/harrisonthompson.jl matches the analitical
        # solution (which assumes width=0.0) better with width=0.5 than with,
        # e.g., width=0.15. Possibly narrower widths would require more vpa
        # resolution, which then causes crashes due to overshoots giving
        # negative f??
        width = 0.5
        for is ∈ composition.ion_species_local_range
            for ivpa ∈ vpa.outer_loop_range_ions
                @. f_out[ivpa,:,is] += dt*collisions.ionization/width*exp(-(vpa.grid[ivpa]/width)^2)
            end
        end
    else
        # apply ionization collisions to all ion species
        for is ∈ composition.ion_species_local_range
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            for isn ∈ 1:n_neutral_species
                isp = isn + n_ion_species
                for iz ∈ z.outer_loop_range_ions
                    for ivpa ∈ 1:vpa.n
                        #NB: used quasineutrality to replace electron density with ion density
                        f_out[ivpa,iz,is] += dt*collisions.ionization*fvec_in.pdf[ivpa,iz,isp]*fvec_in.density[iz,is]
                    end
                end
            end
        end
        # apply ionization collisions to all neutral species
        for is ∈ composition.neutral_species_local_range
            # for each neutral species, obtain affect of ionization collisions
            # with all of the ion species
            for isp ∈ 1:n_ion_species
                for iz ∈ z.outer_loop_range_neutrals
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] -= dt*collisions.ionization*fvec_in.pdf[ivpa,iz,is]*fvec_in.density[iz,isp]
                    end
                end
            end
        end
        !moments.evolve_upar && block_synchronize()
    end
end

end
