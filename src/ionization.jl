module ionization

export ionization_collisions!

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
        for is ∈ composition.species_local_range
            if is ∈ composition.ion_species_range
                for iz ∈ z.outer_loop_range, ivpa ∈ 1:vpa.n
                    f_out[ivpa,iz,is] += dt*collisions.ionization/width*exp(-(vpa.grid[ivpa]/width)^2)
                end
            end
        end
    else
        for is ∈ composition.species_local_range
            # apply ionization collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    for iz ∈ z.outer_loop_range
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
                    for iz ∈ z.outer_loop_range
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
