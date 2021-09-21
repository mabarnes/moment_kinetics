module charge_exchange

export charge_exchange_collisions!

function charge_exchange_collisions!(f_out, fvec_in, moments, composition, vpa, z,
                                     charge_exchange_frequency, dt)

    if moments.evolve_density
        # apply CX collisions to all ion species
        @inbounds for is ∈ composition.ion_species_local_range
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            for isp ∈ composition.n_ion_species+1:composition.n_species
                #cxfac = dt*charge_exchange_frequency[is,isp]
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ z.outer_loop_range_ions
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] +=
                           dt*charge_exchange_frequency*fvec_in.density[iz,isp]*(
                               fvec_in.pdf[ivpa,iz,isp] - fvec_in.pdf[ivpa,iz,is])
                    end
                end
            end
        end
        # apply CX collisions to all neutral species
        @inbounds for is ∈ composition.neutral_species_local_range
            # for each neutral species, obtain affect of charge exchange collisions
            # with all of the ion species
            for isp ∈ 1:composition.n_ion_species
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ z.outer_loop_range_neutrals
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] +=
                            dt*charge_exchange_frequency*fvec_in.density[iz,isp]*(
                                fvec_in.pdf[ivpa,iz,isp] - fvec_in.pdf[ivpa,iz,is])
                    end
                end
            end
        end
    else
        # apply CX collisions to all ion species
        @inbounds for is ∈ composition.ion_species_local_range
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            for isp ∈ composition.n_ion_species+1:composition.n_species
                #cxfac = dt*charge_exchange_frequency[is,isp]
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ z.outer_loop_range_ions
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,iz,isp]*fvec_in.density[iz,is]
                                - fvec_in.pdf[ivpa,iz,is]*fvec_in.density[iz,isp])
                    end
                end
            end
        end
        # apply CX collisions to all neutral species
        @inbounds for is ∈ composition.neutral_species_local_range
            # for each neutral species, obtain affect of charge exchange collisions
            # with all of the ion species
            for isp ∈ 1:composition.n_ion_species
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ z.outer_loop_range_neutrals
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,iz,isp]*fvec_in.density[iz,is]
                                - fvec_in.pdf[ivpa,iz,is]*fvec_in.density[iz,isp])
                    end
                end
            end
        end
    end
end

end
