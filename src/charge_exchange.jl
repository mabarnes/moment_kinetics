module charge_exchange

export charge_exchange_collisions!

using ..optimization

function charge_exchange_collisions!(f_out, fvec_in, moments, n_ion_species,
        n_species, vpa_vec, charge_exchange_frequency, nz, dt)

    if moments.evolve_density
        # apply CX collisions to all ion species
        @inbounds for is ∈ 1:n_ion_species
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            for isp ∈ n_ion_species+1:n_species
                vpa = vpa_vec[threadid()]
                #cxfac = dt*charge_exchange_frequency[is,isp]
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ 1:nz
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,is] +=
                           dt*charge_exchange_frequency*fvec_in.density[iz,isp]*(
                               fvec_in.pdf[ivpa,iz,isp] - fvec_in.pdf[ivpa,iz,is])
                    end
                end
            end
        end
        # apply CX collisions to all neutral species
        @inbounds for is ∈ n_ion_species+1:n_species
            # for each neutral species, obtain affect of charge exchange collisions
            # with all of the ion species
            for isp ∈ 1:n_ion_species
                #cxfac = dt*charge_exchange_frequency
                vpa = vpa_vec[threadid()]
                for iz ∈ 1:nz
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
        @inbounds for is ∈ 1:n_ion_species
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            for isp ∈ n_ion_species+1:n_species
                #cxfac = dt*charge_exchange_frequency[is,isp]
                #cxfac = dt*charge_exchange_frequency
                vpa = vpa_vec[threadid()]
                for iz ∈ 1:nz
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
        @inbounds for is ∈ n_ion_species+1:n_species
            # for each neutral species, obtain affect of charge exchange collisions
            # with all of the ion species
            for isp ∈ 1:n_ion_species
                #cxfac = dt*charge_exchange_frequency
                vpa = vpa_vec[threadid()]
                for iz ∈ 1:nz
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
