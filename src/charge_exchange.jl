module charge_exchange

export charge_exchange_collisions!

using ..looping

function charge_exchange_collisions!(f_out, fvec_in, moments, composition, vpa, z, r,
                                     charge_exchange_frequency, dt)

    if moments.evolve_density
        @s_r_z_loop_s is begin
            # apply CX collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    #cxfac = dt*charge_exchange_frequency[is,isp]
                    #cxfac = dt*charge_exchange_frequency
                    @s_r_z_loop_r ir begin
                        @s_r_z_loop_z iz begin
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,ir,is] +=
                                dt*charge_exchange_frequency*fvec_in.density[iz,ir,isp]*
                                (fvec_in.pdf[ivpa,iz,ir,isp] - fvec_in.pdf[ivpa,iz,ir,is])
                            end
                        end
                    end
                end
            end
            # apply CX collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @s_r_z_loop_r ir begin
                        @s_r_z_loop_z iz begin
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,ir,is] +=
                                dt*charge_exchange_frequency*fvec_in.density[iz,ir,isp]*
                                (fvec_in.pdf[ivpa,iz,ir,isp] - fvec_in.pdf[ivpa,iz,ir,is])
                            end
                        end
                    end
                end
            end
        end
    else
        @s_r_z_loop_s is begin
            # apply CX collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    #cxfac = dt*charge_exchange_frequency[is,isp]
                    #cxfac = dt*charge_exchange_frequency
                    @s_r_z_loop_r ir begin
                        @s_r_z_loop_z iz begin
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,ir,is] +=
                                    dt*charge_exchange_frequency*(
                                        fvec_in.pdf[ivpa,iz,ir,isp]*fvec_in.density[iz,ir,is]
                                        - fvec_in.pdf[ivpa,iz,ir,is]*fvec_in.density[iz,ir,isp])
                            end
                        end
                    end
                end
            end
            # apply CX collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @s_r_z_loop_r ir begin
                        @s_r_z_loop_z iz begin
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,ir,is] +=
                                    dt*charge_exchange_frequency*(
                                        fvec_in.pdf[ivpa,iz,ir,isp]*fvec_in.density[iz,ir,is]
                                        - fvec_in.pdf[ivpa,iz,ir,is]*fvec_in.density[iz,ir,isp])
                            end
                        end
                    end
                end
            end
        end
    end
end

end
