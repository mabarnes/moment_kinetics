"""
"""
module charge_exchange

export charge_exchange_collisions!

using ..looping

"""
"""
function charge_exchange_collisions!(f_out, fvec_in, moments, composition, vpa, vperp, z, r,
                                     charge_exchange_frequency, dt)

    if moments.evolve_density
        @loop_s is begin
            # apply CX collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    #cxfac = dt*charge_exchange_frequency[is,isp]
                    #cxfac = dt*charge_exchange_frequency
                    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                        f_out[ivpa,ivperp,iz,ir,is] +=
                        dt*charge_exchange_frequency*fvec_in.density[iz,ir,isp]*
                        (fvec_in.pdf[ivpa,ivperp,iz,ir,isp] - fvec_in.pdf[ivpa,ivperp,iz,ir,is])
                    end
                end
            end
            # apply CX collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                        f_out[ivpa,ivperp,iz,ir,is] +=
                        dt*charge_exchange_frequency*fvec_in.density[iz,ir,isp]*
                        (fvec_in.pdf[ivpa,ivperp,iz,ir,isp] - fvec_in.pdf[ivpa,ivperp,iz,ir,is])
                    end
                end
            end
        end
    else
        @loop_s is begin
            # apply CX collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    #cxfac = dt*charge_exchange_frequency[is,isp]
                    #cxfac = dt*charge_exchange_frequency
                    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                        f_out[ivpa,ivperp,iz,ir,is] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,ivperp,iz,ir,isp]*fvec_in.density[iz,ir,is]
                                - fvec_in.pdf[ivpa,ivperp,iz,ir,is]*fvec_in.density[iz,ir,isp])
                    end
                end
            end
            # apply CX collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                        f_out[ivpa,ivperp,iz,ir,is] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,ivperp,iz,ir,isp]*fvec_in.density[iz,ir,is]
                                - fvec_in.pdf[ivpa,ivperp,iz,ir,is]*fvec_in.density[iz,ir,isp])
                    end
                end
            end
        end
    end
end

end
