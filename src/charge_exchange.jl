"""
"""
module charge_exchange

export charge_exchange_collisions!

using ..looping

"""
"""
function charge_exchange_collisions!(f_out, fvec_in, moments, composition, vpa, z, r,
                                     charge_exchange_frequency, dt)

    if moments.evolve_density
        @loop_s is begin
            # apply CX collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    @views charge_exchange_collisions_single_species!(f_out[:,:,:,is], fvec_in,
                        moments, charge_exchange_frequency, dt, is, isp)
                end
            end
            # apply CX collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @views charge_exchange_collisions_single_species!(f_out[:,:,:,is], fvec_in,
                        moments, charge_exchange_frequency, dt, is, isp)
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
                    @loop_r_z_vpa ir iz ivpa begin
                        f_out[ivpa,iz,ir,is] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,iz,ir,isp]*fvec_in.density[iz,ir,is]
                                - fvec_in.pdf[ivpa,iz,ir,is]*fvec_in.density[iz,ir,isp])
                    end
                end
            end
            # apply CX collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @loop_r_z_vpa ir iz ivpa begin
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

"""
"""
function charge_exchange_collisions_single_species!(f_out, fvec_in, moments, charge_exchange_frequency, dt, is, isp)
    @loop_r_z_vpa ir iz ivpa begin
        # if the parallel flow and/or the parallel pressure are separately evolved,
        # then the parallel velocity coordinate is re-defined so that the jth
        # vpa grid point for different species corresponds to different physical
        # values of dz/dt; as charge exchange and ionization collisions require
        # the evaluation of the pdf for species s' to obtain the update for species s,
        # will thus have to interpolate between the different vpa grids
        if moments.evolve_ppar
            vth_ratio = moments.vth[iz,ir,is]/moments.vth[iz,ir,isp]
        else
            vth_ratio = 1.0
        end
        f_out[ivpa,iz,ir] += dt * charge_exchange_frequency * fvec_in.density[iz,ir,isp] *
            (fvec_in.pdf[ivpa,iz,ir,isp] * vth_ratio - fvec_in.pdf[ivpa,iz,ir,is])
    end
end

end
