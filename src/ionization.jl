"""
"""
module ionization

export ionization_collisions!

using ..interpolation: interpolate_to_grid_vpa!
using ..looping

"""
"""
function ionization_collisions!(f_out, fvec_in, moments, n_ion_species,
        n_neutral_species, vpa, z, r, vpa_spectral, composition, collisions, nz, dt)

    begin_s_r_z_region()

    if collisions.constant_ionization_rate
        # Oddly the test in test/harrisonthompson.jl matches the analytical
        # solution (which assumes width=0.0) better with width=0.5 than with,
        # e.g., width=0.15. Possibly narrower widths would require more vpa
        # resolution, which then causes crashes due to overshoots giving
        # negative f??
        width = 0.5
        @loop_s is begin
            if is ∈ composition.ion_species_range
                @loop_r_z ir iz begin
                    if moments.evolve_ppar && moments.evolve_upar
                        @. vpa.scratch = vpa.grid / moments.vth[iz] + fvec_in.upar[iz]
                        prefactor = moments.vth[iz] / fvec_in.dens[iz]
                    elseif moments.evolve_ppar
                        @. vpa.scratch = vpa.grid / moments.vth[iz]
                        prefactor = moments.vth[iz] / fvec_in.dens[iz]
                    elseif moments.evolve_upar
                        @. vpa.scratch = vpa.grid + fvec_in.upar[iz]
                        prefactor = 1.0 / fvec_in.dens[iz]
                    elseif moments.evolve_density
                        @. vpa.scratch = vpa.grid
                        prefactor = 1.0 / fvec_in.dens[iz]
                    else
                        @. vpa.scratch = vpa.grid
                        prefactor = 1.0
                    end
                    #if moments.evolve_density
                    #    @. fvec_out.density += dt*collisions.ionization
                    #end
                    #if moments.evolve_ppar
                    #    @. fvec_out.ppar += dt*collisions.ionization
                    #end
                    @loop_vpa ivpa begin
                        f_out[ivpa,iz,ir,is] += dt*collisions.ionization/width*prefactor*exp(-(vpa.scratch[ivpa]/width)^2)
                    end
                end
            end
        end
    elseif moments.evolve_density
        @loop_s is begin
            # apply ionization collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain effect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    @views ionization_collisions_single_species!(f_out[:,:,:,is], fvec_in,
                        moments, vpa, vpa_spectral, collisions.ionization, dt, is, isp)
                end
            end
            # when working with the normalised distribution (pdf_unnorm / density),
            # the ionisation collisions drop out of the neutral kinetic equation
        end
    else
        @loop_s is begin
            # apply ionization collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    @loop_r_z_vpa ir iz ivpa begin
                        #NB: used quasineutrality to replace electron density with ion density
                        f_out[ivpa,iz,ir,is] += dt*collisions.ionization*fvec_in.pdf[ivpa,iz,ir,isp]*fvec_in.density[iz,ir,is]
                    end
                end
            end
            # apply ionization collisions to all neutral species
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of ionization collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    @loop_r_z_vpa ir iz ivpa begin
                        f_out[ivpa,iz,ir,is] -= dt*collisions.ionization*fvec_in.pdf[ivpa,iz,ir,is]*fvec_in.density[iz,ir,isp]
                    end
                end
            end
        end
    end
end

function ionization_collisions_single_species!(f_out, fvec_in, moments, vpa, vpa_spectral, ionization, dt, is, isp)
    @loop_r_z ir iz begin
        if moments.evolve_ppar
            # will need the ratio of thermal speeds both to interpolate between vpa grids
            # for different species and to account for different normalizations of each species' pdf
            vth_ratio = moments.vth[iz,ir,is]/moments.vth[iz,ir,isp]
        else
            vth_ratio = 1.0
        end
        # if the parallel flow and/or the parallel pressure are separately evolved,
        # then the parallel velocity coordinate is re-defined so that the jth
        # vpa grid point for different species corresponds to different physical
        # values of dz/dt; as charge exchange and ionization collisions require
        # the evaluation of the pdf for species s' to obtain the update for species s,
        # will thus have to interpolate between the different vpa grids
        if moments.evolve_ppar || moments.evolve_upar
            if !moments.evolve_upar
                # if evolve_ppar = true and evolve_upar = false, vpa coordinate is
                # vpahat_s = vpa/vth_s;
                # we have f_{s'}(vpahat_{s'}) = f_{s'}(vpahat_s * vth_s / vth_{s'});
                # to get f_{s'}(vpahat_s), need to obtain vpahat_s grid locations
                # in terms of the vpahat_{s'} coordinate:
                # (vpahat_s)_j = (vpahat_{s'})_j * vth_{s'} / vth_{s}
                @. vpa.scratch = vpa.grid / vth_ratio
            elseif !moments.evolve_ppar
                # if evolve_ppar = false and evolve_upar = true, vpa coordinate is
                # wpa_s = vpa-upar_s;
                # we have f_{s'}(wpa_{s'}) = f_{s'}((wpa_s + upar_s - upar_{s'};
                # to get f_{s'}(wpa_s), need to obtain wpa_s grid locations
                # in terms of the wpa_{s'} coordinate:
                # (wpa_s)_j = (wpa_{s'})_j + upar_{s'} - upar_{s}
                @. vpa.scratch = vpa.grid + fvec_in.upar[iz,ir,is] - fvec_in.upar[iz,ir,isp]
            else
                # if evolve_ppar = true and evolve_upar = true, vpa coordinate is
                # wpahat_s = (vpa-upar_s)/vth_s;
                # we have f_{s'}(wpahat_{s'}) = f_{s'}((wpahat_s * vth_s + upar_s - upar_{s'}) / vth_{s'});
                # to get f_{s'}(wpahat_s), need to obtain wpahat_s grid locations
                # in terms of the wpahat_{s'} coordinate:
                # (wpahat_{s'})_j = ((wpahat_{s})_j * vth_{s} + upar_{s} - upar_{s'}) / vth_{s'}
                @. vpa.scratch = (vpa.grid * moments.vth[iz,ir,is] + fvec_in.upar[iz,ir,is] - fvec_in.upar[iz,ir,isp]) / moments.vth[iz,ir,isp]
            end
            # interpolate to the new grid (passed in as vpa.scratch)
            # and return interpolated values in vpa.scratch2
            @views interpolate_to_grid_vpa!(vpa.scratch2, vpa.scratch, fvec_in.pdf[:,iz,ir,isp], vpa, vpa_spectral)
        else
            # no need to interpolate if neither upar or ppar evolved separately from pdf
            vpa.scratch2 .= fvec_in.pdf[:,iz,ir,isp]
        end
        @loop_vpa ivpa begin
            f_out[ivpa,iz,ir] +=
                dt*ionization*fvec_in.density[iz,ir,isp]*
                (vpa.scratch2[ivpa]*vth_ratio - fvec_in.pdf[ivpa,iz,ir,is])
        end
    end
end

end
