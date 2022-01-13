module charge_exchange

export charge_exchange_collisions!

using ..looping
using ..interpolation: interpolate_to_grid_vpa

function charge_exchange_collisions!(f_out, fvec_in, moments, composition, vpa, z,
                                     spectral, charge_exchange_frequency, dt)

    if moments.evolve_density
        # if evolving upar (and possibly also ppar), then
        # the velocity coordinate is either vpa-upars or (vpa-upars)/vths.
        # this means that the velocity grid for different species
        # corrseponds to different physical vpa locations.
        # in this case, must interpolate pdf for neutrals (when solving for ions)
        # and ions (when solving for neutrals) onto the opposite species' wpa grid
        if moments.evolve_upar
            @s_z_loop_s is begin
                # we have the neutral pdf on a set of vpa grid points given by
                # vpa_n = wpa*vthn + upar_n
                # we want to interpolate the this pdf onto a set of vpa grid points
                # determined by the ions:
                # vpa_i = wpa*vthi + upar_i
                # the difference between vpa_n and vpa_i for the jth grid point is
                # (vpa_n - vpa_i)_j = wpa_j*(vthn-vthi) + upar_n-upar_i
                # NB: we assume Tn=Ti and mn=mi so that vthn-vthi=0;
                # i.e., (vpa_n - vpa_i)_j = upar_n - upar_i
                # the modified wpa locations at which we need the neutral pdf are thus
                # wpamod_j = (vpai_j - upar_n)/vthn = (vpan_j - upar_n)/vthn + (upar_i - upar_n)/vthn
                # = wpa_j + (upar_i - upar_n)/vthn
                # obtain the ion wpa values corresponding to the neutral wpa grid points
                if is ∈ composition.neutral_species_range
                    # identify the 'is' index as a neutral species index
                    isn = is
                    @s_z_loop_z iz begin
                        # wpa = vpa - upa_s if upar evolved but no ppar
                        # if ppar also evolved, wpa = (vpa - upa_s)/vths
                        if moments.evolve_ppar
                            wpa_shift_norm = moments.vth[iz,isn]
                        else
                            wpa_shift_norm = 1.0
                        end
                        for isi ∈ composition.ion_species_range
                            # calculate the shift in the wpa grid to the desired (ion) wpa locations
                            wpa_shift = (fvec_in.upar[iz,isi] - fvec_in.upar[iz,isn])/wpa_shift_norm
                            # construct the wpa grid on which to interpolate
                            for ivpa ∈ 1:vpa.n
                                vpa.scratch[ivpa] = vpa.grid[ivpa] + wpa_shift
                            end
                            # interpolate to the new grid (passed in as vpa.scratch)
                            # and return interpolated values in vpa.scratch
                            vpa.scratch .= interpolate_to_grid_vpa(vpa.scratch, view(fvec_in.pdf,:,iz,isn), vpa, spectral)
                            # add the charge exchange contribution to df/dt
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,isi] += dt*charge_exchange_frequency*fvec_in.density[iz,isn] *
                                    (vpa.scratch[ivpa] - fvec_in.pdf[ivpa,iz,isi])
                            end
                        end
                    end
                # repeat the above interpolation process to get the ion pdf on the neutral vpa grid
                elseif is ∈ composition.ion_species_range
                    # identify the 'is' index as corresponding to an ion species
                    isi = is
                    # wpa = vpa - upa_s if upar evolved but no ppar
                    # if ppar also evolved, wpa = (vpa - upa_s)/vths
                    if moments.evolve_ppar
                        wpa_shift_norm = moments.vth[iz,isi]
                    else
                        wpa_shift_norm = 1.0
                    end
                    for isn ∈ composition.neutral_species_range
                        # calculate the shift in the wpa grid to the desired (neutral) wpa locations
                        wpa_shift = (fvec_in.upar[iz,isn] - fvec_in.upar[iz,isi])/wpa_shift_norm
                        # construct the wpa grid on which to interpolate
                        for ivpa ∈ 1:vpa.n
                            vpa.scratch[ivpa] = vpa.grid[ivpa] + wpa_shift
                        end
                        # interpolate to the new grid (passed in as vpa.scratch)
                        # and return interpolated values in vpa.scratch
                        vpa.scratch .= interpolate_to_grid_vpa(vpa.scratch, view(fvec_in.pdf,:,iz,isi), vpa, spectral)
                        # add the charge exchange contribution to df/dt
                        for ivpa ∈ 1:vpa.n
                            f_out[ivpa,iz,isn] += dt*charge_exchange_frequency*fvec_in.density[iz,isi] *
                                (vpa.scratch[ivpa]- fvec_in.pdf[ivpa,iz,isn])
                        end
                    end
                end
            end
        else
            # apply CX collisions to all species
            @s_z_loop_s is begin
                # apply CX collisions to ion species
                if is ∈ composition.ion_species_range
                    # use index 'isi' to identify ion species
                    isi = is
                    # for each ion species, obtain affect of charge exchange collisions
                    # with all of the neutral species
                    for isn ∈ composition.neutral_species_range
                        @s_z_loop_z iz begin
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,isi] +=
                                    dt*charge_exchange_frequency*fvec_in.density[iz,isn]*(
                                    fvec_in.pdf[ivpa,iz,isn] - fvec_in.pdf[ivpa,iz,isi])
                            end
                        end
                    end
                end
                # apply CX collisions to neutral species
                if is ∈ composition.neutral_species_range
                    # use index 'isn' to identify neutral species
                    isn = is
                    # for each neutral species, obtain affect of charge exchange collisions
                    # with all of the ion species
                    for isi ∈ composition.ion_species_range
                        @s_z_loop_z iz begin
                            for ivpa ∈ 1:vpa.n
                                f_out[ivpa,iz,isn] +=
                                    dt*charge_exchange_frequency*fvec_in.density[iz,isi]*(
                                    fvec_in.pdf[ivpa,iz,isi] - fvec_in.pdf[ivpa,iz,isn])
                            end
                        end
                    end
                end
            end
        end
    else
        @s_z_loop_s is begin
            # apply CX collisions to all ion species
            if is ∈ composition.ion_species_range
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for isp ∈ composition.neutral_species_range
                    #cxfac = dt*charge_exchange_frequency[is,isp]
                    #cxfac = dt*charge_exchange_frequency
                    @s_z_loop_z iz begin
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
            if is ∈ composition.neutral_species_range
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isp ∈ composition.ion_species_range
                    #cxfac = dt*charge_exchange_frequency
                    @s_z_loop_z iz begin
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

end
