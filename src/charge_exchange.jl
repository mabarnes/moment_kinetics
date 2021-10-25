module charge_exchange

export charge_exchange_collisions!

using ..interpolation: interpolate_to_grid_vpa

function charge_exchange_collisions!(f_out, fvec_in, moments, n_ion_species,
        n_neutral_species, vpa, spectral, charge_exchange_frequency, nz, dt)

    if moments.evolve_density
        # if evolving upar (and possibly also ppar), then
        # the velocity coordinate is either vpa-upars or (vpa-upars)/vths.
        # this means that the velocity grid for different species
        # corrseponds to different physical vpa locations.
        # in this case, must interpolate pdf for neutrals (when solving for ions)
        # and ions (when solving for neutrals) onto the opposite species' wpa grid
        if moments.evolve_upar
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
            for is ∈ 1:n_neutral_species
                isn = is + n_ion_species
                for iz ∈ 1:nz
                    # wpa = vpa - upa_s if upar evolved but no ppar
                    # if ppar also evolved, wpa = (vpa - upa_s)/vths
                    if moments.evolve_ppar
                        wpa_shift_norm = moments.vth[iz,isn]
                    else
                        wpa_shift_norm = 1.0
                    end
                    for isi ∈ 1:n_ion_species
                        # calculate the shift in the wpa grid to the desired (ion) wpa locations
                        wpa_shift = (fvec_in.upar[iz,isi] - fvec_in.upar[iz,isn])/wpa_shift_norm
                        # construct the wpa grid on which to interpolate
                        # NB: TMP FOR TESTING !!
                        #@. vpa.scratch = vpa.grid - wpa_shift
                        @. vpa.scratch = vpa.grid + wpa_shift
                        # for ivpa ∈ 1:vpa.n
                        #     println("vpa_unshifted: ", vpa.grid[ivpa], " vpa_shifted: ", vpa.scratch[ivpa])
                        # end
                        # interpolate to the new grid (passed in as vpa.scratch)
                        # and return interpolated values in vpa.scratch
                        vpa.scratch .= interpolate_to_grid_vpa(vpa.scratch, view(fvec_in.pdf,:,iz,isn), vpa, spectral)
                        # for ivpa ∈ 1:vpa.n
                        #     println("uninterp: ", fvec_in.pdf[ivpa,iz,isn], "  interp: ", vpa.scratch[ivpa])
                        # end
                        # add the charge exchange contribution to df/dt
                        for ivpa ∈ 1:vpa.n
                            #interp_pdf = interp_neutral_pdf_to_ion_grid(vpa.grid[ivpa]+wpa_shift)
                            f_out[ivpa,iz,isi] += dt*charge_exchange_frequency*fvec_in.density[iz,isn] *
                                (vpa.scratch[ivpa] - fvec_in.pdf[ivpa,iz,isi])
                        end
                    end
                end
            end
            # repeat the above interpolation process to get the ion pdf on the neutral vpa grid
            for isi ∈ 1:n_ion_species
                for iz ∈ 1:nz
                    # vpa.scratch .= fvec_in.pdf[:,iz,isi]
                    # # create an interpolation object (linear interpolation for simplicity but can be improved)
                    # #interp_ion_pdf_to_neutral_grid = LinearInterpolation(vpa.grid, view(fvec_in.pdf[:,iz,isi]))
                    # interp_ion_pdf_to_neutral_grid = LinearInterpolation(vpa.grid, vpa.scratch)
                    # wpa = vpa - upa_s if upar evolved but no ppar
                    # if ppar also evolved, wpa = (vpa - upa_s)/vths
                    if moments.evolve_ppar
                        wpa_shift_norm = moments.vth[iz,isi]
                    else
                        wpa_shift_norm = 1.0
                    end
                    for is ∈ 1:n_neutral_species
                        isn = is + n_ion_species
                        # calculate the shift in the wpa grid to the desired (neutral) wpa locations
                        wpa_shift = (fvec_in.upar[iz,isn] - fvec_in.upar[iz,isi])/wpa_shift_norm
                        # construct the wpa grid on which to interpolate
                        @. vpa.scratch = vpa.grid + wpa_shift
                        # interpolate to the new grid (passed in as vpa.scratch)
                        # and return interpolated values in vpa.scratch
                        vpa.scratch .= interpolate_to_grid_vpa(vpa.scratch, view(fvec_in.pdf,:,iz,isi), vpa, spectral)
                        # add the charge exchange contribution to df/dt
                        for ivpa ∈ 1:vpa.n
                            #interp_pdf = interp_ion_pdf_to_neutral_grid(vpa.grid[ivpa]+wpa_shift)
                            f_out[ivpa,iz,isn] += dt*charge_exchange_frequency*fvec_in.density[iz,isi] *
                                (vpa.scratch[ivpa]- fvec_in.pdf[ivpa,iz,isn])
                        end
                    end
                end
            end
        else
            # apply CX collisions to all ion species
            @inbounds for isi ∈ 1:n_ion_species
                # for each ion species, obtain affect of charge exchange collisions
                # with all of the neutral species
                for is ∈ 1:n_neutral_species
                    isn = is + n_ion_species
                    #cxfac = dt*charge_exchange_frequency[is,isp]
                    #cxfac = dt*charge_exchange_frequency
                    for iz ∈ 1:nz
                        for ivpa ∈ 1:vpa.n
                            f_out[ivpa,iz,isi] +=
                               dt*charge_exchange_frequency*fvec_in.density[iz,isn]*(
                                   fvec_in.pdf[ivpa,iz,isn] - fvec_in.pdf[ivpa,iz,isi])
                        end
                    end
                end
            end
            # apply CX collisions to all neutral species
            @inbounds for is ∈ 1:n_neutral_species
                isn = is + n_ion_species
                # for each neutral species, obtain affect of charge exchange collisions
                # with all of the ion species
                for isi ∈ 1:n_ion_species
                    #cxfac = dt*charge_exchange_frequency
                    for iz ∈ 1:nz
                        for ivpa ∈ 1:vpa.n
                            f_out[ivpa,iz,isn] +=
                                dt*charge_exchange_frequency*fvec_in.density[iz,isi]*(
                                    fvec_in.pdf[ivpa,iz,isi] - fvec_in.pdf[ivpa,iz,isn])
                        end
                    end
                end
            end
        end
    else
        # apply CX collisions to all ion species
        @inbounds for isi ∈ 1:n_ion_species
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            for is ∈ 1:n_neutral_species
                isn = is + n_ion_species
                #cxfac = dt*charge_exchange_frequency[is,isp]
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ 1:nz
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,isi] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,iz,isn]*fvec_in.density[iz,isi]
                                - fvec_in.pdf[ivpa,iz,isi]*fvec_in.density[iz,isn])
                    end
                end
            end
        end
        # apply CX collisions to all neutral species
        @inbounds for is ∈ 1:n_neutral_species
            isn = is + n_ion_species
            # for each neutral species, obtain affect of charge exchange collisions
            # with all of the ion species
            for isi ∈ 1:n_ion_species
                #cxfac = dt*charge_exchange_frequency
                for iz ∈ 1:nz
                    for ivpa ∈ 1:vpa.n
                        f_out[ivpa,iz,isn] +=
                            dt*charge_exchange_frequency*(
                                fvec_in.pdf[ivpa,iz,isi]*fvec_in.density[iz,isn]
                                - fvec_in.pdf[ivpa,iz,isn]*fvec_in.density[iz,isi])
                    end
                end
            end
        end
    end
end

end
