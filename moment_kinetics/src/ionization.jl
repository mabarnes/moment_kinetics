"""
"""
module ionization

export ion_ionization_collisions_1V!
export neutral_ionization_collisions_1V!
export ion_ionization_collisions_3V!
export neutral_ionization_collisions_3V!

using ..interpolation: interpolate_to_grid_vpa!
using ..looping
using ..timer_utils

@timeit global_timer ion_ionization_collisions_1V!(
                         f_out, fvec_in, vz, vpa, vperp, z, r, vz_spectral, moments,
                         composition, collisions, dt) = begin
    # This routine assumes a 1D model with:
    # nvz = nvpa and identical vz and vpa grids 
    # nvperp = nvr = nveta = 1
    # constant charge_exchange_frequency independent of species
    @boundscheck vpa.n == size(f_out,1) || throw(BoundsError(f_out))
    @boundscheck 1 == size(f_out,2) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out,3) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out,4) || throw(BoundsError(f_out))
    @boundscheck composition.n_ion_species == size(f_out,5) || throw(BoundsError(f_out))
    
    @begin_r_z_region()

    if moments.evolve_density
        # For now, assume species index `is` corresponds to the neutral
        # species index `isn`.
        @loop_s_r_z is ir iz begin
            isn = is
            if moments.evolve_ppar
                # will need the ratio of thermal speeds both to interpolate between vpa grids
                # for different species and to account for different normalizations of each species' pdf
                vth_ratio = moments.ion.vth[iz,ir,is]/moments.neutral.vth[iz,ir,isn]
            else
                vth_ratio = 1.0
            end
            # if the parallel flow and/or the parallel pressure are separately evolved,
            # then the parallel velocity coordinate is re-defined so that the jth
            # vpa grid point for different species corresponds to different physical
            # values of dz/dt; as charge exchange and ionization collisions require
            # the evaluation of the pdf for species s' to obtain the update for species s,
            # will thus have to interpolate between the different vpa grids
            if moments.evolve_upar && moments.evolve_ppar
                # if evolve_ppar = true and evolve_upar = true, vpa coordinate is
                # wpahat_s = (vpa-upar_s)/vth_s;
                # we have f_{s'}(wpahat_{s'}) = f_{s'}((wpahat_s * vth_s + upar_s - upar_{s'}) / vth_{s'});
                # to get f_{s'}(wpahat_s), need to obtain wpahat_s grid locations
                # in terms of the wpahat_{s'} coordinate:
                # (wpahat_{s'})_j = ((wpahat_{s})_j * vth_{s} + upar_{s} - upar_{s'}) / vth_{s'}
                new_grid = @. vpa.scratch = (vpa.grid * moments.ion.vth[iz,ir,is] + fvec_in.upar[iz,ir,is] - fvec_in.uz_neutral[iz,ir,isn]) / moments.neutral.vth[iz,ir,isn]
            elseif !moments.evolve_upar
                # if evolve_ppar = true and evolve_upar = false, vpa coordinate is
                # vpahat_s = vpa/vth_s;
                # we have f_{s'}(vpahat_{s'}) = f_{s'}(vpahat_s * vth_s / vth_{s'});
                # to get f_{s'}(vpahat_s), need to obtain vpahat_s grid locations
                # in terms of the vpahat_{s'} coordinate:
                # (vpahat_s)_j = (vpahat_{s'})_j * vth_{s'} / vth_{s}
                new_grid = @. vpa.scratch = vpa.grid / vth_ratio
            elseif !moments.evolve_ppar
                # if evolve_ppar = false and evolve_upar = true, vpa coordinate is
                # wpa_s = vpa-upar_s;
                # we have f_{s'}(wpa_{s'}) = f_{s'}((wpa_s + upar_s - upar_{s'};
                # to get f_{s'}(wpa_s), need to obtain wpa_s grid locations
                # in terms of the wpa_{s'} coordinate:
                # (wpa_s)_j = (wpa_{s'})_j + upar_{s'} - upar_{s}
                new_grid = @. vpa.scratch = vpa.grid + fvec_in.upar[iz,ir,is] - fvec_in.uz_neutral[iz,ir,isn]
            else
                new_grid = vpa.grid
            end
            # interpolate to the new grid (passed in as vpa.scratch)
            # and return interpolated values in vpa.scratch2
            @views interpolate_to_grid_vpa!(vpa.scratch2, vpa.scratch, fvec_in.pdf_neutral[:,1,1,iz,ir,isn], vz, vz_spectral)
            ionization = collisions.reactions.ionization_frequency
            @loop_vpa ivpa begin
                f_out[ivpa,1,iz,ir,is] +=
                    dt*ionization*fvec_in.density_neutral[iz,ir,isn]*
                    (vpa.scratch2[ivpa]*vth_ratio - fvec_in.pdf[ivpa,1,iz,ir,is])
            end
        end
    else
        @loop_s is begin
            # ion ionisation rate =   < f_n > n_e R_ion
            # neutral "ionisation" (depopulation) rate =   -  f_n  n_e R_ion
            # no gyroaverage here as 1V code
            #NB: used quasineutrality to replace electron density n_e with ion density
            #NEEDS GENERALISATION TO n_ion_species > 1 (missing species charge: Sum_i Z_i n_i = n_e)
            ionization = collisions.reactions.ionization_frequency
            isn = is
            @loop_r_z ir iz begin
                @views interpolate_to_grid_vpa!(vpa.scratch, vpa.grid,
                                                fvec_in.pdf_neutral[:,1,1,iz,ir,isn], vz,
                                                vz_spectral)
                @loop_vpa ivpa begin
                    # apply ionization collisions to all ion species
                    f_out[ivpa,1,iz,ir,is] += dt*ionization*vpa.scratch[ivpa]*fvec_in.density[iz,ir,is]
                end
            end
        end
    end
end

@timeit global_timer neutral_ionization_collisions_1V!(
                         f_neutral_out, fvec_in, vz, vpa, vperp, z, r, vz_spectral,
                         moments, composition, collisions, dt) = begin
    # This routine assumes a 1D model with:
    # nvperp = nvr = nveta = 1
    # constant charge_exchange_frequency independent of species
    @boundscheck vz.n == size(f_neutral_out,1) || throw(BoundsError(f_neutral_out))
    @boundscheck 1 == size(f_neutral_out,2) || throw(BoundsError(f_neutral_out))
    @boundscheck 1 == size(f_neutral_out,3) || throw(BoundsError(f_neutral_out))
    @boundscheck z.n == size(f_neutral_out,4) || throw(BoundsError(f_neutral_out))
    @boundscheck r.n == size(f_neutral_out,5) || throw(BoundsError(f_neutral_out))
    @boundscheck composition.n_neutral_species == size(f_neutral_out,6) || throw(BoundsError(f_neutral_out))

    if !moments.evolve_density
        @begin_sn_r_z_vz_region()

        ionization = collisions.reactions.ionization_frequency
        @loop_sn isn begin
            # ion ionisation rate =   < f_n > n_e R_ion
            # neutral "ionisation" (depopulation) rate =   -  f_n  n_e R_ion
            # no gyroaverage here as 1V code
            #NB: used quasineutrality to replace electron density n_e with ion density
            #NEEDS GENERALISATION TO n_ion_species > 1 (missing species charge: Sum_i Z_i n_i = n_e)
            is = isn
            @loop_r_z_vz ir iz ivz begin
                # apply ionization collisions to all neutral species
                f_neutral_out[ivz,1,1,iz,ir,isn] -= dt*ionization*fvec_in.pdf_neutral[ivz,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
            end
        end
    end
end

@timeit global_timer ion_ionization_collisions_3V!(
                         f_out, f_neutral_gav_in, fvec_in, composition, vz, vr, vzeta,
                         vpa, vperp, z, r, collisions, dt) = begin
    # This routine assumes a 3V model with:
    @boundscheck vpa.n == size(f_out,1) || throw(BoundsError(f_out))
    @boundscheck vperp.n == size(f_out,2) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out,3) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out,4) || throw(BoundsError(f_out))
    @boundscheck composition.n_ion_species == size(f_out,5) || throw(BoundsError(f_out))
    @boundscheck vpa.n == size(f_neutral_gav_in,1) || throw(BoundsError(f_neutral_gav_in))
    @boundscheck vperp.n == size(f_neutral_gav_in,2) || throw(BoundsError(f_neutral_gav_in))
    @boundscheck z.n == size(f_neutral_gav_in,3) || throw(BoundsError(f_neutral_gav_in))
    @boundscheck r.n == size(f_neutral_gav_in,4) || throw(BoundsError(f_neutral_gav_in))
    @boundscheck composition.n_neutral_species == size(f_neutral_gav_in,5) || throw(BoundsError(f_neutral_gav_in))
    
    ionization_frequency = collisions.reactions.ionization_frequency
    
    @begin_s_r_z_vperp_vpa_region()

    # ion ionization rate =   < f_n > n_e R_ion
    # neutral "ionization" (depopulation) rate =   -  f_n  n_e R_ion
    #NB: used quasineutrality to replace electron density n_e with ion density
    #NEEDS GENERALISATION TO n_ion_species > 1 (missing species charge: Sum_i Z_i n_i = n_e)
    # for ion species we need gyroaveraged neutral pdf, which is not stored in fvec (scratch[istage])
    @loop_s is begin
        for isn ∈ 1:composition.n_neutral_species
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                # apply ionization collisions to all ion species
                f_out[ivpa,ivperp,iz,ir,is] += dt*ionization_frequency*f_neutral_gav_in[ivpa,ivperp,iz,ir,isn]*fvec_in.density[iz,ir,is]
            end
        end
    end
end

@timeit global_timer neutral_ionization_collisions_3V!(
                         f_neutral_out, fvec_in, composition, vz, vr, vzeta, vpa, vperp,
                         z, r, collisions, dt) = begin
    # This routine assumes a 3V model with:
    @boundscheck vz.n == size(f_neutral_out,1) || throw(BoundsError(f_neutral_out))
    @boundscheck vr.n == size(f_neutral_out,2) || throw(BoundsError(f_neutral_out))
    @boundscheck vzeta.n == size(f_neutral_out,3) || throw(BoundsError(f_neutral_out))
    @boundscheck z.n == size(f_neutral_out,4) || throw(BoundsError(f_neutral_out))
    @boundscheck r.n == size(f_neutral_out,5) || throw(BoundsError(f_neutral_out))
    @boundscheck composition.n_neutral_species == size(f_neutral_out,6) || throw(BoundsError(f_neutral_out))

    ionization_frequency = collisions.reactions.ionization_frequency

    # ion ionization rate =   < f_n > n_e R_ion
    # neutral "ionization" (depopulation) rate =   -  f_n  n_e R_ion
    #NB: used quasineutrality to replace electron density n_e with ion density
    #NEEDS GENERALISATION TO n_ion_species > 1 (missing species charge: Sum_i Z_i n_i = n_e)
    @begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn isn begin
        for is ∈ 1:composition.n_ion_species
            @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
                # apply ionization collisions to all neutral species
                f_neutral_out[ivz,ivr,ivzeta,iz,ir,isn] -= dt*ionization_frequency*fvec_in.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]*fvec_in.density[iz,ir,is]
            end
        end
    end
end

end
