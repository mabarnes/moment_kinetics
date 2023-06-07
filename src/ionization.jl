"""
"""
module ionization

export ionization_collisions_1V!
export ionization_collisions_3V!
export constant_ionization_source!

using ..interpolation: interpolate_to_grid_vpa!
using ..looping

"""
"""

function constant_ionization_source!(f_out, vpa, vperp, z, r, moments, composition,
                                     collisions, dt)
    @boundscheck vpa.n == size(f_out,1) || throw(BoundsError(f_out))
    @boundscheck vperp.n == size(f_out,2) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out,3) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out,4) || throw(BoundsError(f_out))
    @boundscheck composition.n_ion_species == size(f_out,5) || throw(BoundsError(f_out))
    
    begin_s_r_z_region()

    # Oddly the test in test/harrisonthompson.jl matches the analitical
    # solution (which assumes width=0.0) better with width=0.5 than with,
    # e.g., width=0.15. Possibly narrower widths would require more vpa
    # resolution, which then causes crashes due to overshoots giving
    # negative f??
    width = 0.5
    rwidth = 0.25
    @loop_s_r is ir begin
        rfac = exp( - (r.grid[ir]/rwidth)^2)

        @loop_z iz begin
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
            @loop_vpa ivpa begin
                f_out[ivpa,1,iz,ir,is] += dt*rfac*collisions.ionization/width*prefactor*exp(-(vpa.scratch[ivpa]/width)^2)
            end
        end
    end
end 

function ionization_collisions_1V!(f_out, f_neutral_out, fvec_in, vz, vpa, vperp, z, r,
                                   vz_spectral, moments, composition, collisions, dt)
    # This routine assumes a 1D model with:
    # nvz = nvpa and identical vz and vpa grids 
    # nvperp = nvr = nveta = 1
    # constant charge_exchange_frequency independent of species
    @boundscheck vpa.n == size(f_neutral_out,1) || throw(BoundsError(f_neutral_out))
    @boundscheck 1 == size(f_neutral_out,2) || throw(BoundsError(f_neutral_out))
    @boundscheck 1 == size(f_neutral_out,3) || throw(BoundsError(f_neutral_out))
    @boundscheck z.n == size(f_neutral_out,4) || throw(BoundsError(f_neutral_out))
    @boundscheck r.n == size(f_neutral_out,5) || throw(BoundsError(f_neutral_out))
    @boundscheck composition.n_neutral_species == size(f_neutral_out,6) || throw(BoundsError(f_neutral_out))
    @boundscheck vpa.n == size(f_out,1) || throw(BoundsError(f_out))
    @boundscheck 1 == size(f_out,2) || throw(BoundsError(f_out))
    @boundscheck z.n == size(f_out,3) || throw(BoundsError(f_out))
    @boundscheck r.n == size(f_out,4) || throw(BoundsError(f_out))
    @boundscheck composition.n_ion_species == size(f_out,5) || throw(BoundsError(f_out))
    
    
    # keep vpa vperp vz vr vzeta local so that
    # vpa loop below can also be used for vz
    begin_r_z_vpa_region()

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
                    @. vpa.scratch = vpa.grid + fvec_in.upar[iz,ir,is] - fvec_in.uz_neutral[iz,ir,isn]
                else
                    # if evolve_ppar = true and evolve_upar = true, vpa coordinate is
                    # wpahat_s = (vpa-upar_s)/vth_s;
                    # we have f_{s'}(wpahat_{s'}) = f_{s'}((wpahat_s * vth_s + upar_s - upar_{s'}) / vth_{s'});
                    # to get f_{s'}(wpahat_s), need to obtain wpahat_s grid locations
                    # in terms of the wpahat_{s'} coordinate:
                    # (wpahat_{s'})_j = ((wpahat_{s})_j * vth_{s} + upar_{s} - upar_{s'}) / vth_{s'}
                    @. vpa.scratch = (vpa.grid * moments.ion.vth[iz,ir,is] + fvec_in.upar[iz,ir,is] - fvec_in.uz_neutral[iz,ir,isn]) / moments.neutral.vth[iz,ir,isn]
                end
                # interpolate to the new grid (passed in as vpa.scratch)
                # and return interpolated values in vpa.scratch2
                @views interpolate_to_grid_vpa!(vpa.scratch2, vpa.scratch, fvec_in.pdf_neutral[:,1,1,iz,ir,isn], vz, vz_spectral)
            else
                # no need to interpolate if neither upar or ppar evolved separately from pdf
                vpa.scratch2 .= fvec_in.pdf_neutral[:,1,1,iz,ir,isn]
            end
            @loop_vpa ivpa begin
                f_out[ivpa,1,iz,ir,is] +=
                    dt*collisions.ionization*fvec_in.density_neutral[iz,ir,isn]*
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
            isn = is
            @loop_r_z_vpa ir iz ivpa begin
                # apply ionization collisions to all ion species
                f_out[ivpa,1,iz,ir,is] += dt*collisions.ionization*fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
                # apply ionization collisions to all neutral species
                f_neutral_out[ivpa,1,1,iz,ir,isn] -= dt*collisions.ionization*fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
            end
        end
    end
end

function ionization_collisions_3V!(f_out, f_neutral_out, f_neutral_gav_in, fvec_in, composition, vz, vr, vzeta, vpa, vperp, z, r, collisions, dt)
    # This routine assumes a 3V model with:
    @boundscheck vz.n == size(f_neutral_out,1) || throw(BoundsError(f_neutral_out))
    @boundscheck vr.n == size(f_neutral_out,2) || throw(BoundsError(f_neutral_out))
    @boundscheck vzeta.n == size(f_neutral_out,3) || throw(BoundsError(f_neutral_out))
    @boundscheck z.n == size(f_neutral_out,4) || throw(BoundsError(f_neutral_out))
    @boundscheck r.n == size(f_neutral_out,5) || throw(BoundsError(f_neutral_out))
    @boundscheck composition.n_neutral_species == size(f_neutral_out,6) || throw(BoundsError(f_neutral_out))
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
    
    ionization_frequency = collisions.ionization
    
    begin_s_r_z_vperp_vpa_region()

    #    #if collisions.constant_ionization_rate
    #    #    ## Oddly the test in test/harrisonthompson.jl matches the analitical
    #    #    ## solution (which assumes width=0.0) better with width=0.5 than with,
    #    #    ## e.g., width=0.15. Possibly narrower widths would require more vpa
    #    #    ## resolution, which then causes crashes due to overshoots giving
    #    #    ## negative f??
    #    #    #width = 0.5
    #    #    #@loop_s is begin
    #    #    #    #@loop_r_z_vperp_vpa ir iz ivperp ivpa begin
    #    #    #    #    #f_out[ivpa,ivperp,iz,ir,is] += dt*collisions.ionization/width^3*exp(-((vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)/width^2))
    #    #    #    #end
    #    #    #end
    #    #    #return nothing
    #    #end

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
    begin_sn_r_z_vzeta_vr_vz_region()
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
