"""
"""
module ionization

export ionization_collisions_1V!
export ionization_collisions_3V!

using ..looping

"""
"""
function ionization_collisions_1V!(f_out, f_neutral_out, fvec_in, vpa, vperp, z, r, composition, collisions, dt)
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

    if collisions.constant_ionization_rate
        # Oddly the test in test/harrisonthompson.jl matches the analitical
        # solution (which assumes width=0.0) better with width=0.5 than with,
        # e.g., width=0.15. Possibly narrower widths would require more vpa
        # resolution, which then causes crashes due to overshoots giving
        # negative f??
        width = 0.5
        @loop_s is begin
            @loop_r_z_vpa ir iz ivpa begin
                f_out[ivpa,1,iz,ir,is] += dt*collisions.ionization/width*exp(-(vpa.grid[ivpa]/width)^2)
            end
        end
    else
        @loop_s is begin
            # ion ionisation rate =   < f_n > n_e R_ion
            # neutral "ionisation" (depopulation) rate =   -  f_n  n_e R_ion
            # no gyroaverage here as 1V code
            #NB: used quasineutrality to replace electron density n_e with ion density
            #NEEDS GENERALISATION TO n_ion_species > 1 (missing species charge: Sum_i Z_i n_i = n_e)
            @loop_sn isn begin
                @loop_r_z_vpa ir iz ivpa begin
                    # apply ionization collisions to all ion species
                    f_out[ivpa,1,iz,ir,is] += dt*collisions.ionization*fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
                    # apply ionization collisions to all neutral species
                    f_neutral_out[ivpa,1,1,iz,ir,isn] -= dt*collisions.ionization*fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
                end
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
