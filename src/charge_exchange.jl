"""
"""
module charge_exchange

#export charge_exchange_collisions!
export charge_exchange_collisions_1V!

using ..looping

"""
"""
function charge_exchange_collisions_1V!(f_out, f_neutral_out, fvec_in, composition, vpa, vperp, z, r,
                                     charge_exchange_frequency, dt)
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
    begin_s_sn_r_z_region()
    
    @loop_s is begin
        # apply CX collisions to all ion species 
        # for each ion species, obtain affect of charge exchange collisions
        # with all of the neutral species
        @loop_sn isn begin
            # if different CX frequency per species
            #cxfac = dt*charge_exchange_frequency[is,isn]
            #cxfac = dt*charge_exchange_frequency
            @loop_r_z_vpa ir iz ivpa begin
                f_out[ivpa,1,iz,ir,is] +=
                    dt*charge_exchange_frequency*(
                        fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
                        - fvec_in.pdf[ivpa,1,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn])
            end
        end
    
        # apply CX collisions to all neutral species
        # for each neutral species, obtain affect of charge exchange collisions
        # with all of the ion species
        @loop_sn isn begin
            #cxfac = dt*charge_exchange_frequency
            @loop_r_z_vpa ir iz ivpa begin
                f_neutral_out[ivpa,1,1,iz,ir,isn] +=
                    dt*charge_exchange_frequency*(
                        fvec_in.pdf[ivpa,1,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn]
                        - fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is])
            end
        end
    end
end

end
