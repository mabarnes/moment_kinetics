"""
"""
module charge_exchange

export charge_exchange_collisions_3V!
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
    begin_r_z_vpa_region()
    
    @loop_s_sn is isn begin
        @loop_r_z_vpa ir iz ivpa begin
            # apply CX collisions to all ion species
            # for each ion species, obtain affect of charge exchange collisions
            # with all of the neutral species
            f_out[ivpa,1,iz,ir,is] +=
                dt*charge_exchange_frequency*(
                    fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is]
                    - fvec_in.pdf[ivpa,1,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn])
            # apply CX collisions to all neutral species
            # for each neutral species, obtain affect of charge exchange collisions
            # with all of the ion species
            f_neutral_out[ivpa,1,1,iz,ir,isn] +=
                dt*charge_exchange_frequency*(
                    fvec_in.pdf[ivpa,1,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn]
                    - fvec_in.pdf_neutral[ivpa,1,1,iz,ir,isn]*fvec_in.density[iz,ir,is])
        end
    end
end

function charge_exchange_collisions_3V!(f_out, f_neutral_out, f_neutral_gav_in, f_charged_vrvzvzeta_in, fvec_in, composition, vz, vr, vzeta, vpa, vperp, z, r,
                                     charge_exchange_frequency, dt)
    # This routine assumes a 3V model with:
    @boundscheck vz.n == size(f_neutral_out,1) || throw(BoundsError(f_neutral_out))
    @boundscheck vr.n == size(f_neutral_out,2) || throw(BoundsError(f_neutral_out))
    @boundscheck vzeta.n == size(f_neutral_out,3) || throw(BoundsError(f_neutral_out))
    @boundscheck z.n == size(f_neutral_out,4) || throw(BoundsError(f_neutral_out))
    @boundscheck r.n == size(f_neutral_out,5) || throw(BoundsError(f_neutral_out))
    @boundscheck composition.n_neutral_species == size(f_neutral_out,6) || throw(BoundsError(f_neutral_out))
    @boundscheck vz.n == size(f_charged_vrvzvzeta_in,1) || throw(BoundsError(f_charged_vrvzvzeta_in))
    @boundscheck vr.n == size(f_charged_vrvzvzeta_in,2) || throw(BoundsError(f_charged_vrvzvzeta_in))
    @boundscheck vzeta.n == size(f_charged_vrvzvzeta_in,3) || throw(BoundsError(f_charged_vrvzvzeta_in))
    @boundscheck z.n == size(f_charged_vrvzvzeta_in,4) || throw(BoundsError(f_charged_vrvzvzeta_in))
    @boundscheck r.n == size(f_charged_vrvzvzeta_in,5) || throw(BoundsError(f_charged_vrvzvzeta_in))
    @boundscheck composition.n_neutral_species == size(f_charged_vrvzvzeta_in,6) || throw(BoundsError(f_charged_vrvzvzeta_in))
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

    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        # apply CX collisions to all ion species
        # for each ion species, obtain affect of charge exchange collisions
        # with all of the neutral species
        for isn ∈ 1:composition.n_neutral_species
            f_out[ivpa,ivperp,iz,ir,is] +=
                dt*charge_exchange_frequency*(
                    f_neutral_gav_in[ivpa,ivperp,iz,ir,isn]*fvec_in.density[iz,ir,is]
                    - fvec_in.pdf[ivpa,ivperp,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn])
        end
    end
    begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        # apply CX collisions to all neutral species
        # for each neutral species, obtain affect of charge exchange collisions
        # with all of the ion species
        for is ∈ 1:composition.n_ion_species
            f_neutral_out[ivz,ivr,ivzeta,iz,ir,isn] +=
                dt*charge_exchange_frequency*(
                    f_charged_vrvzvzeta_in[ivz,ivr,ivzeta,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn]
                    - fvec_in.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]*fvec_in.density[iz,ir,is])
        end
    end
end

end
