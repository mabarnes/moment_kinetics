"""
"""
module charge_exchange

export charge_exchange_collisions_3V!
export charge_exchange_collisions_1V!

using ..looping
using ..interpolation: interpolate_to_grid_vpa!
using ..timer_utils

"""
update the evolved pdf for each ion species to account for charge exchange collisions
between ions and neutrals
"""
@timeit global_timer ion_charge_exchange_collisions_1V!(
                         f_out, fvec_in, moments, composition, vpa, vz,
                         charge_exchange_frequency, vpa_spectral, vz_spectral, dt) = begin
    # This routine assumes a 1D model with:
    # nvz = nvpa and identical vz and vpa grids 

    if moments.evolve_density
        @begin_s_r_z_region()
        @loop_s is begin
            # apply CX collisions to all ion species
            # for each ion species, obtain affect of charge exchange collisions
            # with the correponding neutral species
            @views charge_exchange_collisions_single_species!(
                f_out[:,1,:,:,is], fvec_in.pdf[:,1,:,:,is],
                fvec_in.pdf_neutral[:,1,1,:,:,is],
                fvec_in.density_neutral[:,:,is], fvec_in.upar[:,:,is],
                fvec_in.uz_neutral[:,:,is], moments.ion.vth[:,:,is],
                moments.neutral.vth[:,:,is], moments, vpa, vz, charge_exchange_frequency,
                vz_spectral, dt; neutrals=false)
        end
    else
        @begin_s_r_z_region()
        @loop_s is begin
            # apply CX collisions to all ion species
            # for each ion species, obtain affect of charge exchange collisions
            # with the corresponding neutral species
            isn = is
            @loop_r_z ir iz begin
                @views interpolate_to_grid_vpa!(vpa.scratch, vpa.grid,
                                                fvec_in.pdf_neutral[:,1,1,iz,ir,isn], vz,
                                                vz_spectral)
                @loop_vpa ivpa begin
                    f_out[ivpa,1,iz,ir,is] +=
                        dt*charge_exchange_frequency*(
                            vpa.scratch[ivpa]*fvec_in.density[iz,ir,is]
                            - fvec_in.pdf[ivpa,1,iz,ir,is]*fvec_in.density_neutral[iz,ir,is])
                end
            end
        end
    end
end

"""
update the evolved pdf for each neutral species to account for charge exchange collisions
between ions and neutrals
"""
@timeit global_timer neutral_charge_exchange_collisions_1V!(
                         f_neutral_out, fvec_in, moments, composition, vpa, vz,
                         charge_exchange_frequency, vpa_spectral, vz_spectral, dt) = begin
    # This routine assumes a 1D model with:
    # nvz = nvpa and identical vz and vpa grids

    if moments.evolve_density
        @begin_sn_r_z_region()
        @loop_sn isn begin
            # apply CX collisions to all neutral species
            # for each neutral species, obtain affect of charge exchange collisions
            # with the corresponding ion species
            @views charge_exchange_collisions_single_species!(
                f_neutral_out[:,1,1,:,:,isn], fvec_in.pdf_neutral[:,1,1,:,:,isn],
                fvec_in.pdf[:,1,:,:,isn], fvec_in.density[:,:,isn],
                fvec_in.uz_neutral[:,:,isn], fvec_in.upar[:,:,isn],
                moments.neutral.vth[:,:,isn], moments.ion.vth[:,:,isn], moments,
                vz, vpa, charge_exchange_frequency, vpa_spectral, dt; neutrals=true)
        end
    else
        @begin_sn_r_z_region()
        @loop_sn isn begin
            # apply CX collisions to all neutral species
            # for each neutral species, obtain affect of charge exchange collisions
            # with the corresponding ion species
            @loop_r_z ir iz begin
                @views interpolate_to_grid_vpa!(vz.scratch, vz.grid,
                                                fvec_in.pdf[:,1,iz,ir,isn], vpa,
                                                vpa_spectral)
                @loop_vz ivz begin
                    f_neutral_out[ivz,1,1,iz,ir,isn] +=
                        dt*charge_exchange_frequency*(
                            vz.scratch[ivz]*fvec_in.density_neutral[iz,ir,isn]
                            - fvec_in.pdf_neutral[ivz,1,1,iz,ir,isn]*fvec_in.density[iz,ir,isn])
                end
            end
        end
    end
end

"""
update the evolved pdf for a single species to account for charge exchange collisions
with a single species of the opposite type; e.g., ions with neutrals or neutrals with ions
"""
function charge_exchange_collisions_single_species!(f_out, pdf_in, pdf_other,
        density_other, upar, upar_other, vth, vth_other, moments, vpa, vpa_other,
        charge_exchange_frequency, spectral_other, dt; neutrals)
    @loop_r_z ir iz begin
        if moments.evolve_p
            # will need the ratio of thermal speeds both to interpolate between vpa grids
            # for different species and to account for different normalizations of each species' pdf
            vth_ratio = vth[iz,ir]/vth_other[iz,ir]
        else
            vth_ratio = 1.0
        end
        # if the parallel flow and/or the parallel pressure are separately evolved,
        # then the parallel velocity coordinate is re-defined so that the jth
        # vpa grid point for different species corresponds to different physical
        # values of dz/dt; as charge exchange and ionization collisions require
        # the evaluation of the pdf for species s' to obtain the update for species s,
        # will thus have to interpolate between the different vpa grids
        if moments.evolve_upar && moments.evolve_p
            # if evolve_p = true and evolve_upar = true, vpa coordinate is
            # wpahat_s = (vpa-upar_s)/vth_s;
            # we have f_{s'}(wpahat_{s'}) = f_{s'}((wpahat_s * vth_s + upar_s - upar_{s'}) / vth_{s'});
            # to get f_{s'}(wpahat_s), need to obtain wpahat_s grid locations
            # in terms of the wpahat_{s'} coordinate:
            # (wpahat_{s'})_j = ((wpahat_{s})_j * vth_{s} + upar_{s} - upar_{s'}) / vth_{s'}
            new_grid = @. vpa.scratch = (vpa.grid * vth[iz,ir] + upar[iz,ir] - upar_other[iz,ir]) / vth_other[iz,ir]
        elseif !moments.evolve_upar
            # if evolve_p = true and evolve_upar = false, vpa coordinate is
            # vpahat_s = vpa/vth_s;
            # we have f_{s'}(vpahat_{s'}) = f_{s'}(vpahat_s * vth_s / vth_{s'});
            # to get f_{s'}(vpahat_s), need to obtain vpahat_s grid locations
            # in terms of the vpahat_{s'} coordinate:
            # (vpahat_s)_j = (vpahat_{s'})_j * vth_{s'} / vth_{s}
            new_grid = @. vpa.scratch = vpa.grid / vth_ratio
        elseif !moments.evolve_p
            # if evolve_p = false and evolve_upar = true, vpa coordinate is
            # wpa_s = vpa-upar_s;
            # we have f_{s'}(wpa_{s'}) = f_{s'}((wpa_s + upar_s - upar_{s'};
            # to get f_{s'}(wpa_s), need to obtain wpa_s grid locations
            # in terms of the wpa_{s'} coordinate:
            # (wpa_s)_j = (wpa_{s'})_j + upar_{s'} - upar_{s}
            new_grid = @. vpa.scratch = vpa.grid + upar[iz,ir] - upar_other[iz,ir]
        else
            # Interpolate even when using 'drift-kinetic' mode, so that vpa and vz
            # coordinates can be different.
            new_grid = vpa.grid
        end
        # interpolate to new_grid and return interpolated values in vpa.scratch2
        @views interpolate_to_grid_vpa!(vpa.scratch2, new_grid, pdf_other[:,iz,ir], vpa_other, spectral_other)

        if neutrals
            @loop_vz ivz begin
                f_out[ivz,iz,ir] += dt * charge_exchange_frequency * density_other[iz,ir] *
                (vpa.scratch2[ivz] * vth_ratio - pdf_in[ivz,iz,ir])
            end
        else
            @loop_vpa ivpa begin
                f_out[ivpa,iz,ir] += dt * charge_exchange_frequency * density_other[iz,ir] *
                (vpa.scratch2[ivpa] * vth_ratio - pdf_in[ivpa,iz,ir])
            end
        end
    end
end

@timeit global_timer ion_charge_exchange_collisions_3V!(
                         f_out, f_neutral_gav_in, fvec_in, composition, vz, vr, vzeta,
                         vpa, vperp, z, r, charge_exchange_frequency, dt) = begin
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

    @begin_s_r_z_vperp_vpa_region()
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
end

@timeit global_timer neutral_charge_exchange_collisions_3V!(
                         f_neutral_out, f_ion_vrvzvzeta_in, fvec_in, composition, vz, vr,
                         vzeta, vpa, vperp, z, r, charge_exchange_frequency, dt) = begin
    # This routine assumes a 3V model with:
    @boundscheck vz.n == size(f_neutral_out,1) || throw(BoundsError(f_neutral_out))
    @boundscheck vr.n == size(f_neutral_out,2) || throw(BoundsError(f_neutral_out))
    @boundscheck vzeta.n == size(f_neutral_out,3) || throw(BoundsError(f_neutral_out))
    @boundscheck z.n == size(f_neutral_out,4) || throw(BoundsError(f_neutral_out))
    @boundscheck r.n == size(f_neutral_out,5) || throw(BoundsError(f_neutral_out))
    @boundscheck composition.n_neutral_species == size(f_neutral_out,6) || throw(BoundsError(f_neutral_out))
    @boundscheck vz.n == size(f_ion_vrvzvzeta_in,1) || throw(BoundsError(f_ion_vrvzvzeta_in))
    @boundscheck vr.n == size(f_ion_vrvzvzeta_in,2) || throw(BoundsError(f_ion_vrvzvzeta_in))
    @boundscheck vzeta.n == size(f_ion_vrvzvzeta_in,3) || throw(BoundsError(f_ion_vrvzvzeta_in))
    @boundscheck z.n == size(f_ion_vrvzvzeta_in,4) || throw(BoundsError(f_ion_vrvzvzeta_in))
    @boundscheck r.n == size(f_ion_vrvzvzeta_in,5) || throw(BoundsError(f_ion_vrvzvzeta_in))
    @boundscheck composition.n_neutral_species == size(f_ion_vrvzvzeta_in,6) || throw(BoundsError(f_ion_vrvzvzeta_in))

    @begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        # apply CX collisions to all neutral species
        # for each neutral species, obtain affect of charge exchange collisions
        # with all of the ion species
        for is ∈ 1:composition.n_ion_species
            f_neutral_out[ivz,ivr,ivzeta,iz,ir,isn] +=
                dt*charge_exchange_frequency*(
                    f_ion_vrvzvzeta_in[ivz,ivr,ivzeta,iz,ir,is]*fvec_in.density_neutral[iz,ir,isn]
                    - fvec_in.pdf_neutral[ivz,ivr,ivzeta,iz,ir,isn]*fvec_in.density[iz,ir,is])
        end
    end
end

end
