"""
"""
module continuity

export continuity_equation!

using ..calculus: derivative!
using ..looping
using ..timer_utils

"""
use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all ion
species
"""
@timeit global_timer continuity_equation!(
                         dens_out, fvec_in, moments, composition, dt, spectral,
                         ionization, ion_source_settings, num_diss_params) = begin
    @begin_s_r_z_region()

    @loop_s_r_z is ir iz begin
        # Use ddens_dz is upwinded using upar
        dens_out[iz,ir,is] -=
            dt*(fvec_in.upar[iz,ir,is]*moments.ion.ddens_dz_upwind[iz,ir,is] +
                fvec_in.density[iz,ir,is]*moments.ion.dupar_dz[iz,ir,is])
    end

    # update the density to account for ionization collisions;
    # ionization collisions increase the density for ions and decrease the density for neutrals
    if composition.n_neutral_species > 0 && ionization > 0.0
        @loop_s_r_z is ir iz begin
            dens_out[iz,ir,is] += dt*ionization*fvec_in.density[iz,ir,is]*fvec_in.density_neutral[iz,ir,is]
        end
    end

    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active
            @views source_amplitude = moments.ion.external_source_density_amplitude[:, :, index]
            @loop_s_r_z is ir iz begin
                dens_out[iz,ir,is] +=
                    dt * source_amplitude[iz,ir]
            end
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.ion.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_s_r_z is ir iz begin
            dens_out[iz,ir,is] += dt*diffusion_coefficient*moments.ion.d2dens_dz2[iz,ir,is]
        end
    end
end

"""
use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all neutral
species
"""
@timeit global_timer neutral_continuity_equation!(
                         dens_out, fvec_in, moments, composition, dt, spectral,
                         ionization, neutral_source_settings, num_diss_params) = begin
    @begin_sn_r_z_region()

    @loop_sn_r_z isn ir iz begin
        # Use ddens_dz is upwinded using uz
        dens_out[iz,ir,isn] -=
            dt*(fvec_in.uz_neutral[iz,ir,isn]*moments.neutral.ddens_dz_upwind[iz,ir,isn] +
                fvec_in.density_neutral[iz,ir,isn]*moments.neutral.duz_dz[iz,ir,isn])
    end

    # update the density to account for ionization collisions;
    # ionization collisions increase the density for ions and decrease the density for neutrals
    if composition.n_neutral_species > 0 && ionization > 0.0
        @loop_sn_r_z isn ir iz begin
            dens_out[iz,ir,isn] -= dt*ionization*fvec_in.density[iz,ir,isn]*fvec_in.density_neutral[iz,ir,isn]
        end
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_amplitude = moments.neutral.external_source_density_amplitude[:, :, index]
            @loop_s_r_z is ir iz begin
                dens_out[iz,ir,is] +=
                    dt * source_amplitude[iz,ir]
            end
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.neutral.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_sn_r_z isn ir iz begin
            dens_out[iz,ir,isn] += dt*diffusion_coefficient*moments.neutral.d2dens_dz2[iz,ir,isn]
        end
    end
end

end
