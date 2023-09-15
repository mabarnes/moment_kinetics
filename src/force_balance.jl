"""
"""
module force_balance

export force_balance!

using ..calculus: derivative!
using ..looping

"""
use the force balance equation d(nu)/dt + d(ppar + n*upar*upar)/dz =
-(dens/2)*dphi/dz + R*dens_i*dens_n*(upar_n-upar_i)
to update the parallel particle flux dens*upar for each species
"""
function force_balance!(pflx, density_out, fvec, moments, fields, collisions, dt,
                        spectral, composition, geometry, ion_source_settings,
                        num_diss_params)

    begin_s_r_z_region()

    # account for momentum flux contribution to force balance
    density = fvec.density
    upar = fvec.upar
    @loop_s_r_z is ir iz begin
        pflx[iz,ir,is] = density[iz,ir,is]*upar[iz,ir,is] -
                         dt*(moments.charged.dppar_dz[iz,ir,is] +
                             upar[iz,ir,is]*upar[iz,ir,is]*moments.charged.ddens_dz_upwind[iz,ir,is] +
                             2.0*density[iz,ir,is]*upar[iz,ir,is]*moments.charged.dupar_dz_upwind[iz,ir,is] -
                             0.5*geometry.bzed*fields.Ez[iz,ir]*density[iz,ir,is])
    end

    if ion_source_settings.active && false
        source_amplitude = moments.charged.external_source_momentum_amplitude
        @loop_s_r_z is ir iz begin
            pflx[iz,ir,is] +=
                dt * source_amplitude[iz,ir]
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_s_r_z is ir iz begin
            pflx[iz,ir,is] += dt*diffusion_coefficient*moments.charged.d2upar_dz2[iz,ir,is]*density[iz,ir,is]
        end
    end

    # if neutrals present account for charge exchange and/or ionization collisions
    if composition.n_neutral_species > 0
        # account for collisional friction between ions and neutrals
        if abs(collisions.charge_exchange) > 0.0
            @loop_s_r_z is ir iz begin
                pflx[iz,ir,is] += dt*collisions.charge_exchange*density[iz,ir,is]*fvec.density_neutral[iz,ir,is]*(fvec.uz_neutral[iz,ir,is]-upar[iz,ir,is])
            end
        end
        # account for ionization collisions
        if abs(collisions.ionization) > 0.0
            @loop_s_r_z is ir iz begin
                pflx[iz,ir,is] += dt*collisions.ionization*density[iz,ir,is]*fvec.density_neutral[iz,ir,is]*fvec.uz_neutral[iz,ir,is]
            end
        end
    end

    @loop_s_r_z is ir iz begin
        # convert from the particle flux to the parallel flow
        pflx[iz,ir,is] /= density_out[iz,ir,is]
    end
end

function neutral_force_balance!(pflx, density_out, fvec, moments, fields, collisions, dt,
                                spectral, composition, geometry, neutral_source_settings,
                                num_diss_params)

    begin_sn_r_z_region()

    # account for momentum flux contribution to force balance
    density = fvec.density_neutral
    uz = fvec.uz_neutral
    @loop_sn_r_z isn ir iz begin
        pflx[iz,ir,isn] = density[iz,ir,isn]*uz[iz,ir,isn] -
                          dt*(moments.neutral.dpz_dz[iz,ir,isn] +
                             uz[iz,ir,isn]*uz[iz,ir,isn]*moments.neutral.ddens_dz_upwind[iz,ir,isn] +
                             2.0*density[iz,ir,isn]*uz[iz,ir,isn]*moments.neutral.duz_dz_upwind[iz,ir,isn])
    end

    if neutral_source_settings.active && false
        source_amplitude = moments.neutral.external_source_momentum_amplitude
        @loop_sn_r_z isn ir iz begin
            pflx[iz,ir,isn] +=
                dt * source_amplitude[iz,ir]
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_sn_r_z isn ir iz begin
            pflx[iz,ir,isn] += dt*diffusion_coefficient*moments.neutral.d2uz_dz2[iz,ir,isn]*density[iz,ir,isn]
        end
    end

    # if neutrals present account for charge exchange and/or ionization collisions
    if composition.n_neutral_species > 0
        # account for collisional friction between ions and neutrals
        if abs(collisions.charge_exchange) > 0.0
            @loop_sn_r_z isn ir iz begin
                pflx[iz,ir,isn] += dt*collisions.charge_exchange*density[iz,ir,isn]*fvec.density[iz,ir,isn]*(fvec.upar[iz,ir,isn]-uz[iz,ir,isn])
            end
        end
        # account for ionization collisions
        if abs(collisions.ionization) > 0.0
            @loop_sn_r_z isn ir iz begin
                pflx[iz,ir,isn] -= dt*collisions.ionization*fvec.density[iz,ir,isn]*density[iz,ir,isn]*uz[iz,ir,isn]
            end
        end
    end

    @loop_sn_r_z isn ir iz begin
        # convert from the particle flux to the parallel flow
        pflx[iz,ir,isn] /= density_out[iz,ir,isn]
    end
end

end
