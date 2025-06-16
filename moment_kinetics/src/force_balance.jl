"""
"""
module force_balance

export force_balance!

using ..calculus: derivative!
using ..looping
using ..timer_utils
using ..input_structs: coll_krook_ions

"""
use the force balance equation d(nu)/dt + d(ppar + n*upar*upar)/dz =
-(dens/2)*dphi/dz + R*dens_i*dens_n*(upar_n-upar_i)
to update the parallel particle flux dens*upar for each species
"""
@timeit global_timer force_balance!(
                         upar_out, density_out, fvec, moments, fields, collisions, dt,
                         spectral, composition, geometry, ion_source_settings,
                         num_diss_params, z) = begin
    @begin_s_r_z_region()

    dnupar_dt = moments.ion.dnupar_dt

    # account for momentum flux contribution to force balance
    density = fvec.density
    upar = fvec.upar
    @loop_s_r_z is ir iz begin
        dnupar_dt[iz,ir,is] = -(moments.ion.dppar_dz[iz,ir,is] +
                                upar[iz,ir,is]*upar[iz,ir,is]*moments.ion.ddens_dz_upwind[iz,ir,is] +
                                2.0*density[iz,ir,is]*upar[iz,ir,is]*moments.ion.dupar_dz_upwind[iz,ir,is] -
                                geometry.bzed[iz,ir]*fields.Ez[iz,ir]*density[iz,ir,is])
    end

    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active && false
            @views source_amplitude = moments.ion.external_source_momentum_amplitude[:, :, index]
            @loop_s_r_z is ir iz begin
                dnupar_dt[iz,ir,is] += source_amplitude[iz,ir]
            end
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.ion.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_s_r_z is ir iz begin
            dnupar_dt[iz,ir,is] += diffusion_coefficient*moments.ion.d2upar_dz2[iz,ir,is]*density[iz,ir,is]
        end
    end

    # if neutrals present account for charge exchange and/or ionization collisions
    if composition.n_neutral_species > 0
        # account for collisional friction between ions and neutrals
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0
            @loop_s_r_z is ir iz begin
                dnupar_dt[iz,ir,is] += charge_exchange*density[iz,ir,is]*fvec.density_neutral[iz,ir,is]*(fvec.uz_neutral[iz,ir,is]-upar[iz,ir,is])
            end
        end
        # account for ionization collisions
        if abs(ionization) > 0.0
            @loop_s_r_z is ir iz begin
                dnupar_dt[iz,ir,is] += ionization*density[iz,ir,is]*fvec.density_neutral[iz,ir,is]*fvec.uz_neutral[iz,ir,is]
            end
        end
    end

    @loop_s_r_z is ir iz begin
        # convert from the particle flux to the parallel flow
        upar_out[iz,ir,is] = (density[iz,ir,is]*upar[iz,ir,is] + dt * dnupar_dt[iz,ir,is]) / density_out[iz,ir,is]
    end

    if composition.ion_physics == coll_krook_ions
        # boundary condition for fluid simulation on ion flow at wall is that flow must be at least sonic.
        if z.irank == 0 && (z.irank == z.nrank - 1)
            z_indices = (1, z.n)
        elseif z.irank == 0
            z_indices = (1,)
        elseif z.irank == z.nrank - 1
            z_indices = (z.n,)
        else
            return nothing
        end
        T_e = composition.T_e
        @loop_s_r is ir begin
            for iz ∈ z_indices
                # set the ion flow to local sound speed at wall
                if iz == 1
                    if upar_out[iz,ir,is] > -sqrt(T_e + moments.ion.temp[iz,ir,is])
                        upar_out[iz,ir,is] = -sqrt(T_e + moments.ion.temp[iz,ir,is])
                    end
                else
                    if upar_out[iz,ir,is] < sqrt(T_e + moments.ion.temp[iz,ir,is])
                        upar_out[iz,ir,is] = sqrt(T_e + moments.ion.temp[iz,ir,is])
                    end
                end
            end
        end
    end
    return nothing
end

@timeit global_timer neutral_force_balance!(
                         uz_out, density_out, fvec, moments, fields, collisions, dt,
                         spectral, composition, geometry, neutral_source_settings,
                         num_diss_params) = begin

    @begin_sn_r_z_region()

    dnuz_dt = moments.neutral.dnuz_dt

    # account for momentum flux contribution to force balance
    density = fvec.density_neutral
    uz = fvec.uz_neutral
    @loop_sn_r_z isn ir iz begin
        dnuz_dt[iz,ir,isn] = -(moments.neutral.dpz_dz[iz,ir,isn] +
                               uz[iz,ir,isn]*uz[iz,ir,isn]*moments.neutral.ddens_dz_upwind[iz,ir,isn] +
                               2.0*density[iz,ir,isn]*uz[iz,ir,isn]*moments.neutral.duz_dz_upwind[iz,ir,isn])
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active && false
            @views source_amplitude = moments.neutral.external_source_momentum_amplitude[:, :, index]
            @loop_sn_r_z isn ir iz begin
                dnuz_dt[iz,ir,isn] += source_amplitude[iz,ir]
            end
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.neutral.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        @loop_sn_r_z isn ir iz begin
            dnuz_dt[iz,ir,isn] += diffusion_coefficient*moments.neutral.d2uz_dz2[iz,ir,isn]*density[iz,ir,isn]
        end
    end

    # if neutrals present account for charge exchange and/or ionization collisions
    if composition.n_neutral_species > 0
        # account for collisional friction between ions and neutrals
        charge_exchange = collisions.reactions.charge_exchange_frequency
        ionization = collisions.reactions.ionization_frequency
        if abs(charge_exchange) > 0.0
            @loop_sn_r_z isn ir iz begin
                dnuz_dt[iz,ir,isn] += charge_exchange*density[iz,ir,isn]*fvec.density[iz,ir,isn]*(fvec.upar[iz,ir,isn]-uz[iz,ir,isn])
            end
        end
        # account for ionization collisions
        if abs(ionization) > 0.0
            @loop_sn_r_z isn ir iz begin
                dnuz_dt[iz,ir,isn] -= ionization*fvec.density[iz,ir,isn]*density[iz,ir,isn]*uz[iz,ir,isn]
            end
        end
    end

    @loop_sn_r_z isn ir iz begin
        # convert from the particle flux to the parallel flow
        uz_out[iz,ir,isn] = (density[iz,ir,isn]*uz[iz,ir,isn] + dt * dnuz_dt[iz,ir,isn]) / density_out[iz,ir,isn]
    end

    return nothing
end

end
