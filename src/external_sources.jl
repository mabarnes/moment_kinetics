"""
Maxwellian source terms with spatially varying parameters representing external sources of
particles and energy.

Note there is no parallel momentum input from the external sources.
"""
module external_sources

export setup_external_sources!, external_ion_source, external_neutral_source

using ..coordinates
using ..input_structs: set_defaults_and_check_section!, Dict_to_NamedTuple
using ..looping

"""
    setup_external_sources!(input_dict, r, z)

Set up parameters for the external sources using settings in `input_dict`.

Updates `input_dict` with defaults for unset parameters.

`r` and `z` are the [`coordinates.coordinate`](@ref) objects for the r-
and z-coordinates.

Returns a NamedTuple `(ion=ion_source_settings, neutral=neutral_source_settings)`
containing two NamedTuples of settings.
"""
function setup_external_sources!(input_dict, r, z)
    external_ion_sources, external_neutral_sources = (
        set_defaults_and_check_section!(
             input_dict, section_name;
             active=false,
             source_strength=1.0,
             source_T=1.0,
             r_profile="constant",
             r_width=1.0,
             r_relative_minimum=0.0,
             z_profile="constant",
             z_width=1.0,
             z_relative_minimum=0.0,
            )
        for section_name ∈ ("ion_source", "neutral_source"))

    function get_settings(input)
        r_amplitude = get_source_profile(input["r_profile"], input["r_width"],
                                         input["r_relative_minimum"], r)
        z_amplitude = get_source_profile(input["z_profile"], input["z_width"],
                                         input["z_relative_minimum"], z)

        return (; (Symbol(k)=>v for (k,v) ∈ input)..., r_amplitude=r_amplitude,
                z_amplitude=z_amplitude)
    end

    return (ion=get_settings(external_ion_sources),
            neutral=get_settings(external_neutral_sources))
end

"""
    get_source_profile(profile_type, width, min_val, coord)

Create a profile of type `profile_type` with width `width` for coordinate `coord`.
"""
function get_source_profile(profile_type, width, relative_minimum, coord)
    if width < 0.0
        error("width must be ≥0, got $width")
    end
    if relative_minimum < 0.0
        error("relative_minimum must be ≥0, got $relative_minimum")
    end
    if profile_type == "constant"
        return ones(coord.n)
    elseif profile_type == "gaussian"
        x = coord.grid
        return @. (1.0 - relative_minimum) * exp(-(x / width)^2) + relative_minimum
    elseif profile_type == "parabolic"
        x = coord.grid
        L = coord.L
        return @. (1.0 - relative_minimum) * (1.0 - (2.0 * x / L)^2) + relative_minimum
    else
        error("Unrecognised source profile type '$profile_type'.")
    end
end

"""
    external_ion_source(pdf, fvec, moments, ion_source_settings, vperp, vpa, dt)

Add external source term to the ion kinetic equation.
"""
function external_ion_source(pdf, fvec, moments, ion_source_settings, vperp, vpa, dt)
    begin_s_r_z_vperp_region()

    source_strength = ion_source_settings.source_strength
    source_T = ion_source_settings.source_T
    r_amplitude = ion_source_settings.r_amplitude
    z_amplitude = ion_source_settings.z_amplitude
    vpa_grid = vpa.grid
    vperp_grid = vperp.grid

    if vperp.n == 1
        # 1V case
        prefactor = source_strength / sqrt(source_T)
    else
        prefactor = source_strength / source_T^1.5
    end

    if moments.evolve_ppar && moments.evolve_upar && moments.evolve_density
        vth = moments.charged.vth
        density = fvec.density
        upar = fvec.upar
        @loop_s_r_z is ir iz begin
            this_vth = vth[iz,ir,is]
            this_upar = upar[iz,ir,is]
            this_prefactor = dt * this_vth / density[iz,ir,is] * prefactor *
                             r_amplitude[ir] * z_amplitude[iz]
            @loop_vperp_vpa ivperp ivpa begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vperp_unnorm = vperp_grid[ivperp] * this_vth
                vpa_unnorm = vpa_grid[ivpa] * this_vth + this_upar
                pdf[ivpa,ivperp,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vperp_unnorm^2 + vpa_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_upar && moments.evolve_density
        density = fvec.density
        upar = fvec.upar
        @loop_s_r_z is ir iz begin
            this_upar = upar[iz,ir,is]
            this_prefactor = dt / density[iz,ir,is] * prefactor *
                             r_amplitude[ir] * z_amplitude[iz]
            @loop_vperp_vpa ivperp ivpa begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vpa_unnorm = vpa_grid[ivpa] + this_upar
                pdf[ivpa,ivperp,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vperp_grid[iz,ir,is]^2 + vpa_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_density
        density = fvec.density
        @loop_s_r_z is ir iz begin
            this_prefactor = dt / density[iz,ir,is] * prefactor * r_amplitude[ir] *
                             z_amplitude[iz]
            @loop_vperp_vpa ivperp ivpa begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                pdf[ivpa,ivperp,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vperp_grid[ivperp]^2 + vpa_grid[ivpa]^2) / source_T)
            end
        end
    elseif !moments.evolve_ppar && !moments.evolve_upar && !moments.evolve_density
        @loop_s_r_z is ir iz begin
            this_prefactor = dt * prefactor * r_amplitude[ir] * z_amplitude[iz]
            @loop_vperp_vpa ivperp ivpa begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                pdf[ivpa,ivperp,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vperp_grid[ivperp]^2 + vpa_grid[ivpa]^2) / source_T)
            end
        end
    else
        error("Unsupported combination evolve_density=$(moments.evolve_density), "
              * "evolve_upar=$(moments.evolve_upar), evolve_ppar=$(moments.evolve_ppar)")
    end
end

"""
    external_neutral_source(pdf, fvec, moments, neutral_source_settings, vzeta, vr,
                            vz, dt)

Add external source term to the neutral kinetic equation.
"""
function external_neutral_source(pdf, fvec, moments, neutral_source_settings, vzeta, vr,
                                 vz, dt)
    begin_s_r_z_vzeta_vr_region()

    source_strength = neutral_source_settings.source_strength
    source_T = neutral_source_settings.source_T
    r_amplitude = neutral_source_settings.r_amplitude
    z_amplitude = neutral_source_settings.z_amplitude
    vzeta_grid = vzeta.grid
    vr_grid = vr.grid
    vz_grid = vz.grid

    if vzeta.n == 1 && vr.n == 1
        # 1V case
        prefactor = source_strength / sqrt(source_T)
    else
        prefactor = source_strength / source_T^1.5
    end
    if moments.evolve_ppar && moments.evolve_upar && moments.evolve_density
        vth = moments.vth_neutral
        density = fvec.density_neutral
        uz = fvec.uz_neutral
        @loop_s_r_z is ir iz begin
            this_vth = vth[iz,ir,is]
            this_uz = uz[iz,ir,is]
            this_prefactor = dt * this_vth / density[iz,ir,is] * prefactor *
                             r_amplitude[ir] * z_amplitude[iz]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vzeta_unnorm = vzeta_grid[ivzeta] * this_vth
                vr_unnorm = vr_grid[ivr] * this_vth
                vz_unnorm = vz_grid[ivz] * this_vth + this_uz
                pdf[ivz,ivr,ivzeta,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vzeta_unnorm^2 + vr_unnorm^2 + vz_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_upar && moments.evolve_density
        density = fvec.density_neutral
        uz = fvec.uz_neutral
        @loop_s_r_z is ir iz begin
            this_prefactor = dt / density[iz,ir,is] * prefactor *
                             r_amplitude[ir] * z_amplitude[iz]
            this_uz = uz[iz,ir,is]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vz_unnorm = vz_grid[ivz] + this_uz
                pdf[ivz,ivr,ivzeta,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vzeta_grid[ivz]^2 + vr_grid[ivr]^2 + vz_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_density
        density = fvec.density_neutral
        @loop_s_r_z is ir iz begin
            this_prefactor = dt / density[iz,ir,is] * prefactor * r_amplitude[ir] *
                             z_amplitude[iz]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                pdf[ivz,ivr,ivzeta,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vzeta_grid[ivzeta]^2 + vr_grid[ivr]^2 + vz_grid[ivz]^2) / source_T)
            end
        end
    elseif !moments.evolve_ppar && !moments.evolve_upar && !moments.evolve_density
        @loop_s_r_z is ir iz begin
            this_prefactor = dt * prefactor * r_amplitude[ir] * z_amplitude[iz]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                pdf[ivz,ivr,ivzeta,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vzeta_grid[ivzeta]^2 + vr_grid[ivr]^2 + vz_grid[ivz]^2) / source_T)
            end
        end
    else
        error("Unsupported combination evolve_density=$(moments.evolve_density), "
              * "evolve_upar=$(moments.evolve_upar), evolve_ppar=$(moments.evolve_ppar)")
    end
end

end
