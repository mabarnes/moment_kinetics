"""
Maxwellian source terms with spatially varying parameters representing external sources of
particles and energy.

Note there is no parallel momentum input from the external sources.

The sources can be controlled by a PI controller to set density to a target value or
profile. Note that the PI controller should not be used with operator splitting -
implementing it in a way that would be compatible with splitting is complicated because
the source contributes to several terms.
"""
module external_sources

export setup_external_sources!, external_ion_source, external_neutral_source,
       external_ion_source_controller!, external_neutral_source_controller!,
       initialize_external_source_amplitude!,
       initialize_external_source_controller_integral!

using ..array_allocation: allocate_shared_float
using ..communication
using ..coordinates
using ..input_structs: set_defaults_and_check_section!, Dict_to_NamedTuple
using ..looping

using MPI

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
             controller_type="",
             PI_density_controller_P=0.0,
             PI_density_controller_I=0.0,
             PI_density_target_amplitude=1.0,
             PI_density_target_r_profile="constant",
             PI_density_target_r_width=1.0,
             PI_density_target_r_relative_minimum=0.0,
             PI_density_target_z_profile="constant",
             PI_density_target_z_width=1.0,
             PI_density_target_z_relative_minimum=0.0,
            )
        for section_name ∈ ("ion_source", "neutral_source"))

    function get_settings(input)
        r_amplitude = get_source_profile(input["r_profile"], input["r_width"],
                                         input["r_relative_minimum"], r)
        z_amplitude = get_source_profile(input["z_profile"], input["z_width"],
                                         input["z_relative_minimum"], z)
        if input["controller_type"] == "density_profile"
            PI_density_target_amplitude = input["PI_density_target_amplitude"]
            PI_density_target_r_factor =
                get_source_profile(input["PI_density_target_r_profile"],
                    input["PI_density_target_r_width"],
                    input["PI_density_target_r_relative_minimum"], r)
            PI_density_target_z_factor =
                get_source_profile(input["PI_density_target_z_profile"],
                    input["PI_density_target_z_width"],
                    input["PI_density_target_z_relative_minimum"], z)
            PI_density_target = allocate_shared_float(z.n,r.n)
            for ir ∈ 1:r.n, iz ∈ 1:z.n
                PI_density_target[iz,ir] =
                    PI_density_target_amplitude * PI_density_target_r_factor[ir] *
                    PI_density_target_z_factor[iz]
            end
            PI_controller_amplitude = nothing
            controller_source_profile = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        elseif input["controller_type"] == "density_midpoint"
            PI_density_target = input["PI_density_target_amplitude"]
            PI_controller_amplitude = allocate_shared_float(1)

            controller_source_profile = allocate_shared_float(z.n, r.n)
            for ir ∈ 1:r.n, iz ∈ 1:z.n
                controller_source_profile[iz,ir] = r_amplitude[ir] * z_amplitude[iz]
            end

            # Find the indices, and process rank of the point at r=0, z=0.
            # The result of findfirst() will be `nothing` if the point was not found.
            PI_density_target_ir = findfirst(x->abs(x)<1.e-14, r.grid)
            PI_density_target_iz = findfirst(x->abs(x)<1.e-14, z.grid)
            if block_rank[] == 0
                # Only need to do communications from the root process of each
                # shared-memory block
                if PI_density_target_ir !== nothing && PI_density_target_iz !== nothing
                    PI_density_target_rank = iblock_index[]
                else
                    PI_density_target_rank = 0
                end
                PI_density_target_rank = MPI.Allreduce(PI_density_target_rank, +,
                                                       comm_inter_block[])
            else
                PI_density_target_rank = nothing
            end
        elseif input["controller_type"] == ""
            PI_density_target = nothing
            PI_controller_amplitude = nothing
            controller_source_profile = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        else
            error("Unrecognised controller_type=$(input["controller_type"])."
                  * "Possible values are: \"\", density_profile, density_midpoint")
        end

        return (; (Symbol(k)=>v for (k,v) ∈ input)..., r_amplitude=r_amplitude,
                z_amplitude=z_amplitude, PI_density_target=PI_density_target,
                PI_controller_amplitude=PI_controller_amplitude,
                controller_source_profile=controller_source_profile,
                PI_density_target_ir=PI_density_target_ir,
                PI_density_target_iz=PI_density_target_iz,
                PI_density_target_rank=PI_density_target_rank)
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
    initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                          vzeta, vr, n_neutral_species)

Initialize the arrays `moments.charged.external_source_amplitude` and
`moments.neutral.external_source_amplitude`, using the settings in
`external_source_settings`
"""
function initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                               vzeta, vr, n_neutral_species)
    ion_source_settings = external_source_settings.ion
    if ion_source_settings.active
        @loop_r_z ir iz begin
            moments.charged.external_source_amplitude[iz,ir] =
                ion_source_settings.source_strength *
                ion_source_settings.r_amplitude[ir] * ion_source_settings.z_amplitude[iz]
        end
    end

    if n_neutral_species > 0
        neutral_source_settings = external_source_settings.neutral
        if neutral_source_settings.active
            @loop_r_z ir iz begin
                moments.neutral.external_source_amplitude[iz,ir] =
                    neutral_source_settings.source_strength *
                    neutral_source_settings.r_amplitude[ir] *
                    neutral_source_settings.z_amplitude[iz]
            end
        end
    end

    return nothing
end

"""
function initialize_external_source_controller_integral!(
             moments, external_source_settings, n_neutral_species)

Initialize the arrays `moments.charged.external_source_controller_integral` and
`moments.neutral.external_source_controller_integral`, using the settings in
`external_source_settings`
"""
function initialize_external_source_controller_integral!(
             moments, external_source_settings, n_neutral_species)
    ion_source_settings = external_source_settings.ion
    if ion_source_settings.active
        if ion_source_settings.PI_density_controller_I != 0.0 &&
            ion_source_settings.controller_type != ""
            moments.charged.external_source_controller_integral .= 0.0
        end
    end

    if n_neutral_species > 0
        neutral_source_settings = external_source_settings.neutral
        if neutral_source_settings.active
            if neutral_source_settings.PI_density_controller_I != 0.0 &&
                neutral_source_settings.controller_type != ""
                moments.neutral.external_source_controller_integral .= 0.0
            end
        end
    end

    return nothing
end

"""
    external_ion_source(pdf, fvec, moments, ion_source_settings, vperp, vpa, dt)

Add external source term to the ion kinetic equation.
"""
function external_ion_source(pdf, fvec, moments, ion_source_settings, vperp, vpa, dt)
    begin_s_r_z_vperp_region()

    source_amplitude = moments.charged.external_source_amplitude
    source_T = ion_source_settings.source_T
    if vperp.n == 1
        vth_factor = 1.0 / sqrt(source_T)
    else
        vth_factor = 1.0 / source_T^1.5
    end
    vpa_grid = vpa.grid
    vperp_grid = vperp.grid

    if moments.evolve_ppar && moments.evolve_upar && moments.evolve_density
        vth = moments.charged.vth
        density = fvec.density
        upar = fvec.upar
        @loop_s_r_z is ir iz begin
            this_vth = vth[iz,ir,is]
            this_upar = upar[iz,ir,is]
            this_prefactor = dt * this_vth / density[iz,ir,is] * vth_factor *
                             source_amplitude[iz,ir]
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
            this_prefactor = dt / density[iz,ir,is] * vth_factor * source_amplitude[iz,ir]
            @loop_vperp_vpa ivperp ivpa begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vpa_unnorm = vpa_grid[ivpa] + this_upar
                pdf[ivpa,ivperp,iz,ir,is] +=
                    this_prefactor *
                    exp(-(vperp_grid[ivperp]^2 + vpa_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_density
        density = fvec.density
        @loop_s_r_z is ir iz begin
            this_prefactor = dt / density[iz,ir,is] * vth_factor * source_amplitude[iz,ir]
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
            this_prefactor = dt * vth_factor * source_amplitude[iz,ir]
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

    source_amplitude = moments.neutral.external_source_amplitude
    source_T = neutral_source_settings.source_T
    if vzeta.n == 1 && vr.n == 1
        vth_factor = 1.0 / sqrt(source_T)
    else
        vth_factor = 1.0 / source_T^1.5
    end
    vzeta_grid = vzeta.grid
    vr_grid = vr.grid
    vz_grid = vz.grid

    if moments.evolve_ppar && moments.evolve_upar && moments.evolve_density
        vth = moments.vth_neutral
        density = fvec.density_neutral
        uz = fvec.uz_neutral
        @loop_s_r_z is ir iz begin
            this_vth = vth[iz,ir,is]
            this_uz = uz[iz,ir,is]
            this_prefactor = dt * this_vth / density[iz,ir,is] * vth_factor *
                             source_amplitude[iz,ir]
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
            this_uz = uz[iz,ir,is]
            this_prefactor = dt / density[iz,ir,is] * vth_factor * source_amplitude[iz,ir]
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
            this_prefactor = dt / density[iz,ir,is] * vth_factor * source_amplitude[iz,ir]
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
            this_prefactor = dt * vth_factor * source_amplitude[iz,ir]
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

"""
    external_ion_source_controller!(fvec_in, ion_moments, ion_source_settings, dt)

Calculate the amplitude when using a PI controller for the density to set the external
source amplitude.
"""
function external_ion_source_controller!(fvec_in, ion_moments, ion_source_settings, dt)

    is = 1

    if ion_source_settings.controller_type == ""
        return nothing
    elseif ion_source_settings.controller_type == "density_midpoint"
        begin_serial_region()

        # controller_amplitude error is a shared memory Vector of length 1
        controller_amplitude = ion_source_settings.PI_controller_amplitude
        @serial_region begin
            if ion_source_settings.PI_density_target_ir !== nothing &&
                    ion_source_settings.PI_density_target_iz !== nothing
                # This process has the target point

                n_mid = fvec_in.density[ion_source_settings.PI_density_target_iz,
                                        ion_source_settings.PI_density_target_ir, is]
                n_error = ion_source_settings.PI_density_target - n_mid

                ion_moments.external_source_controller_integral[1,1] +=
                    dt * ion_source_settings.PI_density_controller_I * n_error

                # Only want a source, so never allow amplitude to be negative
                amplitude = max(
                    ion_source_settings.PI_density_controller_P * n_error +
                    ion_moments.external_source_controller_integral[1,1],
                    0)
            else
                amplitude = nothing
            end
            controller_amplitude[1] =
                MPI.Bcast(amplitude, ion_source_settings.PI_density_target_rank,
                          comm_inter_block)
        end

        begin_r_z_region()

        amplitude = controller_amplitude[1]
        @loop_r_z ir iz begin
            ion_moments.external_source_amplitude[iz,ir] =
                amplitude * ion_source_settings.controller_source_profile[iz,ir]
        end
    elseif ion_source_settings.controller_type == "density_profile"
        begin_r_z_region()

        density = fvec_in.density
        target = ion_source_settings.PI_density_target
        P = ion_source_settings.PI_density_controller_P
        I = ion_source_settings.PI_density_controller_I
        integral = ion_moments.external_source_controller_integral
        amplitude = ion_moments.external_source_amplitude
        @loop_r_z ir iz begin
            n_error = target[iz,ir] - density[iz,ir,is]
            integral[iz,ir] += dt * I * n_error
            # Only want a source, so never allow amplitude to be negative
            amplitude[iz,ir] = max(P * n_error + integral[iz,ir], 0)
        end
    else
        error("Unrecognised controller_type=$(ion_source_settings.controller_type)")
    end

    return nothing
end

"""
    external_neutral_source_controller!(fvec_in, neutral_moments,
                                        neutral_source_settings, dt)

Calculate the amplitude when using a PI controller for the density to set the external
source amplitude.
"""
function external_neutral_source_controller!(fvec_in, neutral_moments,
                                             neutral_source_settings, dt)

    is = 1

    if neutral_source_settings.controller_type == ""
        return nothing
    elseif neutral_source_settings.controller_type == "density_midpoint"
        begin_serial_region()

        # controller_amplitude error is a shared memory Vector of length 1
        controller_amplitude = neutral_source_settings.PI_controller_amplitude
        @serial_region begin
            if neutral_source_settings.PI_density_target_ir !== nothing &&
                    neutral_source_settings.PI_density_target_iz !== nothing
                # This process has the target point

                n_mid = fvec_in.density_neutral[neutral_source_settings.PI_density_target_iz,
                                                neutral_source_settings.PI_density_target_ir,
                                                is]
                n_error = neutral_source_settings.PI_density_target - n_mid

                neutral_moments.external_source_controller_integral[1,1] +=
                    dt * neutral_source_settings.PI_density_controller_I * n_error

                # Only want a source, so never allow amplitude to be negative
                amplitude = max(
                    neutral_source_settings.PI_density_controller_P * n_error +
                    neutral_moments.external_source_controller_integral[1,1],
                    0)
            else
                amplitude = nothing
            end
            controller_amplitude[1] =
                MPI.Bcast(amplitude, neutral_source_settings.PI_density_target_rank,
                          comm_inter_block)
        end

        begin_r_z_region()

        amplitude = controller_amplitude[1]
        @loop_r_z ir iz begin
            neutral_moments.external_source_amplitude[iz,ir] =
                amplitude * neutral_source_settings.controller_source_profile[iz,ir]
        end
    elseif neutral_source_settings.controller_type == "density_profile"
        begin_r_z_region()

        density = fvec_in.density_neutral
        target = neutral_source_settings.PI_density_target
        P = neutral_source_settings.PI_density_controller_P
        I = neutral_source_settings.PI_density_controller_I
        integral = neutral_moments.external_source_controller_integral
        amplitude = neutral_moments.external_source_amplitude
        @loop_r_z ir iz begin
            n_error = target[iz,ir] - density[iz,ir,is]
            integral[iz,ir] += dt * I * n_error
            amplitude[iz,ir] = P * n_error + integral[iz,ir]
        end
    else
        error("Unrecognised controller_type=$(neutral_source_settings.controller_type)")
    end

    return nothing
end

end
