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

export setup_external_sources!, external_ion_source!, external_neutral_source!,
       external_ion_source_controller!, external_neutral_source_controller!,
       initialize_external_source_amplitude!,
       initialize_external_source_controller_integral!,
       add_external_electron_source_to_Jacobian!,
       total_external_ion_sources!, total_external_neutral_sources!,
       total_external_ion_source_controllers!, total_external_neutral_source_controllers!,
       external_electron_source!, total_external_electron_sources!

using ..array_allocation: allocate_float, allocate_shared_float
using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..calculus
using ..communication
using ..coordinates
using ..input_structs
using ..looping
using ..timer_utils
using ..velocity_moments: get_density

using MPI
using OrderedCollections: OrderedDict

"""
    setup_external_sources!(input_dict, r, z)

Set up parameters for the external sources using settings in `input_dict`.

Updates `input_dict` with defaults for unset parameters.

`r` and `z` are the [`coordinates.coordinate`](@ref) objects for the r-
and z-coordinates.

Returns a NamedTuple `(ion=ion_source_settings, neutral=neutral_source_settings)`
containing two NamedTuples of settings.
"""
function setup_external_sources!(input_dict, r, z, electron_physics)
    function get_settings_ions(source_index, active_flag)
        input = set_defaults_and_check_section!(
                     input_dict, "ion_source_$source_index";
                     active=active_flag,
                     source_strength=1.0,
                     source_n=1.0,
                     source_T=1.0,
                     source_v0=0.0, # birth speed for "alphas" option
                     source_vpa0=0.0, # birth vpa for "beam" option
                     source_vperp0=0.0, # birth vperp for "beam" option
                     sink_strength=1.0, # strength of sink in "alphas-with-losses" & "beam-with-losses" option
                     sink_vth=0.0, # thermal speed for sink in "alphas-with-losses" & "beam-with-losses" option 
                     r_profile="constant",
                     r_width=1.0,
                     r_relative_minimum=0.0,
                     z_profile="constant",
                     z_width=1.0,
                     z_relative_minimum=0.0,
                     source_type="Maxwellian", # "energy", "alphas", "alphas-with-losses", "beam", "beam-with-losses"
                     PI_density_controller_P=0.0,
                     PI_density_controller_I=0.0,
                     PI_density_target_amplitude=1.0,
                     PI_density_target_r_profile="constant",
                     PI_density_target_r_width=1.0,
                     PI_density_target_r_relative_minimum=0.0,
                     PI_density_target_z_profile="constant",
                     PI_density_target_z_width=1.0,
                     PI_density_target_z_relative_minimum=0.0,
                     recycling_controller_fraction=0.0,
                    )

        r_amplitude = get_source_profile(input["r_profile"], input["r_width"],
                                         input["r_relative_minimum"], r)
        z_amplitude = get_source_profile(input["z_profile"], input["z_width"],
                                         input["z_relative_minimum"], z)
        if input["source_type"] == "density_profile_control"
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
            @serial_region begin
                for ir ∈ 1:r.n, iz ∈ 1:z.n
                    PI_density_target[iz,ir] =
                        PI_density_target_amplitude * PI_density_target_r_factor[ir] *
                        PI_density_target_z_factor[iz]
                end
            end
            PI_controller_amplitude = nothing
            controller_source_profile = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        elseif input["source_type"] == "density_midpoint_control"
            PI_density_target = input["PI_density_target_amplitude"]

            if comm_block[] != MPI.COMM_NULL
                PI_controller_amplitude = allocate_shared_float(1)
                controller_source_profile = allocate_shared_float(z.n, r.n)
            else
                PI_controller_amplitude = allocate_float(1)
                controller_source_profile = allocate_float(z.n, r.n)
            end
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
                if comm_inter_block[] != MPI.COMM_NULL
                    PI_density_target_rank = MPI.Allreduce(PI_density_target_rank, +,
                                                           comm_inter_block[])
                end
                if PI_density_target_rank == 0 && iblock_index[] == 0 &&
                        (PI_density_target_ir === nothing ||
                         PI_density_target_iz === nothing)
                    error("No grid point with r=0 and z=0 was found for the "
                          * "'density_midpoint' controller.")
                end
            else
                PI_density_target_rank = nothing
            end
        elseif input["source_type"] ∈ ("Maxwellian", "energy", "alphas", "alphas-with-losses", "beam", "beam-with-losses")
            PI_density_target = nothing
            PI_controller_amplitude = nothing
            controller_source_profile = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        else
            error("Unrecognised ion source_type=$(input["source_type"])."
                  * "Possible values are: Maxwellian, density_profile_control, "
                  * "density_midpoint_control, energy, alphas, alphas-with-losses, "
                  * "beam, beam-with-losses")
        end
        return ion_source_data(; OrderedDict(Symbol(k)=>v for (k,v) ∈ input)...,
                r_amplitude=r_amplitude, z_amplitude=z_amplitude,
                PI_density_target=PI_density_target,
                PI_controller_amplitude=PI_controller_amplitude,
                controller_source_profile=controller_source_profile,
                PI_density_target_ir=PI_density_target_ir,
                PI_density_target_iz=PI_density_target_iz,
                PI_density_target_rank=PI_density_target_rank)
    end

    function get_settings_neutrals(source_index, active_flag)
        input = set_defaults_and_check_section!(
                     input_dict, "neutral_source_$source_index";
                     active=active_flag,
                     source_strength=1.0,
                     source_n=1.0,
                     source_T=get(input_dict, "T_wall", 1.0),
                     source_v0=0.0, # birth speed for "alphas" option
                     source_vpa0=0.0, # birth vpa for "beam" option
                     source_vperp0=0.0, # birth vperp for "beam" option
                     sink_strength=1.0, # strength of sink in "alphas-with-losses" & "beam-with-losses" option
                     sink_vth=0.0, # thermal speed for sink in "alphas-with-losses" & "beam-with-losses" option 
                     r_profile="constant",
                     r_width=1.0,
                     r_relative_minimum=0.0,
                     z_profile="constant",
                     z_width=1.0,
                     z_relative_minimum=0.0,
                     source_type="Maxwellian", # "energy", "alphas", "alphas-with-losses", "beam", "beam-with-losses"
                     PI_density_controller_P=0.0,
                     PI_density_controller_I=0.0,
                     PI_density_target_amplitude=1.0,
                     PI_density_target_r_profile="constant",
                     PI_density_target_r_width=1.0,
                     PI_density_target_r_relative_minimum=0.0,
                     PI_density_target_z_profile="constant",
                     PI_density_target_z_width=1.0,
                     PI_density_target_z_relative_minimum=0.0,
                     recycling_controller_fraction=0.0,
                    )

        r_amplitude = get_source_profile(input["r_profile"], input["r_width"],
                                         input["r_relative_minimum"], r)
        z_amplitude = get_source_profile(input["z_profile"], input["z_width"],
                                         input["z_relative_minimum"], z)
        if input["source_type"] == "density_profile_control"
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
            @serial_region begin
                for ir ∈ 1:r.n, iz ∈ 1:z.n
                    PI_density_target[iz,ir] =
                        PI_density_target_amplitude * PI_density_target_r_factor[ir] *
                        PI_density_target_z_factor[iz]
                end
            end
            PI_controller_amplitude = nothing
            controller_source_profile = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        elseif input["source_type"] == "density_midpoint_control"
            PI_density_target = input["PI_density_target_amplitude"]

            if comm_block[] != MPI.COMM_NULL
                PI_controller_amplitude = allocate_shared_float(1)
                controller_source_profile = allocate_shared_float(z.n, r.n)
            else
                PI_controller_amplitude = allocate_float(1)
                controller_source_profile = allocate_float(z.n, r.n)
            end
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
                if comm_inter_block[] != MPI.COMM_NULL
                    PI_density_target_rank = MPI.Allreduce(PI_density_target_rank, +,
                                                           comm_inter_block[])
                end
                if PI_density_target_rank == 0 && iblock_index[] == 0 &&
                        (PI_density_target_ir === nothing ||
                         PI_density_target_iz === nothing)
                    error("No grid point with r=0 and z=0 was found for the "
                          * "'density_midpoint' controller.")
                end
            else
                PI_density_target_rank = nothing
            end
        elseif input["source_type"] == "recycling"
            recycling = input["recycling_controller_fraction"]
            if recycling ≤ 0.0
                # Don't allow 0.0 as this is the default value, but makes no sense to have
                # the recycling source active and not doing anything, so make sure user
                # remembered to set a non-zero value for recycling_controller_fraction.
                error("recycling_controller_fraction must be >0. Got $recycling")
            end
            if recycling > 1.0
                error("recycling_controller_fraction must be ≤1. Got $recycling")
            end

            if comm_block[] != MPI.COMM_NULL
                controller_source_profile = allocate_shared_float(z.n, r.n)
            else
                controller_source_profile = allocate_float(z.n, r.n)
            end
            if block_rank[] == 0
                for ir ∈ 1:r.n, iz ∈ 1:z.n
                    controller_source_profile[iz,ir] = r_amplitude[ir] * z_amplitude[iz]
                end
                # Normalise so that the integral of this profile over r and z is 1. That way
                # we can just multiply by the integral over r of the ion flux to the target to
                # get the source amplitude.
                for ir ∈ 1:r.n
                    @views r.scratch[ir] = integral(controller_source_profile[:,ir], z.wgts)
                end
                controller_source_integral = integral(r.scratch, r.wgts)
                if comm_inter_block[] != MPI.COMM_NULL
                    controller_source_integral = MPI.Allreduce(controller_source_integral,
                                                               +, comm_inter_block[])
                end
                controller_source_profile ./= controller_source_integral
            end

            PI_density_target = nothing
            PI_controller_amplitude = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        elseif input["source_type"] ∈ ("Maxwellian", "energy", "alphas", "alphas-with-losses", "beam", "beam-with-losses")
            PI_density_target = nothing
            PI_controller_amplitude = nothing
            controller_source_profile = nothing
            PI_density_target_ir = nothing
            PI_density_target_iz = nothing
            PI_density_target_rank = nothing
        else
            error("Unrecognised neutral source_type=$(input["source_type"])."
                  * "Possible values are: Maxwellian, density_profile_control, "
                  * "density_midpoint_control, energy, alphas, alphas-with-losses, "
                  * "beam, beam-with-losses, recycling (for neutrals only)")
        end

        return neutral_source_data(; OrderedDict(Symbol(k)=>v for (k,v) ∈ input)...,
                r_amplitude=r_amplitude, z_amplitude=z_amplitude,
                PI_density_target=PI_density_target,
                PI_controller_amplitude=PI_controller_amplitude,
                controller_source_profile=controller_source_profile,
                PI_density_target_ir=PI_density_target_ir,
                PI_density_target_iz=PI_density_target_iz,
                PI_density_target_rank=PI_density_target_rank)
    end
    function get_settings_electrons(i, ion_settings)
        # Note most settings for the electron source are copied from the ion source,
        # because we require that the particle sources are the same for ions and
        # electrons. `source_T` can be set independently, and when using
        # `source_type="energy"`, the `source_strength` could also be set.
        input = set_defaults_and_check_section!(
                     input_dict, "electron_source_$i";
                     source_strength=ion_settings.source_strength,
                     source_T=ion_settings.source_T,
                    )
        if ion_settings.source_type != "energy"
            # Need to keep same amplitude for ions and electrons so there is no charge
            # source.
            if input["source_strength"] != ion_settings.source_strength
                println("When not using source_type=\"energy\", source_strength for "
                        * "electrons must be equal to source_strength for ions to ensure "
                        * "no charge is injected by the source. Overriding electron "
                        * "source_strength...")
            end
            input["source_strength"] = ion_settings.source_strength
        end
        return electron_source_data(source_strength=input["source_strength"],
                                    source_T=input["source_T"],
                                    active=ion_settings.active,
                                    r_amplitude=ion_settings.r_amplitude,
                                    z_amplitude=ion_settings.z_amplitude,
                                    source_type=ion_settings.source_type)
    end

    # put all ion sources into ion_source_data struct vector
    ion_sources = ion_source_data[]
    counter = 1
    while "ion_source_$counter" ∈ keys(input_dict)
        push!(ion_sources, get_settings_ions(counter, true))
        counter += 1
    end

    # If there are no ion sources, add an inactive ion source to the vector
    if counter == 1
        push!(ion_sources, get_settings_ions(1, false))
    end

    # put all electron sources into electron_source_data struct vector, where 
    # each entry is a mirror of the ion source vector.
    electron_sources = electron_source_data[]
    if electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                           kinetic_electrons_with_temperature_equation)
        electron_sources = [get_settings_electrons(i, this_source) for (i,this_source) ∈ enumerate(ion_sources)]
    else
        electron_sources = [get_settings_electrons(1, get_settings_ions(1, false))]
    end

    # put all neutral sources into neutral_source_data struct vector
    neutral_sources = neutral_source_data[]
    counter = 1
    while "neutral_source_$counter" ∈ keys(input_dict)
        push!(neutral_sources, get_settings_neutrals(counter, true))
        counter += 1
    end
    # If there are no neutral sources, add an inactive neutral source to the vector
    if counter == 1
        inactive_neutral_source = get_settings_neutrals(1, false)
        push!(neutral_sources, inactive_neutral_source)
    end
    return (ion=ion_sources, electron=electron_sources, neutral=neutral_sources)
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
        profile = @. (1.0 - relative_minimum) * (1.0 - (2.0 * x / width)^2) + relative_minimum
        for i ∈ eachindex(profile)
            if profile[i] < relative_minimum
                profile[i] = relative_minimum
            end
        end
        return profile
    elseif profile_type == "wall_exp_decay"
        x = coord.grid
        L = coord.L
        return @. (1.0 - relative_minimum) * exp(-(x+0.5*L) / width) + relative_minimum +
                  (1.0 - relative_minimum) * exp(-(0.5*L-x) / width) + relative_minimum
    else
        error("Unrecognised source profile type '$profile_type'.")
    end
end

"""
    initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                          vzeta, vr, n_neutral_species)

Initialize the arrays `moments.ion.external_source_amplitude`,
`moments.ion.external_source_density_amplitude`,
`moments.ion.external_source_momentum_amplitude`,
`moments.ion.external_source_pressure_amplitude`,
`moments.electron.external_source_amplitude`,
`moments.electron.external_source_density_amplitude`,
`moments.electron.external_source_momentum_amplitude`,
`moments.electron.external_source_pressure_amplitude`,
`moments.neutral.external_source_amplitude`,
`moments.neutral.external_source_density_amplitude`,
`moments.neutral.external_source_momentum_amplitude`, and
`moments.neutral.external_source_pressure_amplitude`, using the settings in
`external_source_settings`
"""
function initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                               vzeta, vr, n_neutral_species)
    begin_r_z_region()

    ion_source_settings = external_source_settings.ion
    # The electron loop must be in the same as the ion loop so that each electron source
    # can be matched to the corresponding ion source.
    electron_source_settings = external_source_settings.electron

    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active
            if ion_source_settings[index].source_type == "energy"
                @loop_r_z ir iz begin
                    moments.ion.external_source_amplitude[iz,ir,index] =
                        ion_source_settings[index].source_strength *
                        ion_source_settings[index].r_amplitude[ir] *
                        ion_source_settings[index].z_amplitude[iz]
                end
                if moments.evolve_density
                    @loop_r_z ir iz begin
                        moments.ion.external_source_density_amplitude[iz,ir,index] = 0.0
                    end
                end
                if moments.evolve_upar
                    @loop_r_z ir iz begin
                        moments.ion.external_source_momentum_amplitude[iz,ir,index]=
                            - moments.ion.dens[iz,ir] * moments.ion.upar[iz,ir] *
                            ion_source_settings[index].source_strength *
                            ion_source_settings[index].r_amplitude[ir] *
                            ion_source_settings[index].z_amplitude[iz]
                    end
                end
                if moments.evolve_ppar
                    @loop_r_z ir iz begin
                        moments.ion.external_source_pressure_amplitude[iz,ir,index] =
                            (0.5 * ion_source.source_T +
                            moments.ion.upar[iz,ir]^2 - moments.ion.ppar[iz,ir]) *
                            ion_source_settings[index].source_strength *
                            ion_source_settings[index].r_amplitude[ir] *
                            ion_source_settings[index].z_amplitude[iz]
                    end
                end
            else
                @loop_r_z ir iz begin
                    moments.ion.external_source_amplitude[iz,ir,index] =
                        ion_source_settings[index].source_strength *
                        ion_source_settings[index].r_amplitude[ir] *
                        ion_source_settings[index].z_amplitude[iz]
                end
                if moments.evolve_density
                    @loop_r_z ir iz begin
                        moments.ion.external_source_density_amplitude[iz,ir,index] =
                            ion_source_settings[index].source_strength *
                            ion_source_settings[index].r_amplitude[ir] *
                            ion_source_settings[index].z_amplitude[iz]
                    end
                end
                if moments.evolve_upar
                    @loop_r_z ir iz begin
                        moments.ion.external_source_momentum_amplitude[iz,ir,index] = 0.0
                    end
                end
                if moments.evolve_ppar
                    @loop_r_z ir iz begin
                        moments.ion.external_source_pressure_amplitude[iz,ir,index] =
                            (0.5 * ion_source_settings[index].source_T +
                            moments.ion.upar[iz,ir]^2) *
                            ion_source_settings[index].source_strength *
                            ion_source_settings[index].r_amplitude[ir] *
                            ion_source_settings[index].z_amplitude[iz]
                    end
                end
            end
        end
    end

    # now do same for electron sources, which (if present) are mostly mirrors of ion sources
    for index ∈ eachindex(electron_source_settings)
        if electron_source_settings[index].active
            if electron_source_settings[index].source_type == "energy"
                @loop_r_z ir iz begin
                    moments.electron.external_source_amplitude[iz,ir,index] =
                        electron_source_settings[index].source_strength *
                        electron_source_settings[index].r_amplitude[ir] *
                        electron_source_settings[index].z_amplitude[iz]
                end
                @loop_r_z ir iz begin
                    moments.electron.external_source_density_amplitude[iz,ir,index] = 0.0
                end
                @loop_r_z ir iz begin
                    moments.electron.external_source_momentum_amplitude[iz,ir,index] =
                        - moments.electron.dens[iz,ir] * moments.electron.upar[iz,ir] *
                        electron_source_settings[index].source_strength *
                        electron_source_settings[index].r_amplitude[ir] *
                        electron_source_settings[index].z_amplitude[iz]
                end
                @loop_r_z ir iz begin
                    moments.electron.external_source_pressure_amplitude[iz,ir,index] =
                        (0.5 * electron_source_settings[index].source_T +
                        moments.electron.upar[iz,ir]^2 - moments.electron.ppar[iz,ir]) *
                        electron_source_settings[index].source_strength *
                        electron_source_settings[index].r_amplitude[ir] *
                        electron_source_settings[index].z_amplitude[iz]
                end
            else
                @loop_r_z ir iz begin
                    moments.electron.external_source_amplitude[iz,ir,index] =
                        moments.ion.external_source_amplitude[iz,ir,index]
                end
                if moments.evolve_density
                    @loop_r_z ir iz begin
                        moments.electron.external_source_density_amplitude[iz,ir,index] =
                            moments.ion.external_source_density_amplitude[iz,ir,index]
                    end
                else
                    @loop_r_z ir iz begin
                        # Note set this using *ion* settings to force electron density source
                        # to always be equal to ion density source (even when
                        # evolve_density=false) to ensure the source does not inject charge
                        # into the simulation.
                        moments.electron.external_source_density_amplitude[iz,ir,index] =
                            ion_source_settings[index].source_strength *
                            ion_source_settings[index].r_amplitude[ir] *
                            ion_source_settings[index].z_amplitude[iz]
                    end
                end
                @loop_r_z ir iz begin
                    moments.electron.external_source_momentum_amplitude[iz,ir,index] = 0.0
                end
                @loop_r_z ir iz begin
                    moments.electron.external_source_pressure_amplitude[iz,ir,index] =
                        (0.5 * electron_source_settings[index].source_T +
                        moments.electron.upar[iz,ir]^2) *
                        moments.electron.external_source_amplitude[iz,ir,index]
                end
            end
        end
    end

    if n_neutral_species > 0
        neutral_source_settings = external_source_settings.neutral
        for index ∈ eachindex(neutral_source_settings)
            if neutral_source_settings[index].active
                if neutral_source_settings[index].source_type == "energy"
                    @loop_r_z ir iz begin
                        moments.neutral.external_source_amplitude[iz,ir,index] =
                            neutral_source_settings[index].source_strength *
                            neutral_source_settings[index].r_amplitude[ir] *
                            neutral_source_settings[index].z_amplitude[iz]
                    end
                    if moments.evolve_density
                        @loop_r_z ir iz begin
                            moments.neutral.external_source_density_amplitude[iz,ir,index] = 0.0
                        end
                    end
                    if moments.evolve_upar
                        @loop_r_z ir iz begin
                            moments.neutral.external_source_momentum_amplitude[iz,ir,index] =
                                - moments.neutral.dens[iz,ir] * moments.neutral.upar[iz,ir] *
                                neutral_source_settings[index].source_strength *
                                neutral_source_settings[index].r_amplitude[ir] *
                                neutral_source_settings[index].z_amplitude[iz]
                        end
                    end
                    if moments.evolve_ppar
                        @loop_r_z ir iz begin
                            moments.neutral.external_source_pressure_amplitude[iz,ir,index] =
                                (0.5 * neutral_source_settings[index].source_T +
                                moments.neutral.upar[iz,ir]^2 -
                                moments.neutral.ppar[iz,ir]) *
                                neutral_source_settings[index].source_strength *
                                neutral_source_settings[index].r_amplitude[ir] *
                                neutral_source_settings[index].z_amplitude[iz]
                        end
                    end
                else
                    @loop_r_z ir iz begin
                        moments.neutral.external_source_amplitude[iz,ir,index] =
                            neutral_source_settings[index].source_strength *
                            neutral_source_settings[index].r_amplitude[ir] *
                            neutral_source_settings[index].z_amplitude[iz]
                    end
                    if moments.evolve_density
                        @loop_r_z ir iz begin
                            moments.neutral.external_source_density_amplitude[iz,ir,index] =
                                neutral_source_settings[index].source_strength *
                                neutral_source_settings[index].r_amplitude[ir] *
                                neutral_source_settings[index].z_amplitude[iz]
                        end
                    end
                    if moments.evolve_upar
                        @loop_r_z ir iz begin
                            moments.neutral.external_source_momentum_amplitude[iz,ir,index] = 0.0
                        end
                    end
                    if moments.evolve_ppar
                        @loop_r_z ir iz begin
                            moments.neutral.external_source_pressure_amplitude[iz,ir,index] =
                                (0.5 * neutral_source_settings[index].source_T +
                                moments.neutral.uz[iz,ir]^2) *
                                neutral_source_settings[index].source_strength *
                                neutral_source_settings[index].r_amplitude[ir] *
                                neutral_source_settings[index].z_amplitude[iz]
                        end
                    end
                end
            end
        end
    end

    return nothing
end

"""
function initialize_external_source_controller_integral!(
             moments, external_source_settings, n_neutral_species)

Initialize the arrays `moments.ion.external_source_controller_integral` and
`moments.neutral.external_source_controller_integral`, using the settings in
`external_source_settings`
"""
function initialize_external_source_controller_integral!(
             moments, external_source_settings, n_neutral_species)
    begin_serial_region()
    @serial_region begin
        ion_source_settings = external_source_settings.ion
        for index ∈ eachindex(ion_source_settings)
            if ion_source_settings[index].active && 
               ion_source_settings[index].PI_density_controller_I != 0.0 && 
               ion_source_settings[index].source_type ∈ ("density_profile_control", "density_midpoint_control")
                moments.ion.external_source_controller_integral[:, :, index] .= 0.0
            end
        end

        if n_neutral_species > 0
            neutral_source_settings = external_source_settings.neutral
            for index ∈ eachindex(neutral_source_settings)
                if neutral_source_settings[index].active && 
                   neutral_source_settings[index].PI_density_controller_I != 0.0 && 
                   neutral_source_settings[index].source_type ∈ ("density_profile_control", "density_midpoint_control")
                    moments.neutral.external_source_controller_integral[:, :, index] .= 0.0
                end
            end
        end
    end

    return nothing
end

"""
    total_external_ion_sources!(pdf, fvec, moments, ion_sources, vperp, vpa, dt, scratch_dummy)

Contribute all of the ion sources to the ion pdf, one by one.
"""
function total_external_ion_sources!(pdf, fvec, moments, ion_sources, vperp, 
                                     vpa, dt, scratch_dummy)
    for index ∈ eachindex(ion_sources)
        if ion_sources[index].active
            external_ion_source!(pdf, fvec, moments, ion_sources[index], 
                                 index, vperp, vpa, dt, scratch_dummy)
        end
    end
    return nothing
end

"""
    external_ion_source!(pdf, fvec, moments, ion_source_settings, vperp, vpa, dt)

Add external source term to the ion kinetic equation.
"""
@timeit global_timer external_ion_source!(
                         pdf, fvec, moments, ion_source, index, vperp, vpa, dt,
                         scratch_dummy) = begin
    
    source_type = ion_source.source_type
    @views source_amplitude = moments.ion.external_source_amplitude[:,:,index]
    source_T = ion_source.source_T
    source_n = ion_source.source_n
    if vperp.n == 1
        vth_factor = 1.0 / sqrt(source_T)
    else
        vth_factor = 1.0 / source_T^1.5
    end
    vpa_grid = vpa.grid
    vperp_grid = vperp.grid
    if source_type in ("Maxwellian","energy")
        begin_s_r_z_vperp_region()
        if moments.evolve_ppar && moments.evolve_upar && moments.evolve_density
            vth = moments.ion.vth
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
                        this_prefactor * source_n *
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
                        this_prefactor * source_n *
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
                        this_prefactor * source_n *
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
                        this_prefactor * source_n *
                        exp(-(vperp_grid[ivperp]^2 + vpa_grid[ivpa]^2) / source_T)
                end
            end
        else
            error("Unsupported combination evolve_density=$(moments.evolve_density), "
                  * "evolve_upar=$(moments.evolve_upar), evolve_ppar=$(moments.evolve_ppar)")
        end

        if source_type == "energy"
            if moments.evolve_density
                # Take particles out of pdf so source does not change density
                @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
                    pdf[ivpa,ivperp,iz,ir,is] -= dt * source_amplitude[iz,ir] *
                        fvec.pdf[ivpa,ivperp,iz,ir,is]
                end
            else
                # Take particles out of pdf so source does not change density
                @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
                    pdf[ivpa,ivperp,iz,ir,is] -= dt * source_amplitude[iz,ir] *
                        fvec.pdf[ivpa,ivperp,iz,ir,is] / fvec.density[iz,ir,is]
                end
            end
        end
    elseif source_type == "alphas" || source_type == "alphas-with-losses"
        begin_s_r_z_region()
        source_v0 = ion_source.source_v0
        if !(source_v0 > 1.0e-8)
            error("source_v0=$source_v0 < 1.0e-8")
        end
        dummy_vpavperp = scratch_dummy.dummy_vpavperp
        if !moments.evolve_ppar && !moments.evolve_upar && !moments.evolve_density
            @loop_s_r_z is ir iz begin
                this_prefactor = dt * source_amplitude[iz,ir]
                # first assign source to local scratch array
                @loop_vperp_vpa ivperp ivpa begin
                    v2 = vperp_grid[ivperp]^2 + vpa_grid[ivpa]^2
                    fac = 2.0/(source_T*source_v0^2)
                    dummy_vpavperp[ivpa,ivperp] = exp(-fac*(v2 - source_v0^2)^2 )
                end
                # get the density for normalisation purposes
                normfac = get_density(dummy_vpavperp, vpa, vperp)
                # add the source
                @loop_vperp_vpa ivperp ivpa begin
                    # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                    # normalisation of F
                    pdf[ivpa,ivperp,iz,ir,is] +=
                        this_prefactor * dummy_vpavperp[ivpa,ivperp] / normfac
                end
            end
            
            if source_type == "alphas-with-losses"
                sink_vth = ion_source.sink_vth
                sink_strength = ion_source.sink_strength
                if !(sink_vth > 1.0e-8)
                   error("sink_vth=$sink_vth < 1.0e-8")
                end
                # subtract a sink function representing the loss of slow ash particles
                @loop_s_r_z is ir iz begin
                    # first assign sink to local scratch array
                    @loop_vperp_vpa ivperp ivpa begin
                        v2 = vperp_grid[ivperp]^2 + vpa_grid[ivpa]^2
                        fac = 1.0/(sink_vth^2)
                        dummy_vpavperp[ivpa,ivperp] = (1.0/(sink_vth^3))*exp(-fac*v2)
                    end
                    # numerical correction to normalisation
                    normfac = get_density(dummy_vpavperp, vpa, vperp)
                    # println("sink norm", normfac)
                    # add the source
                    @loop_vperp_vpa ivperp ivpa begin
                        # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                        # normalisation of F
                        pdf[ivpa,ivperp,iz,ir,is] -=
                            dt * sink_strength * dummy_vpavperp[ivpa,ivperp] * pdf[ivpa,ivperp,iz,ir,is] / normfac
                    end
                end
            end
        else
            error("Unsupported combination in source_type=$(source_type) evolve_density=$(moments.evolve_density), "
                  * "evolve_upar=$(moments.evolve_upar), evolve_ppar=$(moments.evolve_ppar)")
        end
    elseif source_type == "beam" || source_type == "beam-with-losses"
        begin_s_r_z_region()
        source_vpa0 = ion_source.source_vpa0
        source_vperp0 = ion_source.source_vperp0
        if !(source_vpa0 > 1.0e-8)
            error("source_vpa0=$source_vpa0 < 1.0e-8")
        end
        if !(source_vperp0 > 1.0e-8)
            error("source_vperp0=$source_vperp0 < 1.0e-8")
        end
        dummy_vpavperp = scratch_dummy.dummy_vpavperp
        if !moments.evolve_ppar && !moments.evolve_upar && !moments.evolve_density
            @loop_s_r_z is ir iz begin
                this_prefactor = dt * source_amplitude[iz,ir]
                # first assign source to local scratch array
                @loop_vperp_vpa ivperp ivpa begin
                    vth0  = sqrt(2.0*source_T) # sqrt(2 T / m), m = mref = 1
                    v2 = ((vperp_grid[ivperp]-source_vperp0)^2 + (vpa_grid[ivpa]-source_vpa0)^2)/(vth0^2)
                    dummy_vpavperp[ivpa,ivperp] = (1.0/vth0^3)*exp(-v2)
                end
                # get the density for normalisation purposes
                normfac = get_density(dummy_vpavperp, vpa, vperp)
                # add the source
                @loop_vperp_vpa ivperp ivpa begin
                    # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                    # normalisation of F
                    pdf[ivpa,ivperp,iz,ir,is] +=
                        this_prefactor * dummy_vpavperp[ivpa,ivperp] / normfac
                end
            end
            
            if source_type == "beam-with-losses"
                sink_vth = ion_source.sink_vth
                sink_strength = ion_source.sink_strength
                if !(sink_vth > 1.0e-8)
                   error("sink_vth=$sink_vth < 1.0e-8")
                end
                # subtract a sink function representing the loss of slow ash particles
                @loop_s_r_z is ir iz begin
                    # first assign sink to local scratch array
                    @loop_vperp_vpa ivperp ivpa begin
                        v2 = vperp_grid[ivperp]^2 + vpa_grid[ivpa]^2
                        fac = 1.0/(sink_vth^2)
                        dummy_vpavperp[ivpa,ivperp] = (1.0/(sink_vth^3))*exp(-fac*v2)
                    end
                    # numerical correction to normalisation
                    normfac = get_density(dummy_vpavperp, vpa, vperp)
                    # println("sink norm", normfac)
                    # add the source
                    @loop_vperp_vpa ivperp ivpa begin
                        # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                        # normalisation of F
                        pdf[ivpa,ivperp,iz,ir,is] -=
                            dt * sink_strength * dummy_vpavperp[ivpa,ivperp] * pdf[ivpa,ivperp,iz,ir,is] / normfac
                    end
                end
            end
        else
            error("Unsupported combination in source_type=$(source_type) evolve_density=$(moments.evolve_density), "
                  * "evolve_upar=$(moments.evolve_upar), evolve_ppar=$(moments.evolve_ppar)")
        end
    else
        error("Unsupported source_type=$(source_type) ")
    end
    return nothing
end

"""
    total_external_electron_sources!(pdf_out, pdf_in, electron_density, electron_upar,
                                     moments, composition, electron_sources, vperp,
                                     vpa, dt, ir)

Contribute all of the electron sources to the electron pdf, one by one.
"""
function total_external_electron_sources!(pdf_out, pdf_in, electron_density, electron_upar,
                                          moments, composition, electron_sources, vperp,
                                          vpa, dt, ir)
    for index ∈ eachindex(electron_sources)
        if electron_sources[index].active
            external_electron_source!(pdf_out, pdf_in, electron_density, electron_upar,
                                      moments, composition, electron_sources[index], index,
                                      vperp, vpa, dt, ir)
        end
    end
    return nothing
end

"""
    external_electron_source!(pdf_out, pdf_in, electron_density, electron_upar,
                              moments, composition, electron_source, index, vperp,
                              vpa, dt, ir)

Add external source term to the electron kinetic equation.

Note that this function operates on a single point in `r`, given by `ir`, and `pdf_out`,
`pdf_in`, `electron_density`, and `electron_upar` should have no r-dimension.
"""
@timeit global_timer external_electron_source!(
                         pdf_out, pdf_in, electron_density, electron_upar, moments,
                         composition, electron_source, index, vperp, vpa, dt, ir) = begin
    begin_z_vperp_region()

    me_over_mi = composition.me_over_mi

    @views source_amplitude = moments.electron.external_source_amplitude[:,ir,index]
    source_T = electron_source.source_T
    if vperp.n == 1
        vth_factor = 1.0 / sqrt(source_T / me_over_mi)
    else
        vth_factor = 1.0 / (source_T / me_over_mi)^1.5
    end
    vpa_grid = vpa.grid
    vperp_grid = vperp.grid

    vth = @view moments.electron.vth[:,ir]
    @loop_z iz begin
        this_vth = vth[iz]
        this_upar = electron_upar[iz]
        this_prefactor = dt * this_vth / electron_density[iz] * vth_factor *
                         source_amplitude[iz]
        @loop_vperp_vpa ivperp ivpa begin
            # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
            # normalisation of F
            vperp_unnorm = vperp_grid[ivperp] * this_vth
            vpa_unnorm = vpa_grid[ivpa] * this_vth + this_upar
            pdf_out[ivpa,ivperp,iz] +=
                this_prefactor *
                exp(-(vperp_unnorm^2 + vpa_unnorm^2) * me_over_mi / source_T)
        end
    end

    if electron_source.source_type == "energy"
        # Take particles out of pdf so source does not change density
        @loop_z_vperp_vpa iz ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz] -= dt * source_amplitude[iz] *
                                            pdf_in[ivpa,ivperp,iz]
        end
    end

    return nothing
end

function add_total_external_electron_source_to_Jacobian!(
        jacobian_matrix, f, moments, me, z_speed, electron_sources, z, vperp, vpa, dt, ir;
        f_offset=0, ppar_offset=0)
    for index ∈ eachindex(electron_sources)
        add_external_electron_source_to_Jacobian!(jacobian_matrix, f, moments, me,
                                                  z_speed, electron_sources[index], index,
                                                  z, vperp, vpa, dt, ir;
                                                  f_offset=f_offset,
                                                  ppar_offset=ppar_offset)
    end
end

function add_external_electron_source_to_Jacobian!(jacobian_matrix, f, moments, me,
                                                   z_speed, electron_source, index, z,
                                                   vperp, vpa, dt, ir; f_offset=0,
                                                   ppar_offset=0)
    if f_offset == ppar_offset
        error("Got f_offset=$f_offset the same as ppar_offset=$ppar_offset. f and ppar "
              * "cannot be in same place in state vector.")
    end
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2)
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n
    @boundscheck size(jacobian_matrix, 1) ≥ ppar_offset + z.n

    if !electron_source.active
        return nothing
    end

    source_amplitude = @view moments.electron.external_source_amplitude[:,ir,index]
    source_T = electron_source.source_T
    dens = @view moments.electron.dens[:,ir]
    upar = @view moments.electron.upar[:,ir]
    ppar = @view moments.electron.ppar[:,ir]
    vth = @view moments.electron.vth[:,ir]
    if vperp.n == 1
        vth_factor = 1.0 / sqrt(source_T / me)
    else
        vth_factor = 1.0 / sqrt(source_T / me)^1.5
    end
    vperp_grid = vperp.grid
    vpa_grid = vpa.grid
    v_size = vperp.n * vpa.n

    begin_z_vperp_vpa_region()
    if electron_source.source_type == "energy"
        @loop_z_vperp_vpa iz ivperp ivpa begin
            if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa,
                                                     z_speed)
                continue
            end

            # Rows corresponding to pdf_electron
            row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

            # Contribution from `external_electron_source!()`
            jacobian_matrix[row,row] += dt * source_amplitude[iz]
        end
    end
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

        # Contributions from
        #   -vth/n*vth_factor*source_amplitude*exp(-((w_⟂*vth)^2+(w_∥*vth+u)^2)*me/source_T)
        # Using
        #   d(vth[irowz])/d(ppar[icolz]) = 1/2*vth/ppar * delta(irowz,icolz)
        #
        #   d(exp(-((w_⟂*vth)^2+(w_∥*vth+u)^2)*me/source_T)[irowz])/d(ppar[icolz])
        #     = -2*(w_⟂^2+(w_∥*vth+u)*w_∥)*me/source_T * 1/2*vth/ppar * exp(-((w_⟂*vth)^2+(w_∥*vth+u)^2)*me/source_T) * delta(irowz,icolz)
        #     = -(w_⟂^2+(w_∥*vth+u)*w_∥)*me/source_T * vth/ppar * exp(-((w_⟂*vth)^2+(w_∥*vth+u)^2)*me/source_T) * delta(irowz,icolz)
        jacobian_matrix[row,ppar_offset+iz] +=
            -dt * vth[iz] / dens[iz] * vth_factor * source_amplitude[iz] *
                  (0.5/ppar[iz] - (vperp_grid[ivperp]^2 + (vpa_grid[ivpa]*vth[iz] + upar[iz])*vpa_grid[ivpa])*me/source_T*vth[iz]/ppar[iz]) *
                  exp(-((vperp_grid[ivperp]*vth[iz])^2 + (vpa_grid[ivpa]*vth[iz] + upar[iz])^2) * me / source_T)
    end

    return nothing
end

"""
    external_neutral_source!(pdf, fvec, moments, neutral_source_settings, vzeta, vr,
                            vz, dt)

Add external source term to the neutral kinetic equation.
"""
@timeit global_timer external_neutral_source!(
                         pdf, fvec, moments, neutral_source, index, vzeta, vr, vz,
                         dt) = begin
    begin_sn_r_z_vzeta_vr_region()

    @views source_amplitude = moments.neutral.external_source_amplitude[:, :, index]
    source_T = neutral_source.source_T
    if vzeta.n == 1 && vr.n == 1
        vth_factor = 1.0 / sqrt(source_T)
    else
        vth_factor = 1.0 / source_T^1.5
    end
    vzeta_grid = vzeta.grid
    vr_grid = vr.grid
    vz_grid = vz.grid

    if moments.evolve_ppar && moments.evolve_upar && moments.evolve_density
        vth = moments.neutral.vth
        density = fvec.density_neutral
        uz = fvec.uz_neutral
        @loop_sn_r_z isn ir iz begin
            this_vth = vth[iz,ir,isn]
            this_uz = uz[iz,ir,isn]
            this_prefactor = dt * this_vth / density[iz,ir,isn] * vth_factor *
                             source_amplitude[iz,ir]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vzeta_unnorm = vzeta_grid[ivzeta] * this_vth
                vr_unnorm = vr_grid[ivr] * this_vth
                vz_unnorm = vz_grid[ivz] * this_vth + this_uz
                pdf[ivz,ivr,ivzeta,iz,ir,isn] +=
                    this_prefactor *
                    exp(-(vzeta_unnorm^2 + vr_unnorm^2 + vz_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_upar && moments.evolve_density
        density = fvec.density_neutral
        uz = fvec.uz_neutral
        @loop_sn_r_z isn ir iz begin
            this_uz = uz[iz,ir,isn]
            this_prefactor = dt / density[iz,ir,isn] * vth_factor * source_amplitude[iz,ir]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                vz_unnorm = vz_grid[ivz] + this_uz
                pdf[ivz,ivr,ivzeta,iz,ir,isn] +=
                    this_prefactor *
                    exp(-(vzeta_grid[ivzeta]^2 + vr_grid[ivr]^2 + vz_unnorm^2) / source_T)
            end
        end
    elseif moments.evolve_density
        density = fvec.density_neutral
        @loop_sn_r_z isn ir iz begin
            this_prefactor = dt / density[iz,ir,isn] * vth_factor * source_amplitude[iz,ir]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                pdf[ivz,ivr,ivzeta,iz,ir,isn] +=
                    this_prefactor *
                    exp(-(vzeta_grid[ivzeta]^2 + vr_grid[ivr]^2 + vz_grid[ivz]^2) / source_T)
            end
        end
    elseif !moments.evolve_ppar && !moments.evolve_upar && !moments.evolve_density
        @loop_sn_r_z isn ir iz begin
            this_prefactor = dt * vth_factor * source_amplitude[iz,ir]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                # Factor of 1/sqrt(π) (for 1V) or 1/π^(3/2) (for 2V/3V) is absorbed by the
                # normalisation of F
                pdf[ivz,ivr,ivzeta,iz,ir,isn] +=
                    this_prefactor *
                    exp(-(vzeta_grid[ivzeta]^2 + vr_grid[ivr]^2 + vz_grid[ivz]^2) / source_T)
            end
        end
    else
        error("Unsupported combination evolve_density=$(moments.evolve_density), "
              * "evolve_upar=$(moments.evolve_upar), evolve_ppar=$(moments.evolve_ppar)")
    end


    if neutral_source.source_type == "energy"
        # Take particles out of pdf so source does not change density
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            pdf[iveta,ivr,ivz,iz,ir,isn] -= dt * source_amplitude[iz,ir] *
                fvec.pdf_neutral[iveta,ivr,ivz,iz,ir,isn]
        end
    end

    return nothing
end

"""
    total_external_ion_source_controllers!(fvec_in, moments, ion_sources, dt)

Contribute all of the ion source controllers to fvec_in, one by one.
"""
function total_external_ion_source_controllers!(fvec_in, moments, ion_sources, dt)
    for index ∈ eachindex(ion_sources)
        if ion_sources[index].active
            external_ion_source_controller!(fvec_in, moments, ion_sources[index], index, dt)
        end
    end
    return nothing
end

"""
    external_ion_source_controller!(fvec_in, moments, ion_source_settings, dt)

Calculate the amplitude when using a PI controller for the density to set the external
source amplitude.
"""
@timeit global_timer external_ion_source_controller!(
                         fvec_in, moments, ion_source_settings, index, dt) = begin
    begin_r_z_region()

    is = 1
    ion_moments = moments.ion
    density = fvec_in.density
    upar = fvec_in.upar
    ppar = fvec_in.ppar

    if ion_source_settings.source_type == "Maxwellian"
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                ion_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * ion_source_settings.source_T + upar[iz,ir,is]^2) *
                    ion_moments.external_source_amplitude[iz,ir,index]
            end
        end
    elseif ion_source_settings.source_type == "energy"
        if moments.evolve_upar
            @loop_r_z ir iz begin
                ion_moments.external_source_momentum_amplitude[iz,ir,index] =
                    - density[iz,ir] * upar[iz,ir] *
                      ion_source_settings.source_strength *
                      ion_source_settings.r_amplitude[ir] *
                      ion_source_settings.z_amplitude[iz]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                ion_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * ion_source_settings.source_T + upar[iz,ir]^2 - ppar[iz,ir]) *
                    ion_source_settings.source_strength *
                    ion_source_settings.r_amplitude[ir] *
                    ion_source_settings.z_amplitude[iz]
            end
        end
    elseif ion_source_settings.source_type == "density_midpoint_control"
        begin_serial_region()

        # controller_amplitude error is a shared memory Vector of length 1
        controller_amplitude = ion_source_settings.PI_controller_amplitude
        @serial_region begin
            if ion_source_settings.PI_density_target_ir !== nothing &&
                    ion_source_settings.PI_density_target_iz !== nothing
                # This process has the target point

                n_mid = density[ion_source_settings.PI_density_target_iz,
                                ion_source_settings.PI_density_target_ir, is]
                n_error = ion_source_settings.PI_density_target - n_mid

                ion_moments.external_source_controller_integral[1,1,index] +=
                    dt * ion_source_settings.PI_density_controller_I * n_error

                # Only want a source, so never allow amplitude to be negative
                amplitude = max(
                    ion_source_settings.PI_density_controller_P * n_error +
                    ion_moments.external_source_controller_integral[1,1,index],
                    0)
            else
                amplitude = nothing
            end
            controller_amplitude[1] =
                MPI.Bcast(amplitude, ion_source_settings.PI_density_target_rank,
                          comm_inter_block[])
        end

        begin_r_z_region()

        amplitude = controller_amplitude[1]
        @loop_r_z ir iz begin
            ion_moments.external_source_amplitude[iz,ir,index] =
                amplitude * ion_source_settings.controller_source_profile[iz,ir]
        end
        if moments.evolve_density
            @loop_r_z ir iz begin
                ion_moments.external_source_density_amplitude[iz,ir,index] =
                    amplitude * ion_source_settings.controller_source_profile[iz,ir]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                ion_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * ion_source_settings.source_T + upar[iz,ir,is]^2) *
                    amplitude * ion_source_settings.controller_source_profile[iz,ir]
            end
        end
    elseif ion_source_settings.source_type == "density_profile_control"
        begin_r_z_region()

        target = ion_source_settings.PI_density_target
        P = ion_source_settings.PI_density_controller_P
        I = ion_source_settings.PI_density_controller_I
        integral = ion_moments.external_source_controller_integral
        amplitude = ion_moments.external_source_amplitude
        @loop_r_z ir iz begin
            n_error = target[iz,ir] - density[iz,ir,is]
            integral[iz,ir,index] += dt * I * n_error
            # Only want a source, so never allow amplitude to be negative
            amplitude[iz,ir,index] = max(P * n_error + integral[iz,ir,index], 0)
        end
        if moments.evolve_density
            @loop_r_z ir iz begin
                ion_moments.external_source_density_amplitude[iz,ir,index] = amplitude[iz,ir,index]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                ion_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * ion_source_settings.source_T + upar[iz,ir,is]^2) *
                    amplitude[iz,ir,index]
            end
        end
    elseif ion_source_settings.source_type == "alphas"
        # do nothing
    elseif ion_source_settings.source_type == "alphas-with-losses"
        # do nothing
    elseif ion_source_settings.source_type == "beam"
        # do nothing
    elseif ion_source_settings.source_type == "beam-with-losses"
        # do nothing
    else
        error("Unrecognised source_type=$(ion_source_settings.source_type)")
    end

    return nothing
end

"""
    total_external_electron_source_controllers!(fvec_in, moments, electron_sources, dt)

Contribute all of the electron source controllers to fvec_in, one by one.
"""
function total_external_electron_source_controllers!(fvec_in, moments, electron_sources, dt)
    for index ∈ eachindex(electron_sources)
        if electron_sources[index].active
            external_electron_source_controller!(fvec_in, moments, electron_sources[index], index, dt)
        end
    end
    return nothing
end

"""
    external_electron_source_controller!(fvec_in, moments, electron_source_settings, dt)

Calculate the amplitude, e.g. when using a PI controller for the density to set the
external source amplitude.

As the electron density source must be equal to the ion density source in order not to
inject charge into the simulation, the electron source (at least in some modes of
operation) depends on the ion source, so [`external_ion_source_controller!`](@ref) must be
called before this function is called so that `moments.ion.external_source_amplitude` is
up to date.
"""
@timeit global_timer external_electron_source_controller!(
                         fvec_in, moments, electron_source_settings, index, dt) = begin
    begin_r_z_region()

    is = 1
    electron_moments = moments.electron
    @views ion_source_amplitude = moments.ion.external_source_amplitude[:, :, index]

    if electron_source_settings.source_type == "Maxwellian"
        @loop_r_z ir iz begin
            electron_moments.external_source_pressure_amplitude[iz,ir,index] =
                (0.5 * electron_source_settings.source_T +
                 fvec_in.electron_upar[iz,ir,is]^2) *
                electron_moments.external_source_amplitude[iz,ir,index]
        end
    elseif electron_source_settings.source_type == "energy"
        @loop_r_z ir iz begin
            electron_moments.external_source_momentum_amplitude[iz,ir,index] =
                - electron_moments.density[iz,ir] * electron_moments.upar[iz,ir] *
                  electron_source_settings.source_strength *
                  electron_source_settings.r_amplitude[ir] *
                  electron_source_settings.z_amplitude[iz]
        end
        @loop_r_z ir iz begin
            electron_moments.external_source_pressure_amplitude[iz,ir,index] =
                (0.5 * electron_source_settings.source_T + electron_moments.upar[iz,ir]^2 -
                 electron_moments.ppar[iz,ir]) *
                electron_source_settings.source_strength *
                electron_source_settings.r_amplitude[ir] *
                electron_source_settings.z_amplitude[iz]
        end
    else
        @loop_r_z ir iz begin
            electron_moments.external_source_amplitude[iz,ir,index] = ion_source_amplitude[iz,ir,index]
        end
        @loop_r_z ir iz begin
            electron_moments.external_source_momentum_amplitude[iz,ir,index] =
                - electron_moments.density[iz,ir] * electron_moments.upar[iz,ir] *
                  electron_moments.external_source_amplitude[iz,ir,index]
        end
        @loop_r_z ir iz begin
            electron_moments.external_source_pressure_amplitude[iz,ir,index] =
                (0.5 * electron_source_settings.source_T + electron_moments.upar[iz,ir]^2 -
                 electron_moments.ppar[iz,ir]) *
                electron_moments.external_source_amplitude[iz,ir,index]
        end
    end

    # Density source is always the same as the ion one
    @loop_r_z ir iz begin
        electron_moments.external_source_density_amplitude[iz,ir,index] =
            moments.ion.external_source_density_amplitude[iz,ir,index]
    end

    return nothing
end

"""
    total_external_neutral_source_controllers!(fvec_in, moments, neutral_sources, dt)

Contribute all of the neutral source controllers to fvec_in, one by one.
"""
function total_external_neutral_source_controllers!(fvec_in, moments, neutral_sources, r, z, dt)
    for index ∈ eachindex(neutral_sources)
        if neutral_sources[index].active
            external_neutral_source_controller!(fvec_in, moments, neutral_sources[index], 
                                                index, r, z, dt)
        end
    end
    return nothing
end

"""
    external_neutral_source_controller!(fvec_in, moments, neutral_source_settings, r,
                                        z, dt)

Calculate the amplitude when using a PI controller for the density to set the external
source amplitude.
"""
@timeit global_timer external_neutral_source_controller!(
                         fvec_in, moments, neutral_source_settings, index, r, z,
                         dt) = begin
    begin_r_z_region()

    is = 1
    neutral_moments = moments.neutral

    if neutral_source_settings.source_type == "Maxwellian"
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                neutral_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * neutral_source_settings.source_T + fvec_in.upar[iz,ir,is]^2) *
                    neutral_moments.external_source_amplitude[iz,ir,index]
            end
        end
    elseif neutral_source_settings.source_type == "energy"
        if moments.evolve_upar
            @loop_r_z ir iz begin
                neutral_moments.external_source_momentum_amplitude[iz,ir,index] =
                    - neutral_moments.density[iz,ir] * neutral_moments.uz[iz,ir] *
                      neutral_source_settings.source_strength *
                      neutral_source_settings.r_amplitude[ir] *
                      neutral_source_settings.z_amplitude[iz]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                neutral_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * neutral_source_settings.source_T +
                     neutral_moments.uz[iz,ir]^2 - neutral_moments.pz[iz,ir]) *
                    neutral_source_settings.source_strength *
                    neutral_source_settings.r_amplitude[ir] *
                    neutral_source_settings.z_amplitude[iz]
            end
        end
    elseif neutral_source_settings.source_type == "density_midpoint_control"
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

                neutral_moments.external_source_controller_integral[1,1,index] +=
                    dt * neutral_source_settings.PI_density_controller_I * n_error

                # Only want a source, so never allow amplitude to be negative
                amplitude = max(
                    neutral_source_settings.PI_density_controller_P * n_error +
                    neutral_moments.external_source_controller_integral[1,1,index],
                    0)
            else
                amplitude = nothing
            end
            controller_amplitude[1] =
                MPI.Bcast(amplitude, neutral_source_settings.PI_density_target_rank,
                          comm_inter_block[])
        end

        begin_r_z_region()

        amplitude = controller_amplitude[1]
        @loop_r_z ir iz begin
            neutral_moments.external_source_amplitude[iz,ir,index] =
                amplitude * neutral_source_settings.controller_source_profile[iz,ir,index]
        end
        if moments.evolve_density
            @loop_r_z ir iz begin
                neutral_moments.external_source_density_amplitude[iz,ir,index] =
                    amplitude * neutral_source_settings.controller_source_profile[iz,ir,index]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                neutral_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * neutral_source_settings.source_T + fvec_in.upar[iz,ir,is]^2) *
                    amplitude * neutral_source_settings.controller_source_profile[iz,ir,index]
            end
        end
    elseif neutral_source_settings.source_type == "density_profile_control"
        begin_r_z_region()

        density = fvec_in.density_neutral
        target = neutral_source_settings.PI_density_target
        P = neutral_source_settings.PI_density_controller_P
        I = neutral_source_settings.PI_density_controller_I
        PI_integral = neutral_moments.external_source_controller_integral
        amplitude = neutral_moments.external_source_amplitude
        @loop_r_z ir iz begin
            n_error = target[iz,ir] - density[iz,ir,is]
            PI_integral[iz,ir,index] += dt * I * n_error
            amplitude[iz,ir,index] = P * n_error + PI_integral[iz,ir,index]
        end
        if moments.evolve_density
            @loop_r_z ir iz begin
                neutral_moments.external_source_density_amplitude[iz,ir,index] = amplitude[iz,ir,index]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                neutral_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * neutral_source_settings.source_T + fvec_in.upar[iz,ir,is]^2) *
                    amplitude[iz,ir,index]
            end
        end
    elseif neutral_source_settings.source_type == "recycling"
        begin_serial_region()
        target_flux = 0.0
        @boundscheck size(fvec_in.density, 3) == 1
        @boundscheck size(fvec_in.density_neutral, 3) == 1
        is = 1

        # Warning: this target flux is only correct when the magnetic field is
        # perpendicular to the targets. Would be better to define (somewhere) a function
        # that calculates the 'target ion flux' for general conditions...
        # First get target_flux on rank-0 of each shared memory block
        @serial_region begin
            if z.irank == 0
                # Target flux from lower wall
                @views @. r.scratch = -fvec_in.density[1,:,is] * fvec_in.upar[1,:,is]
                target_flux += integral(r.scratch, r.wgts)
            end
            if z.irank == z.nrank - 1
                # Target flux from upper wall
                @views @. r.scratch = fvec_in.density[end,:,is] * fvec_in.upar[end,:,is]
                target_flux += integral(r.scratch, r.wgts)
            end
            target_flux = MPI.Allreduce(target_flux, +, comm_inter_block[])
        end

        # Distribute target_flux to all processes in shared memory block
        target_flux = MPI.Bcast(target_flux, 0, comm_block[])

        # No need to synchronize as MPI.Bcast() synchronized already
        begin_r_z_region(no_synchronize=true)

        amplitude = neutral_moments.external_source_amplitude
        profile = neutral_source_settings.controller_source_profile
        prefactor = target_flux * neutral_source_settings.recycling_controller_fraction
        @loop_r_z ir iz begin
            amplitude[iz,ir,index] = prefactor * profile[iz,ir]
        end
        if moments.evolve_density
            @loop_r_z ir iz begin
                neutral_moments.external_source_density_amplitude[iz,ir,index] = amplitude[iz,ir,index]
            end
        end
        if moments.evolve_ppar
            @loop_r_z ir iz begin
                neutral_moments.external_source_pressure_amplitude[iz,ir,index] =
                    (0.5 * neutral_source_settings.source_T + fvec_in.upar[iz,ir,is]^2) *
                    amplitude[iz,ir,index]
            end
        end
    else
        error("Unrecognised source_type=$(neutral_source_settings.source_type)")
    end

    return nothing
end

end
