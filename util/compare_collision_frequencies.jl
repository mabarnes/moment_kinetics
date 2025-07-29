using moment_kinetics
using makie_post_processing
using makie_post_processing: CairoMakie
using .CairoMakie
using Unitful

function compare_collision_frequencies(input_file::String,
                                       output_file::Union{String,Nothing}=nothing)

    input = moment_kinetics.moment_kinetics_input.read_input_file(input_file)

    io_input, evolve_moments, t_input, z, z_spectral, r, r_spectral, vpa, vpa_spectral,
    vperp, vperp_spectral, gyrophase, gyrophase_spectral, vz, vz_spectral, vr,
    vr_spectral, vzeta, vzeta_spectral, composition, species, collisions, geometry,
    em_input, num_diss_params, manufactured_solns_input =
        moment_kinetics.moment_kinetics_input.mk_input(input)

    dimensional_parameters =
        moment_kinetics.utils.get_unnormalized_parameters(input, true)

    println("Omega_i0 ", dimensional_parameters["Omega_i0"])
    println("rho_i0 ", dimensional_parameters["rho_i0"])
    println("Omega_e0 ", dimensional_parameters["Omega_e0"])
    println("rho_e0 ", dimensional_parameters["rho_e0"])

    # Characteristic rate for parallel streaming
    println("vth_i0/Lz ", dimensional_parameters["vthi0"] / dimensional_parameters["Lz"])

    # Effective collision frequency for dissipation?
    # v_∥ dissipation term is D d^2f/dv_∥^2. Inserting factors of c_ref, this is a bit like
    # pitch angle scattering D cref^2 d^2f/dv_∥^2 ~ D d^2f/dξ^2, so D is similar to a
    # (normalised) collision frequency.
    if num_diss_params.ion.vpa_dissipation_coefficient < 0.0
        nu_vpa_diss = 0.0 / Unitful.s
    else
        nu_vpa_diss = num_diss_params.ion.vpa_dissipation_coefficient /
                      dimensional_parameters["timenorm"]
    end

    println("ionization rate coefficient = ",
            dimensional_parameters["ionization_rate_coefficient"])
    println("charge_exchange rate coefficient = ",
            dimensional_parameters["CX_rate_coefficient"])

    println("nu_ei0 ", dimensional_parameters["coulomb_collision_frequency_ei0"])
    println("nu_ii0 ", dimensional_parameters["coulomb_collision_frequency_ii0"])
    println("nu_ie0 ", dimensional_parameters["coulomb_collision_frequency_ie0"])
    println("nu_vpa_diss ", nu_vpa_diss)

    # Neutral collison rates:
    # The ionization term in the ion/neutral kinetic equations is ±R_ion*n_e*f_n.
    # R_ion*n_e is an 'ionization rate' that just needs unnormalising - it gives the
    # (inverse of the) characteristic time that it takes a neutral atom to be ionized.
    nu_ionization0 = @. collisions.reactions.ionization_frequency *
                        dimensional_parameters["Nnorm"] /
                        dimensional_parameters["timenorm"]
    println("nu_ionization0 ", nu_ionization0)
    # The charge-exchange term in the ion kinetic equation is -R_in*(n_n*f_i-n_i*f_n).
    # So the rate at which ions experience CX reactions is R_in*n_n
    nu_cx0 = @. collisions.reactions.charge_exchange_frequency *
                dimensional_parameters["Nnorm"] / dimensional_parameters["timenorm"]
    println("nu_cx0 ", nu_cx0)

    # Estimate classical particle and ion heat diffusion coefficients, for comparison to
    # numerical dissipation
    # Classical particle diffusivity estimate from Helander, D_⟂ on p.7.
    # D_⟂ ∼ nu_ei * rho_e^2 / 2
    classical_particle_D0 = Unitful.upreferred(dimensional_parameters["coulomb_collision_frequency_ei0"] *
                                               dimensional_parameters["rho_e0"]^2 / 2.0)
    println("classical_particle_D0 ", classical_particle_D0)

    # Classical thermal diffusivity estimate from Helander, eq. (1.8)
    # chi_i = rho_i^2 / tau_ii / 2 = nu_ii * rho_i^2 / 2
    classical_heat_chi_i0 = Unitful.upreferred(dimensional_parameters["coulomb_collision_frequency_ii0"] *
                                              dimensional_parameters["rho_i0"]^2 / 2.0)
    println("classical_heat_chi_i0 ", classical_heat_chi_i0)

    # rhostar is set as an input parameter. For the purposes of cross field transport,
    # effective rho_i is rhostar*R rather than rho_i0. Might as well take R∼Lnorm for now,
    # as difference will be O(1), if any.
    rho_i_effective = Unitful.upreferred(geometry.rhostar *
                                         dimensional_parameters["Lnorm"])
    rho_e_effective =
        Unitful.upreferred(sqrt(dimensional_parameters["m_e"]/dimensional_parameters["m_i"])
                           * rho_i_effective)
    effective_classical_particle_D0 =
        Unitful.upreferred(dimensional_parameters["coulomb_collision_frequency_ei0"] *
                           rho_e_effective^2 / 2.0)
    effective_classical_heat_chi_i0 =
        Unitful.upreferred(dimensional_parameters["coulomb_collision_frequency_ii0"] *
                           rho_i_effective^2 / 2.0)
    println("rho_i_effective ", rho_i_effective)
    println("rho_e_effective ", rho_e_effective)
    println("classical particle D0 with effective rho_e ", effective_classical_particle_D0)
    println("classical heat chi_i0 with effective rho_i ", effective_classical_heat_chi_i0)

    # Get numerical diffusion parameters
    if num_diss_params.ion.r_dissipation_coefficient < 0.0
        D_r = 0.0
    else
        D_r = Unitful.upreferred(num_diss_params.ion.r_dissipation_coefficient *
                                 dimensional_parameters["Lnorm"]^2 /
                                 dimensional_parameters["timenorm"])
    end
    if num_diss_params.ion.z_dissipation_coefficient < 0.0
        D_z = 0.0
    else
        D_z = Unitful.upreferred(num_diss_params.ion.z_dissipation_coefficient *
                                 dimensional_parameters["Lnorm"]^2 /
                                 dimensional_parameters["timenorm"])
    end
    println("numerical D_r ", D_r)
    println("numerical D_z ", D_z)

    if output_file !== nothing
        println("")

        run_info = get_run_info(output_file)
        z = run_info.z
        density = get_variable(run_info, "density")
        parallel_pressure = get_variable(run_info, "parallel_pressure")

        has_neutrals = run_info.n_neutral_species > 0
        if has_neutrals
            neutral_density = get_variable(run_info, "density_neutral")
        end

        parallel_temperature = parallel_pressure ./ density
        parallel_thermal_speed = @. sqrt(2.0 * parallel_temperature)

        # Ignoring variations in logLambda...
        nu_ii = @. dimensional_parameters["coulomb_collision_frequency_ii0"] * density / parallel_thermal_speed^3
        println("nu_ii ", nu_ii[z.n_global÷2,1,1,end])

        # Make plot (using values from the final time point)
        fig, ax, legend_place = get_1d_ax(; get_legend_place=:right, xlabel="z",
                                          ylabel="frequency (s^-1)")

        # Makie.jl seems to get confused with units of s^-1, so just strip units when
        # plotting.
        @views lines!(ax, z.grid, Unitful.ustrip(nu_ii[:,1,1,end]); label="nu_ii")

        if has_neutrals
            # Neutral collison rates:
            # The ionization term in the ion/neutral kinetic equations is ±R_ion*n_e*f_n.
            # R_ion*n_e is an 'ionization rate' that just needs unnormalising - it gives the
            # (inverse of the) characteristic time that it takes a neutral atom to be ionized.
            nu_ionization = @. collisions.ionization * density[:,:,1,:] /
                               dimensional_parameters["timenorm"]
            println("nu_ionization ", nu_ionization[z.n_global÷2,1,end])
            # The charge-exchange term in the ion kinetic equation is -R_in*(n_n*f_i-n_i*f_n).
            # So the rate at which ions experience CX reactions is R_in*n_n
            nu_cx = @. collisions.charge_exchange * neutral_density[:,:,1,:] /
                       dimensional_parameters["timenorm"]
            println("nu_cx ", nu_cx[z.n_global÷2,1,end])

            @views lines!(ax, z.grid, Unitful.ustrip(nu_ionization[:,1,end]); label="nu_ionization")
            @views lines!(ax, z.grid, Unitful.ustrip(nu_cx[:,1,end]); label="nu_cx")
        end

        # Using hlines!() would result in taking the colour from a different colour cycler
        # than lines! uses, so we would get a line with the same colour as an existing
        # one.
        #hlines!(ax, [Unitful.ustrip(nu_vpa_diss)], label="nu_vpa_diss")
        lines!(ax, z.grid, fill(Unitful.ustrip(nu_vpa_diss), z.n), label="nu_vpa_diss")

        ylims!(ax, 0.0, nothing)

        Legend(legend_place, ax)

        save(joinpath(io_input.output_dir,
                      io_input.run_name * "_collision_frequencies.pdf"),
             fig)
    end

    return nothing
end
