using moment_kinetics
using Plots
using Unitful

function compare_collision_frequencies(input_file::String,
                                       output_file::Union{String,Nothing}=nothing)

    input = moment_kinetics.moment_kinetics_input.read_input_file(input_file)

    io_input, evolve_moments, t_input, z, z_spectral, r, r_spectral, vpa,
    vpa_spectral, vperp, vperp_spectral, gyrophase, gyrophase_spectral, vz, vz_spectral,
    vr, vr_spectral, vzeta, vzeta_spectral, composition, species, collisions,
    geometry, em_input, external_source_settings, num_diss_params,
    manufactured_solns_input =
        moment_kinetics.moment_kinetics_input.mk_input(input)

    dimensional_parameters =
        moment_kinetics.utils.get_unnormalized_parameters(input, true)

    println("Omega_i0 ", dimensional_parameters["Omega_i0"])
    println("rho_i0 ", dimensional_parameters["rho_i0"])
    println("Omega_e0 ", dimensional_parameters["Omega_e0"])
    println("rho_e0 ", dimensional_parameters["rho_e0"])

    # Effective collision frequency for dissipation?
    # v_∥ dissipation term is D d^2f/dv_∥^2. Inserting factors of c_ref, this is a bit like
    # pitch angle scattering D cref^2 d^2f/dv_∥^2 ~ D d^2f/dξ^2, so D is similar to a
    # (normalised) collision frequency.
    if num_diss_params.ion.vpa_dissipation_coefficient < 0.0
        nu_vpa_diss = 0.0
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

        temp, file_ext = splitext(output_file)
        temp, ext = splitext(temp)
        basename, iblock = splitext(temp)
        iblock = parse(moment_kinetics.type_definitions.mk_int, iblock[2:end])
        fid = moment_kinetics.load_data.open_readonly_output_file(basename, ext[2:end];
                                                                  iblock=iblock)

        z = moment_kinetics.load_data.load_coordinate_data(fid, "z")

        density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed,
        evolve_ppar = moment_kinetics.load_data.load_charged_particle_moments_data(fid)

        neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed =
        moment_kinetics.load_data.load_neutral_particle_moments_data(fid)

        parallel_temperature = parallel_pressure ./ density

        # Ignoring variations in logLambda...
        nu_ii = @. dimensional_parameters["nu_ii0"] * density / parallel_temperature^1.5
        println("nu_ii ", nu_ii[z.n_global÷2,1,1,end])

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

        # Make plot (using values from the final time point)
        plot(legend=:outerright, xlabel="z", ylabel="frequency", ylims=(0.0, :auto))
        @views plot!(z.grid, nu_ii[:,1,1,end], label="nu_ii")
        @views plot!(z.grid, nu_ionization[:,1,end], label="nu_ionization")
        @views plot!(z.grid, nu_cx[:,1,end], label="nu_cx")
        hline!([nu_vpa_diss], label="nu_vpa_diss")
        ylabel!("frequency (s^-1)")

        savefig(joinpath(io_input.output_dir,
                         io_input.run_name * "_collision_frequencies.pdf"))
    end

    return nothing
end
