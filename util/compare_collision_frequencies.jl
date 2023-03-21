using moment_kinetics
using Plots

function compare_collision_frequencies(input_file::String,
                                       output_file::Union{String,Nothing}=nothing;
                                       Bnorm=1, Lnorm=10, Nnorm=1.e19, Tnorm=100)

    input = moment_kinetics.moment_kinetics_input.read_input_file(input_file)
    io_input, evolve_moments, t_input, z_input, r_input, vpa_input, vperp_input,
    gyrophase_input, vz_input, vr_input, vzeta_input, composition, species, collisions,
    geometry, drive_input, num_diss_params =
    moment_kinetics.moment_kinetics_input.mk_input(input)

    dimensional_parameters = moment_kinetics.utils.get_unnormalized_parameters(
        input_file; Bnorm=Bnorm, Lnorm=Lnorm, Nnorm=Nnorm, Tnorm=Tnorm)

    println("Omega_i0 ", dimensional_parameters["Omega_i0"])
    println("rho_i0 ", dimensional_parameters["rho_i0"])
    println("Omega_e0 ", dimensional_parameters["Omega_e0"])
    println("rho_e0 ", dimensional_parameters["rho_e0"])

    # Effective collision frequency for dissipation?
    # v_∥ dissipation term is D d^2f/dv_∥^2. Inserting factors of c_ref, this is a bit like
    # pitch angle scattering D cref^2 d^2f/dv_∥^2 ~ D d^2f/dξ^2, so D is similar to a
    # (normalised) collision frequency.
    if num_diss_params.vpa_dissipation_coefficient < 0.0
        nu_vpa_diss = 0.0
    else
        nu_vpa_diss = num_diss_params.vpa_dissipation_coefficient /
                      dimensional_parameters["timenorm"]
    end

    println("ionization rate coefficient = ",
            dimensional_parameters["ionization_rate_coefficient"])
    println("charge_exchange rate coefficient = ",
            dimensional_parameters["CX_rate_coefficient"])

    println("nu_ei0 ", dimensional_parameters["nu_ei0"])
    println("nu_ii0 ", dimensional_parameters["nu_ii0"])
    println("nu_ie0 ", dimensional_parameters["nu_ie0"])
    println("nu_vpa_diss ", nu_vpa_diss)

    # Neutral collison rates:
    # The ionization term in the ion/neutral kinetic equations is ±R_ion*n_e*f_n.
    # R_ion*n_e is an 'ionization rate' that just needs unnormalising - it gives the
    # (inverse of the) characteristic time that it takes a neutral atom to be ionized.
    nu_ionization0 = @. collisions.ionization * dimensional_parameters["Nnorm"] /
                        dimensional_parameters["timenorm"]
    println("nu_ionization0 ", nu_ionization0)
    # The charge-exchange term in the ion kinetic equation is -R_in*(n_n*f_i-n_i*f_n).
    # So the rate at which ions experience CX reactions is R_in*n_n
    nu_cx0 = @. collisions.charge_exchange * dimensional_parameters["Nnorm"] /
                dimensional_parameters["timenorm"]
    println("nu_cx0 ", nu_cx0)

    if output_file !== nothing
        println("")

        temp, file_ext = splitext(output_file)
        temp, ext = splitext(temp)
        basename, iblock = splitext(temp)
        iblock = parse(moment_kinetics.type_definitions.mk_int, iblock[2:end])
        fid = moment_kinetics.load_data.open_readonly_output_file(basename, ext[2:end];
                                                                  iblock=iblock)

        nz_local, nz_global, zgrid, z_wgts, Lz =
            moment_kinetics.load_data.load_coordinate_data(fid, "z")

        density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed,
        evolve_ppar = moment_kinetics.load_data.load_charged_particle_moments_data(fid)

        neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed =
        moment_kinetics.load_data.load_neutral_particle_moments_data(fid)

        parallel_temperature = parallel_pressure ./ density

        # Ignoring variations in logLambda...
        nu_ii = @. dimensional_parameters["nu_ii0"] * density / parallel_temperature^1.5
        println("nu_ii ", nu_ii[nz_global÷2,1,1,end])

        # Neutral collison rates:
        # The ionization term in the ion/neutral kinetic equations is ±R_ion*n_e*f_n.
        # R_ion*n_e is an 'ionization rate' that just needs unnormalising - it gives the
        # (inverse of the) characteristic time that it takes a neutral atom to be ionized.
        nu_ionization = @. collisions.ionization * density[:,:,1,:] /
                           dimensional_parameters["timenorm"]
        println("nu_ionization ", nu_ionization[nz_global÷2,1,end])
        # The charge-exchange term in the ion kinetic equation is -R_in*(n_n*f_i-n_i*f_n).
        # So the rate at which ions experience CX reactions is R_in*n_n
        nu_cx = @. collisions.charge_exchange * neutral_density[:,:,1,:] /
                   dimensional_parameters["timenorm"]
        println("nu_cx ", nu_cx[nz_global÷2,1,end])

        # Make plot (using values from the final time point)
        plot(legend=:outerright, xlabel="z", ylabel="frequency", ylims=(0.0, :auto))
        @views plot!(zgrid, nu_ii[:,1,1,end], label="nu_ii")
        @views plot!(zgrid, nu_ionization[:,1,end], label="nu_ionization")
        @views plot!(zgrid, nu_cx[:,1,end], label="nu_cx")
        hline!([nu_vpa_diss], label="nu_vpa_diss")
        ylabel!("frequency (s^-1)")

        savefig(joinpath("runs", io_input.run_name,
                         io_input.run_name * "_collision_frequencies.pdf"))
    end

    return nothing
end
