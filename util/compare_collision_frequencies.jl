using moment_kinetics
using Plots

function compare_collision_frequencies(input_file::String, output_file::String; Bnorm=1,
                                       Lnorm=10, Nnorm=1.e19, Tnorm=100)

    input = moment_kinetics.moment_kinetics_input.input_from_TOML(input_file)
    run_name, output_dir, evolve_moments, t_input, z_input, r_input, vpa_input,
    composition, species_input, collisions, drive_input, num_diss_params =
    moment_kinetics.moment_kinetics_input.mk_input(input)

    dimensional_parameters = moment_kinetics.utils.get_unnormalized_parameters(
        input_file; Bnorm=Bnorm, Lnorm=Lnorm, Nnorm=Nnorm, Tnorm=Tnorm)

    fid = moment_kinetics.load_data.open_netcdf_file(splitext(output_file)[1])

    z, z_spectral = moment_kinetics.load_data.load_coordinate_data(fid, "z")

    density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed, T_e,
    n_species, evolve_density, evolve_upar, evolve_ppar =
    moment_kinetics.load_data.load_moments_data(fid)

    parallel_temperature = parallel_pressure ./ density

    println("ionization rate coefficient = ",
            dimensional_parameters["ionization_rate_coefficient"])
    println("charge_exchange rate coefficient = ",
            dimensional_parameters["CX_rate_coefficient"])

    # Ignoring variations in logLambda...
    nu_ii = @. dimensional_parameters["nu_ii0"] * density / parallel_temperature^1.5
    println("nu_ii ", nu_ii[z.n÷2,1,1,end])

    # Effective collision frequency for dissipation?
    # v_∥ dissipation term is D d^2f/dv_∥^2. Inserting factors of c_ref, this is a bit like
    # pitch angle scattering D cref^2 d^2f/dv_∥^2 ~ D d^2f/dξ^2, so D is similar to a
    # (normalised) collision frequency.
    nu_diss = num_diss_params.vpa_dissipation_coefficient /
              dimensional_parameters["timenorm"]
    println("nu_diss ", nu_diss)

    # Neutral collison rates:
    # The ionization term in the ion/neutral kinetic equations is ±R_ion*n_e*f_n.
    # R_ion*n_e is an 'ionization rate' that just needs unnormalising - it gives the
    # (inverse of the) characteristic time that it takes a neutral atom to be ionized.
    nu_ionization = @. collisions.ionization * density[:,:,1,:] /
                       dimensional_parameters["timenorm"]
    println("nu_ionization ", nu_ionization[z.n÷2,1,end])
    # The charge-exchange term in the ion kinetic equation is -R_in*(n_n*f_i-n_i*f_n).
    # So the rate at which ions experience CX reactions is R_in*n_n
    nu_cx = @. collisions.charge_exchange * density[:,:,composition.n_ion_species+1,:] /
               dimensional_parameters["timenorm"]
    println("nu_cx ", nu_cx[z.n÷2,1,end])

    # Make plot (using values from the final time point)
    plot(legend=:outerright, xlabel="z", ylabel="frequency", ylims=(0.0, :auto))
    @views plot!(z.grid, nu_ii[:,1,1,end], label="nu_ii")
    @views plot!(z.grid, nu_ionization[:,1,end], label="nu_ionization")
    @views plot!(z.grid, nu_cx[:,1,end], label="nu_cx")
    hline!([nu_diss], label="nu_diss")
    ylabel!("frequency (s^-1)")

    savefig(joinpath("runs", run_name, run_name * "_collision_frequencies.pdf"))

    return nothing
end
