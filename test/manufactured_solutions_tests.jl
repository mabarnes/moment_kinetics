"""
Test cases using the method of manufactured solutions (MMS)
"""
module ManufacturedSolutionsTests

include("setup.jl")
include("mms_utils.jl")

using moment_kinetics.post_processing: L2_error_norm, L_infinity_error_norm
using moment_kinetics.manufactured_solns
using moment_kinetics.type_definitions

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

const input_sound_wave_periodic = Dict(
    "use_manufactured_solns" => true,
    "n_ion_species" => 1,
    "n_neutral_species" => 0,
    "boltzmann_electron_response" => true,
    "run_name" => "MMS-rperiodic",
    "base_directory" => test_output_directory,
    "evolve_moments_density" => false,
    "evolve_moments_parallel_flow" => false,
    "evolve_moments_parallel_pressure" => false,
    "evolve_moments_conservation" => false,
    "T_e" => 1.0,
    "rhostar" => 1.0,
    "initial_density1" => 0.5,
    "initial_temperature1" => 1.0,
    "initial_density2" => 0.5,
    "initial_temperature2" => 1.0,
    "z_IC_option1" => "sinusoid",
    "z_IC_density_amplitude1" => 0.001,
    "z_IC_density_phase1" => 0.0,
    "z_IC_upar_amplitude1" => 0.0,
    "z_IC_upar_phase1" => 0.0,
    "z_IC_temperature_amplitude1" => 0.0,
    "z_IC_temperature_phase1" => 0.0,
    "z_IC_option2" => "sinusoid",
    "z_IC_density_amplitude2" => 0.001,
    "z_IC_density_phase2" => 0.0,
    "z_IC_upar_amplitude2" => 0.0,
    "z_IC_upar_phase2" => 0.0,
    "z_IC_temperature_amplitude2" => 0.0,
    "z_IC_temperature_phase2" => 0.0,
    "charge_exchange_frequency" => 0.62831853071,
    "ionization_frequency" => 0.0,
    #"nstep" => 10, #1700,
    #"dt" => 0.002,
    #"nwrite" => 10, #1700,
    "nstep" => 1700, #1700,
    "dt" => 0.0002, #0.002,
    "nwrite" => 1700, #1700,
    "use_semi_lagrange" => false,
    "n_rk_stages" => 4,
    "split_operators" => false,
    "z_ngrid" => 4,
    "z_nelement" => 2,
    "z_bc" => "periodic",
    "z_discretization" => "chebyshev_pseudospectral",
    "r_ngrid" => 4,
    "r_nelement" => 2,
    "r_bc" => "periodic",
    "r_discretization" => "chebyshev_pseudospectral",
    "vpa_ngrid" => 4,
    "vpa_nelement" => 4,
    "vpa_L" => 8.0,
    "vpa_bc" => "periodic",
    "vpa_discretization" => "chebyshev_pseudospectral",
    "vperp_ngrid" => 4,
    "vperp_nelement" => 4,
    "vperp_L" => 8.0,
    "vperp_bc" => "periodic",
    "vperp_discretization" => "chebyshev_pseudospectral",
)

"""
    runcase(input::Dict)

Run a simulation with parameters set by `input` using manufactured sources and return
the errors in each variable compared to the manufactured solution.
"""
function runcase(input::Dict)
    in_manf, phi_manf, f_manf = nothing, nothing, nothing
    # call setup_moment_kinetics(), time_advance!(), cleanup_moment_kinetics!()
    # separately so we can run manufactured_solutions_as_arrays() in parallel
    quietoutput() do
        # run simulation
        pdf, vz, vr, vzeta, vpa, vperp, z, r, spectral_objects, composition,
        drive_input, moments, t_input, collisions, species, geometry,
        boundary_distributions = setup_moment_kinetics(input_dict)
        time_advance!(pdf, vz, vr, vzeta, vpa, vperp, z, r, spectral_objects,
                      composition, drive_input, moments, t_input, collisions,
                      species, geometry, boundary_distributions)

        n_manf, phi_manf, f_manf =
            manufactured_solutions_as_arrays(t, r, z, vperp, vpa)

        if global_rank[] == 0
            # Need to copy as cleanup_moment_kinetics!() will invalidate
            # shared-memory arrays
            n_manf = copy(n_manf)
            phi_manf = copy(phi_manf)
            f_manf = copy(f_manf)
        end

        cleanup_moment_kinetics!(io, cdf)
    end

    n_error_2 = nothing
    n_error_inf = nothing
    phi_error_2 = nothing
    phi_error_inf = nothing
    f_error_2 = nothing
    f_error_inf = nothing
    if global_rank[] == 0
        output = load_test_output(input, (:phi, :moments, :f))

        t = output["time"][end]
        n = output["density"][:,:,1,end]
        phi = output["phi"][:,:,end]
        f = output["f"][:,:,:,:,1,end]
        f0 = f[size(f,1)÷2, 1, :, :]

        f0_manf = f_manf[size(f,1)÷2, 1, :, :]

        n_error_2 = L2_error_norm(n, n_manf)
        n_error_inf = L_infinity_error_norm(n, n_manf)

        phi_error_2 = L2_error_norm(phi, phi_manf)
        phi_error_inf = L_infinity_error_norm(phi, phi_manf)

        f_error_2 = L2_error_norm(f, f_manf)
        f_error_inf = L_infinity_error_norm(f, f_manf)

        f0_error_2 = L2_error_norm(f0, f0_manf)
        f0_error_inf = L_infinity_error_norm(f0, f0_manf)

        println("n ", n_error_2, " ", n_error_inf)
        println("phi ", phi_error_2, " ", phi_error_inf)
        println("f ", f_error_2, " ", f_error_inf)
        println("f0 ", f0_error_2, " ", f0_error_inf)
    end

    return n_error_2, n_error_inf, phi_error_2, phi_error_inf, f_error_2, f_error_inf
end

"""
    testconvergence(input::Dict)

Test convergence with spatial resolution

The parameters for the run are given in `input::Dict`.
"""
function testconvergence(input::Dict)
    n_errors_2 = Vector{mk_float}(undef, 0)
    n_errors_inf = Vector{mk_float}(undef, 0)
    phi_errors_2 = Vector{mk_float}(undef, 0)
    phi_errors_inf = Vector{mk_float}(undef, 0)
    f_errors_2 = Vector{mk_float}(undef, 0)
    f_errors_inf = Vector{mk_float}(undef, 0)

    ngrid = get_and_check_ngrid(input)

    nelement_values = [2, 3, 4]
    for nelement ∈ nelement_values
        global_rank[] == 0 && println("testing nelement=$nelement")
        case_input = increase_resolution(input, nelement)

        n_error_2, n_error_inf, phi_error_2, phi_error_inf, f_error_2,
        f_error_inf = runcase(case_input)

        if global_rank[] == 0
            push!(n_errors_2, n_error_2)
            push!(n_errors_inf, n_error_inf)
            push!(phi_errors_2, phi_error_2)
            push!(phi_errors_inf, phi_error_inf)
            push!(f_errors_2, f_error_2)
            push!(f_errors_inf, f_error_inf)
        end
    end

    if global_rank[] == 0
        n_convergence_2 = n_errors_2[1] ./ n_errors_2[2:end]
        n_convergence_inf = n_errors_inf[1] ./ n_errors_inf[2:end]
        phi_convergence_2 = phi_errors_2[1] ./ phi_errors_2[2:end]
        phi_convergence_inf = phi_errors_inf[1] ./ phi_errors_inf[2:end]
        f_convergence_2 = f_errors_2[1] ./ f_errors_2[2:end]
        f_convergence_inf = f_errors_inf[1] ./ f_errors_inf[2:end]
        expected_convergence = @. (nelement_values[2:end] / nelement_values[1])^(ngrid - 1)
        println("n convergence")
        println(n_convergence_2)
        println(n_convergence_inf)
        println("phi convergence")
        println(phi_convergence_2)
        println(phi_convergence_inf)
        println("f convergence")
        println(f_convergence_2)
        println(f_convergence_inf)
        println("expected convergence")
        println(expected_convergence)
    end
end

function runtests()
    @testset "MMS" verbose=use_verbose begin
        global_rank[] == 0 && println("MMS tests")

        @testset "r-periodic, z-periodic" begin
            testconvergence(input_sound_wave_periodic)
        end
    end

    return nothing
end

end # ManufacturedSolutionsTests


using .ManufacturedSolutionsTests

ManufacturedSolutionsTests.runtests()
