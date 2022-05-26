"""
Test cases using the method of manufactured solutions (MMS)
"""
module ManufacturedSolutionsTests

include("setup.jl")

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
    "nstep" => 1700,
    "dt" => 0.002,
    "nwrite" => 1700,
    "use_semi_lagrange" => false,
    "n_rk_stages" => 4,
    "split_operators" => false,
    "z_ngrid" => 5,
    "z_nelement" => 2,
    "z_bc" => "periodic",
    "z_discretization" => "chebyshev_pseudospectral",
    "r_ngrid" => 5,
    "r_nelement" => 2,
    "r_bc" => "periodic",
    "r_discretization" => "chebyshev_pseudospectral",
    "vpa_ngrid" => 5,
    "vpa_nelement" => 8,
    "vpa_L" => 8.0,
    "vpa_bc" => "periodic",
    "vpa_discretization" => "chebyshev_pseudospectral",
    "vperp_ngrid" => 5,
    "vperp_nelement" => 8,
    "vperp_L" => 8.0,
    "vperp_bc" => "periodic",
    "vperp_discretization" => "chebyshev_pseudospectral_vperp",
)

"""
    runcase(input::Dict)

Run a simulation with parameters set by `input` using manufactured sources and return
the errors in each variable compared to the manufactured solution.
"""
function runcase(input::Dict)
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    output = load_test_output(input, (:phi, :moments, :f))

    t = output["time"][end]
    n = output["density"][:,:,1,end]
    phi = output["phi"][:,:,1,end]
    f = output["f"][:,:,:,:,1,end]

    n_manf, phi_manf, f_manf = manufactured_solutions_as_arrays(t, output["r"],
                                   output["z"], output["vperp"], output["vpa"])

    n_error_2 = L2_error_norm(n, n_manf)
    n_error_inf = L_infinity_error_norm(n, n_manf)

    phi_error_2 = L2_error_norm(phi, phi_manf)
    phi_error_inf = L_infinity_error_norm(phi, phi_manf)

    f_error_2 = L2_error_norm(f, f_manf)
    f_error_inf = L_infinity_error_norm(f, f_manf)

    println("n ", n_error_2, " ", n_error_inf)
    println("phi ", phi_error_2, " ", phi_error_inf)
    println("f ", f_error_2, " ", f_error_inf)

    return n_error_2, n_error_inf, phi_error_2, phi_error_inf, f_error_2, f_error_inf
end

"""
    increase_resolution(input::Dict, factor)

Increase resolution of simulation by multiplying the numbers of elements `*_nelement` in
the `input` settings by `factor`.
"""
function increase_resolution(input::Dict, factor)
    result = copy(input)
    for key ∈ keys(result)
        if occursin("_nelement", key)
            result[key] *= factor
        end
    end

    return result
end

"""
    test_error_series(errors::Vector{mk_float}, resolution_factors::Vector,
                      expected_order, expected_lowest)

Test whether the error norms in `errors` converge as expected with increases in
resolution by `resolution_factors`. `expected_order` is the order p such that the error
is expected to be proportional to h^p. `expected_lowest` is the expected value of the
error at the lowest resolution (used as a regression test).

Note the entries in `errors` and `resolution_factors` should be sorted in increasing
order of `resolution_factors`.
"""
function test_error_series(errors::Vector{mk_float}, resolution_factors::Vector,
                           expected_order, expected_lowest)
    error_factors = errors[1:end-1] ./ errors[2:end]
    expected_factors = resolution_factors[2:end].^expected_order
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

    for resolution_factor ∈ [1, 2, 4]
        println("testing $(resolution_factor)x")
        case_input = increase_resolution(input, resolution_factor)

        n_error_2, n_error_inf, phi_error_2, phi_error_inf, f_error_2,
        f_error_inf = runcase(case_input)

        push!(n_errors_2, n_error_2)
        push!(n_errors_inf, n_error_inf)
        push!(phi_errors_2, phi_error_2)
        push!(phi_errors_inf, phi_error_inf)
        push!(f_errors_2, f_error_2)
        push!(f_errors_inf, f_error_inf)
    end
end

function runtests()
    @testset "MMS" verbose=use_verbose begin
        println("MMS tests")

        @testset "r-periodic, z-periodic" begin
            testconvergence(input_sound_wave_periodic)
        end
    end

    return nothing
end

end # ManufacturedSolutionsTests


using .ManufacturedSolutionsTests

ManufacturedSolutionsTests.runtests()
