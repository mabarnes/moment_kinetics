"""
Test cases using the method of manufactured solutions (MMS)
"""
module ManufacturedSolutionsTests

include("setup.jl")

const input_sound_wave_periodic = Dict(
    "use_manufactured_solns" => true,
    "n_ion_species" => 1,
    "n_neutral_species" => 0,
    "boltzmann_electron_response" => true,
    "run_name" => "2D-sound-wave_cheb-vperp-manf",
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
    "nwrite" => 20,
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
    error("not implemented yet!")
end

"""
    increase_resolution(input::Dict, factor)

Increase resolution of simulation by multiplying the numbers of elements `*_nelement` in
the `input` settings by `factor`.
"""
function increase_resolution(input::Dict, factor)
    result = copy(input)
    for key ∈ keys(result)
        if "_nelement" ∈ key
            result[key] *= factor
        end
    end

    return result
end

"""
    testconvergence(input::Dict)

Test convergence with spatial resolution

The parameters for the run are given in `input::Dict`.
"""
function testconvergence(input::Dict)
    errors = []
    for resolution_factor ∈ [1, 2, 4]
        case_input = increase_resolution(input, factor)
        push!(errors, runcase(case_input))
    end
end

function runtests()
    testconvergence(input_sound_wave_periodic)

    return nothing
end

end # ManufacturedSolutionsTests


using .ManufacturedSolutionsTests

ManufacturedSolutionsTests.runtests()
