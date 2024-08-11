# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

input = Dict("run_name" => "precompilation",
             "base_directory" => test_output_directory,
             "evolve_moments_density" => true,
             "evolve_moments_parallel_flow" => true,
             "evolve_moments_parallel_pressure" => true,
             "electron_physics" => "kinetic_electrons",
             "r_ngrid" => 1,
             "r_nelement" => 1,
             "r_bc" => "periodic",
             "r_discretization" => "chebyshev_pseudospectral",
             "z_ngrid" => 5,
             "z_nelement" => 4,
             "z_bc" => "wall",
             "z_discretization" => "chebyshev_pseudospectral",
             "vperp_ngrid" => 1,
             "vperp_nelement" => 1,
             "vperp_bc" => "zero",
             "vperp_L" => 4.0,
             "vperp_discretization" => "chebyshev_pseudospectral",
             "vpa_ngrid" => 7,
             "vpa_nelement" => 8,
             "vpa_bc" => "zero",
             "vpa_L" => 8.0,
             "vpa_discretization" => "chebyshev_pseudospectral",
             "vzeta_ngrid" => 1,
             "vzeta_nelement" => 1,
             "vzeta_bc" => "zero",
             "vzeta_L" => 4.0,
             "vzeta_discretization" => "chebyshev_pseudospectral",
             "vr_ngrid" => 1,
             "vr_nelement" => 1,
             "vr_bc" => "zero",
             "vr_L" => 4.0,
             "vr_discretization" => "chebyshev_pseudospectral",
             "vz_ngrid" => 7,
             "vz_nelement" => 8,
             "vz_bc" => "zero",
             "vz_L" => 8.0,
             "vz_discretization" => "chebyshev_pseudospectral",
             "timestepping" => OptionsDict("nstep" => 1,
                                                "dt" => 2.0e-11),
             "electron_timestepping" => OptionsDict("nstep" => 1,
                                                         "dt" => 2.0e-11,
                                                         "initialization_residual_value" => 1.0e10,
                                                         "converged_residual_value" => 1.0e10,
                                                         "rtol" => 1.0e10,
                                                         "no_restart" => true))


run_moment_kinetics(input)
