# Script to test that get_electron_Jacobian_matrix() function runs. Uses very low
# resolution input, so results are likely to be garbage.

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

include("../util/MKJacobianUtils.jl")

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

input = OptionsDict("output" => OptionsDict("run_name" => "precompilation",
                                            "base_directory" => test_output_directory),
                    "evolve_moments" => OptionsDict("density" => true,
                                                    "parallel_flow" => true,
                                                    "pressure" => true),
                    "composition" => OptionsDict("electron_physics" => "kinetic_electrons"),
                    "r" => OptionsDict("ngrid" => 1,
                                       "nelement" => 1,
                                       "discretization" => "gausslegendre_pseudospectral"),
                    "z" => OptionsDict("ngrid" => 5,
                                       "nelement" => 4,
                                       "bc" => "wall",
                                       "discretization" => "gausslegendre_pseudospectral"),
                    "vperp" => OptionsDict("ngrid" => 1,
                                           "nelement" => 1,
                                           "bc" => "zero",
                                           "L" => 4.0,
                                           "discretization" => "gausslegendre_pseudospectral"),
                    "vpa" => OptionsDict("ngrid" => 7,
                                         "nelement" => 8,
                                         "bc" => "zero",
                                         "L" => 8.0,
                                         "discretization" => "gausslegendre_pseudospectral"),
                    "vzeta" => OptionsDict("ngrid" => 1,
                                           "nelement" => 1,
                                           "bc" => "zero",
                                           "L" => 4.0,
                                           "discretization" => "gausslegendre_pseudospectral"),
                    "vr" => OptionsDict("ngrid" => 1,
                                        "nelement" => 1,
                                        "bc" => "zero",
                                        "L" => 4.0,
                                        "discretization" => "gausslegendre_pseudospectral"),
                    "vz" => OptionsDict("ngrid" => 7,
                                        "nelement" => 8,
                                        "bc" => "zero",
                                        "L" => 8.0,
                                        "discretization" => "gausslegendre_pseudospectral"),
                    "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                                  "nstep" => 1,
                                                  "dt" => 2.0e-11),
                    "electron_timestepping" => OptionsDict("nstep" => 1,
                                                           "dt" => 2.0e-11,
                                                           "initialization_residual_value" => 1.0e10,
                                                           "converged_residual_value" => 1.0e10,
                                                           "rtol" => 1.0e10,
                                                           "no_restart" => true))


run_moment_kinetics(input)

run_directory = joinpath(input["output"]["base_directory"], input["output"]["run_name"])

# Run with all terms
get_electron_Jacobian_matrix(run_directory)

# Run each term individually
get_electron_Jacobian_matrix(run_directory; include_z_advection=true,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=true, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=true,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=true, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=true,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=true,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=true,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=true, include_ion_dt_forcing=false,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=true,
                             include_wall_bc=false)
get_electron_Jacobian_matrix(run_directory; include_z_advection=false,
                             include_vpa_advection=false, include_electron_pdf_term=false,
                             include_dissipation=false, include_krook=false,
                             include_external_source=false,
                             include_constraint_forcing=false,
                             include_energy_equation=false, include_ion_dt_forcing=false,
                             include_wall_bc=true)
nothing
