# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics
using moment_kinetics.utils: recursive_merge
using moment_kinetics.type_definitions: OptionsDict

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

base_input = OptionsDict("output" => OptionsDict("run_name" => "precompilation",
                                                 "base_directory" => test_output_directory),
                         "r" => OptionsDict("ngrid" => 5,
                                            "nelement" => 3,
                                            "bc" => "periodic",
                                            "discretization" => "finite_difference"),
                         "z" => OptionsDict("ngrid" => 5,
                                            "nelement" => 3,
                                            "bc" => "periodic",
                                            "discretization" => "finite_difference"),
                         "vperp" => OptionsDict("ngrid" => 5,
                                                "nelement" => 3,
                                                "bc" => "zero",
                                                "L" => 4.0,
                                                "discretization" => "finite_difference"),
                         "vpa" => OptionsDict("ngrid" => 7,
                                              "nelement" => 3,
                                              "bc" => "zero",
                                              "L" => 8.0,
                                              "discretization" => "finite_difference"),
                         "vzeta" => OptionsDict("ngrid" => 5,
                                                "nelement" => 3,
                                                "bc" => "zero",
                                                "L" => 4.0,
                                                "discretization" => "finite_difference"),
                         "vr" => OptionsDict("ngrid" => 5,
                                             "nelement" => 3,
                                             "bc" => "zero",
                                             "L" => 4.0,
                                             "discretization" => "finite_difference"),
                         "vz" => OptionsDict("ngrid" => 7,
                                             "nelement" => 3,
                                             "bc" => "zero",
                                             "L" => 8.0,
                                             "discretization" => "finite_difference"),
                         "timestepping" => OptionsDict("nstep" => 1, "dt" => 2.0e-11))
cheb_input = recursive_merge(base_input, OptionsDict("r" => OptionsDict("discretization" => "chebyshev_pseudospectral"),
                                                     "z" => OptionsDict("discretization" => "chebyshev_pseudospectral"),
                                                     "vperp" => OptionsDict("discretization" => "chebyshev_pseudospectral"),
                                                     "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral")))
wall_bc_input = recursive_merge(base_input, OptionsDict("z" => OptionsDict("bc" => "wall")))
wall_bc_cheb_input = recursive_merge(cheb_input, OptionsDict("z" => OptionsDict("bc" => "wall")))

inputs_list = Vector{OptionsDict}(undef, 0)
for input ∈ [base_input, cheb_input, wall_bc_input, wall_bc_cheb_input]
    push!(inputs_list, input)
    x = recursive_merge(input, OptionsDict("evolve_moments" => OptionsDict("density" => true),
                                           "reactions" => OptionsDict("ionization_frequency" => 0.0),
                                           "r" => OptionsDict("ngrid" => 1, "nelement" => 1),
                                           "vperp" => OptionsDict("ngrid" => 1, "nelement" => 1),
                                           "vzeta" => OptionsDict("ngrid" => 1, "nelement" => 1),
                                           "vr" => OptionsDict("ngrid" => 1, "nelement" => 1)))
    push!(inputs_list, x)
    x = recursive_merge(x, OptionsDict("evolve_moments" => OptionsDict("parallel_flow" => true)))
    push!(inputs_list, x)
    x = recursive_merge(x, OptionsDict("evolve_moments" => OptionsDict("parallel_pressure" => true)))
    push!(inputs_list, x)
end

collisions_input1 = recursive_merge(wall_bc_cheb_input, OptionsDict("composition" => OptionsDict("n_neutral_species" => 0),
                                                                    "krook_collisions" => OptionsDict("use_krook" => true),
                                                                    "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true, "self_collisions" => true, "slowing_down_test" => true),
                                                                    "vperp" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                    "vpa" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                   ))
collisions_input2 = recursive_merge(wall_bc_cheb_input, OptionsDict("composition" => OptionsDict("n_neutral_species" => 0),
                                                                    "krook_collisions" => OptionsDict("use_krook" => true),
                                                                    "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true, "self_collisions" => true, "slowing_down_test" => true),
                                                                    "vperp" => OptionsDict("discretization" => "gausslegendre_pseudospectral",
                                                                                           "bc" => "zero-impose-regularity"),
                                                                    "vpa" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                   ))
collisions_input3 = recursive_merge(wall_bc_cheb_input, OptionsDict("composition" => OptionsDict("n_neutral_species" => 0),
                                                                    "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true, "self_collisions" => true, "boundary_data_option" => "delta_f_multipole"),
                                                                    "vperp" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                    "vpa" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                   ))
collisions_input4 = recursive_merge(wall_bc_cheb_input, OptionsDict("composition" => OptionsDict("n_neutral_species" => 0),
                                                                    "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true, "self_collisions" => true, "boundary_data_option" => "multipole_expansion"),
                                                                    "vperp" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                    "vpa" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                   ))
collisions_input5 = recursive_merge(wall_bc_cheb_input, OptionsDict("composition" => OptionsDict("n_neutral_species" => 0),
                                                                    "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true, "self_collisions" => true, "boundary_data_option" => "multipole_expansion",
                                                                                                              "nonlinear_solver" => OptionsDict("rtol" => 0.0,
                                                                                                                                               "atol" => 1.0e-14)),
                                                                    "timestepping"=> OptionsDict("kinetic_ion_solver" => "implicit_ion_fp_collisions",
                                                                                                 "type" => "PareschiRusso3(4,3,3)",),
                                                                    "vperp" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                    "vpa" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                   ))
# add an additional input for every geometry option available in addition to the default
geo_input1 = recursive_merge(wall_bc_cheb_input, OptionsDict("composition" => OptionsDict("n_neutral_species" => 0),
                                                             "geometry" => OptionsDict("option" => "1D-mirror", "DeltaB" => 0.5, "pitch" => 0.5, "rhostar" => 1.0)))

kinetic_electron_input = recursive_merge(cheb_input, OptionsDict("evolve_moments" => OptionsDict("density" => true,
                                                                                                 "parallel_flow" => true,
                                                                                                 "parallel_pressure" => true),
                                                                 "z" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                 "vpa" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                 "vz" => OptionsDict("discretization" => "gausslegendre_pseudospectral"),
                                                                 "r" => OptionsDict("ngrid" => 1,
                                                                                    "nelement" => 1),
                                                                 "vperp" => OptionsDict("ngrid" => 1,
                                                                                        "nelement" => 1),
                                                                 "vzeta" => OptionsDict("ngrid" => 1,
                                                                                        "nelement" => 1),
                                                                 "vr" => OptionsDict("ngrid" => 1,
                                                                                     "nelement" => 1),
                                                                 "composition" => OptionsDict("electron_physics" => "kinetic_electrons"),
                                                                 "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324"),
                                                                 "electron_timestepping" => OptionsDict("nstep" => 1,
                                                                                                        "dt" => 2.0e-11,
                                                                                                        "initialization_residual_value" => 1.0e10,
                                                                                                        "converged_residual_value" => 1.0e10,
                                                                                                        "rtol" => 1.0e10,
                                                                                                        "no_restart" => true),
                                                                ))

push!(inputs_list, collisions_input1, collisions_input2,
 collisions_input3, collisions_input4, collisions_input5,
 geo_input1, kinetic_electron_input)

for input in inputs_list
    run_moment_kinetics(input)
end
