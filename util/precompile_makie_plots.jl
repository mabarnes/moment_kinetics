using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict
using makie_post_processing

# Create a temporary directory for test output
test_output_directory = tempname()
run_name = "precompilation"
mkpath(test_output_directory)

input_dict = OptionsDict("output" => OptionsDict("run_name"=>run_name,
                                                 "base_directory" => test_output_directory),
                         "r" => OptionsDict("ngrid" => 5,
                                            "nelement" => 1,
                                            "discretization" => "chebyshev_pseudospectral"),
                         "z" => OptionsDict("ngrid" => 5,
                                            "nelement" => 1,
                                            "bc" => "wall",
                                            "discretization" => "chebyshev_pseudospectral"),
                         "vperp" => OptionsDict("ngrid" => 5,
                                                "nelement" => 1,
                                                #"bc" => "periodic",
                                                "L" => 4.0,
                                                "discretization" => "chebyshev_pseudospectral"),
                         "vpa" => OptionsDict("ngrid" => 7,
                                              "nelement" => 1,
                                              "bc" => "periodic",
                                              "L" => 4.0,
                                              "discretization" => "chebyshev_pseudospectral"),
                         "vzeta" => OptionsDict("ngrid" => 7,
                                                "nelement" => 1,
                                                "bc" => "periodic",
                                                "L" => 4.0,
                                                "discretization" => "chebyshev_pseudospectral"),
                         "vr" => OptionsDict("ngrid" => 7,
                                             "nelement" => 1,
                                             "bc" => "periodic",
                                             "L" => 4.0,
                                             "discretization" => "chebyshev_pseudospectral"),
                         "vz" => OptionsDict("ngrid" => 7,
                                             "nelement" => 1,
                                             "bc" => "periodic",
                                             "L" => 4.0,
                                             "discretization" => "chebyshev_pseudospectral"),
                         "timestepping" => OptionsDict("nstep" => 1, "dt" => 2.0e-11))

run_moment_kinetics(input_dict)

precompile_postproc_options = makie_post_processing.generate_example_input_Dict()

# Try to activate all plot types to get as much compiled as possible
for (k,v) ∈ precompile_postproc_options
    if v === false
        precompile_postproc_options[k] = true
    end
end

makie_post_process(joinpath(test_output_directory, run_name), precompile_postproc_options)
