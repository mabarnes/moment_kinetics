# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

input_dict = OptionsDict("output" => OptionsDict("run_name"=>"precompilation",
                                                 "base_directory" => test_output_directory),
                         "timestepping" => OptionsDict("nstep" => 1, "dt" => 2.0e-11))

run_moment_kinetics(input_dict)
