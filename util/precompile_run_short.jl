# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

input_dict = Dict("run_name"=>"precompilation",
                  "base_directory" => test_output_directory,
                  "timestepping" => Dict{String,Any}("nstep" => 1))

to = TimerOutput()
run_moment_kinetics(to, input_dict)
