# provide option of running from command line via 'julia run_moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using TimerOutputs
using TOML
using moment_kinetics

if length(ARGS) > 0
    input_filename = ARGS[1]
    input_dict = TOML.parsefile(input_filename)
else
    input_dict = Dict()
end

to = TimerOutput()
run_moment_kinetics(to, input_dict)
