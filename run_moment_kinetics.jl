# provide option of running from command line via 'julia run_moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using TimerOutputs
using moment_kinetics

to = TimerOutput()

if length(ARGS) > 0
    run_moment_kinetics(to, ARGS[1])
else
    run_moment_kinetics(to)
end
