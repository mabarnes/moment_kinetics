# provide option of running from command line via 'julia run_moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics

if length(ARGS) > 0
    run_moment_kinetics(ARGS[1])
else
    run_moment_kinetics()
end
