# provide option of running from command line via 'julia run_moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics

restart_moment_kinetics()
