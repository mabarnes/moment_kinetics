# provide option of running from command line via 'julia run_moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using MPI
MPI.Init()
is_rank0 = (MPI.Comm_rank(MPI.COMM_WORLD) == 0)

# Use MPI Barriers to ensure `using moment_kinetics` is completed on rank-0 (in order to
# complete precompilation of moment_kinetics and dependencies) before being run on all the
# other MPI ranks.
if is_rank0
    using moment_kinetics
    MPI.Barrier(MPI.COMM_WORLD)
else
    MPI.Barrier(MPI.COMM_WORLD)
    using moment_kinetics
end

run_moment_kinetics()
