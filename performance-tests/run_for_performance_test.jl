# Similar to run_moment_kinetics.jl script, but first runs a single timestep to ensure all
# compilation is complete, then starts a new run.

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

options = moment_kinetics.get_options()
inputfile = options["inputfile"]
restart = options["restart"]
if options["restartfile"] !== nothing
    restart = options["restartfile"]
end
restart_time_index = options["restart-time-index"]

input = moment_kinetics.read_input_file(inputfile)

compilation_run_input = deepcopy(input)
compilation_run_input["timestepping"]["nstep"] = 1

# Run a single timstep.
run_moment_kinetics(compilation_run_input; restart, restart_time_index)

# Overwrite any output from the single-timestep run with a full run.
run_moment_kinetics(input; restart, restart_time_index)
