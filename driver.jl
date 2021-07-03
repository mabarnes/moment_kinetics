using Distributed

if length(ARGS) > 0
    n_procs = parse(Int64, ARGS[1])
    addprocs(n_procs)
end

@everywhere using TimerOutputs

@everywhere using moment_kinetics
using moment_kinetics.moment_kinetics_input: run_type
using moment_kinetics.moment_kinetics_input: RunType, single, performance_test, scan
@everywhere using moment_kinetics.moment_kinetics_input: mk_input
using moment_kinetics.scan_input: mk_scan_inputs

if run_type == single
    to = TimerOutput()
    input = mk_input()
    run_moment_kinetics(to, input)
elseif run_type == performance_test
    to1 = TimerOutput()
    to2 = TimerOutput()

    input = mk_input()
    @timeit to1 "first call to run_moment_kinetics" run_moment_kinetics(to1, input)
    show(to1)
    println()
    @timeit to2 "second call to run_moment_kinetics" run_moment_kinetics(to2, input)
    show(to2)
    println()
elseif run_type == scan
    scan_inputs = mk_scan_inputs()

    @sync @distributed for s ∈ scan_inputs
        println("running parameters: ", s)
        this_input = mk_input(s)
        to = TimerOutput()
        run_moment_kinetics(to, this_input)
    end
else
    error(run_type, " is not a valid run_type option")
end
