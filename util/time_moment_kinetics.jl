using moment_kinetics

"""
    time_moment_kinetics()

Print timing information for the `time_advance!()` call in a moment_kinetics run,
excluding compilation time.
"""
function time_moment_kinetics()
    options = moment_kinetics.get_options()
    inputfile = options["inputfile"]
    restart = options["restart"]
    if options["restartfile"] !== nothing
        restart = options["restartfile"]
    end
    restart_time_index = options["restart-time-index"]
    if inputfile === nothing
        this_input = Dict()
    else
        this_input = moment_kinetics.read_input_file(inputfile)
    end
    short_input = deepcopy(this_input)
    short_input["nstep"] = 1
    if moment_kinetics.global_rank[] == 0
        println("Preliminary, short run to make sure all functions are compiled...")
    end
    run_moment_kinetics(moment_kinetics.TimerOutput(), short_input; restart=restart,
                        restart_time_index=restart_time_index)
    if moment_kinetics.global_rank[] == 0
        println("\nActual run for timing:")
    end
    run_moment_kinetics(moment_kinetics.TimerOutput(), this_input; restart=restart,
                        restart_time_index=restart_time_index)
end

if abspath(PROGRAM_FILE) == @__FILE__
    time_moment_kinetics()
end
