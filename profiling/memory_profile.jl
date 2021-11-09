using Profile
using TimerOutputs
using TOML

using moment_kinetics: run_moment_kinetics, options

function main(input_file)
    input = TOML.parsefile(input_file)

    test_output_directory = tempname()
    mkpath(test_output_directory)
    input["base_directory"] = test_output_directory

    short_input = deepcopy(input)
    short_input["nstep"] = 2

    to = TimerOutput()

    # Short run to make sure everything is compiled
    run_moment_kinetics(to, short_input)

    # Reset memory allocation counters, so we only count the main run
    Profile.clear_malloc_data()

    run_moment_kinetics(to, input)

    return nothing
end

# Call main() using first argument as input_file name if running as a script
if abspath(PROGRAM_FILE) == @__FILE__
    if options["inputfile"] == nothing
        error("Must provide input file as positional command line argument")
    end
    main(options["inputfile"])
end
