using Profile
using StatProfilerHTML
using TimerOutputs
using TOML

using moment_kinetics

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

    Profile.clear()
    @profilehtml run_moment_kinetics(to, input)
    Profile.print()

    return nothing
end

# Call main() using first argument as input_file name if running as a script
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        error("Must provide input file as first command line argument")
    end
    main(ARGS[1])
end
