using Profile
using StatProfilerHTML
using TOML

using moment_kinetics: run_moment_kinetics, options
using moment_kinetics.communication: block_rank

function main(input_file)
    input = TOML.parsefile(input_file)

    test_output_directory = tempname()
    mkpath(test_output_directory)
    input["base_directory"] = test_output_directory

    short_input = deepcopy(input)
    short_input["nstep"] = 2

    # Short run to make sure everything is compiled
    run_moment_kinetics(short_input)

    Profile.clear()
    Profile.@profile run_moment_kinetics(input)

    # Produce html output
    statprofilehtml(path="statprof/profile$(block_rank[])")

    # Print to stdout
    # Use IOContext to increase width so that lines don't get trucated
    block_rank[] == 0 && Profile.print(IOContext(stdout, :displaysize => (24, 500)))

    return nothing
end

# Call main() using first argument as input_file name if running as a script
if abspath(PROGRAM_FILE) == @__FILE__
    if options["inputfile"] == nothing
        error("Must provide input file as positional command line argument")
    end
    main(options["inputfile"])
end
