using Profile
using StatProfilerHTML
using TOML

using moment_kinetics: run_moment_kinetics, setup_moment_kinetics, time_advance!, cleanup_moment_kinetics!, options
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

    ## profile including setup/cleanup calls
    #Profile.clear()
    #Profile.@profile run_moment_kinetics(input)

    # Don't profile initialization
    mk_state = setup_moment_kinetics(input)
    Profile.clear()
    # Do profile time_advance!()
    Profile.@profile time_advance!(mk_state...)

    # Produce html output
    statprofilehtml(path="statprof/profile$(block_rank[])")

    # Print to stdout
    # Use IOContext to increase width so that lines don't get trucated
    block_rank[] == 0 && Profile.print(IOContext(stdout, :displaysize => (24, 500)))

    # Clean up, without profiling
    cleanup_moment_kinetics!(mk_state[end-1:end]...)

    return nothing
end

# Call main() using first argument as input_file name if running as a script
if abspath(PROGRAM_FILE) == @__FILE__
    if options["inputfile"] == nothing
        error("Must provide input file as positional command line argument")
    end
    main(options["inputfile"])
end
