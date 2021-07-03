using Distributed

try
    n_procs = parse(Int64, ARGS[1])
    addprocs(n_procs)
    @everywhere path_start_ind = 2
catch ArgumentError
    # Default to serial if a number was not given as the first argument
    @everywhere path_start_ind = 1
end

@everywhere using moment_kinetics
@everywhere using moment_kinetics.post_processing: analyze_and_plot_data

# get the run_names from the command-line
@sync @distributed for path ∈ ARGS[path_start_ind:end]
    println("post-processing ", path)
    try
        analyze_and_plot_data(path)
    catch e
        println(path, " failed with ", e)
    end
    println()
end
