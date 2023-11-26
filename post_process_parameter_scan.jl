using Pkg
Pkg.activate(".")

using Distributed

@everywhere using moment_kinetics.makie_post_processing: makie_post_process

# get the run_names from the command-line
function post_process_parameter_scan(scan_dir)
    run_directories = Tuple(d for d ∈ readdir(scan_dir, join=true) if isdir(d))
    @sync @distributed for d ∈ run_directories
        println("post-processing ", d)
        try
            makie_post_process(d)
        catch e
            println(d, " failed with ", e)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    post_process_parameter_scan(ARGS[1])
end
