if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    using moment_kinetics.makie_post_processing

    makie_post_process(ARGS...)
end
