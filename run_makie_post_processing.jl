if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk

    mk.makie_post_processing.makie_post_process(ARGS...)
end
