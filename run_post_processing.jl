if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics as mk

    mk.post_processing.analyze_and_plot_data(ARGS...)
end
