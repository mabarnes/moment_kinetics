if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    import moment_kinetics
    using moment_kinetics.plot_MMS_sequence
    run_mms_test()
end
