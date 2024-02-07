if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    import plots_post_processing
    using plots_post_processing.plot_MMS_sequence
    run_mms_test()
end
