"""
Parse command line arguments

Have to include test options here too, because ArgParse errors on unrecognized options.
"""
module command_line_options

export options

using ArgParse

const s = ArgParseSettings()
@add_arg_table! s begin
    "inputfile"
        help = "Name of TOML input file."
        arg_type = String
        default = nothing
    "restartfile"
        help = "Name of output file (HDF5 or NetCDF) to restart from"
        arg_type = String
        default = nothing
    "--debug", "-d"
        help = "Set debugging level, default is 0 (no extra debugging). Higher " *
               "integer values activate more checks (and increase run time)"
        arg_type = Int
        default = 0
    "--restart"
        help = "Restart from latest output file in run directory (ignored if " *
               "`restartfile` is passed)"
        action = :store_true
    "--restart-time-index"
        help = "Time index in output file to restart from, defaults to final time point"
        arg_type = Int
        default = -1
    # Options for tests
    "--long"
        help = "Include more tests, increasing test run time."
        action = :store_true
    "--verbose", "-v"
        help = "Print verbose output from tests."
        action = :store_true
    # Options for performance tests and plotting
    "--machine-name"
        help = "Which machine to plot for when using plot_performance.jl?"
        arg_type = String
        default = nothing
end

"""
"""
function get_options()
    # Use getter function instead of calling parse_args(s) in __init__() and storing the
    # result in a variable because the __init__() version ignores command line arguments
    # when moment_kinetics is compiled into a static system image using `precompile.jl`.
    return parse_args(s)
end

end
