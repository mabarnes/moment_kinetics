"""
Parse command line arguments

Have to include test options here too, because ArgParse errors on unrecognized options.
"""
module command_line_options

export options

using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "inputfile"
        help = "Name of TOML input file."
        arg_type = String
        default = nothing
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
# parsing here means options are available at (pre)compile-time
const options = parse_args(s)

function __init__()
    # merging here means options are updated at run-time
    merge!(options, parse_args(s))
end

end
