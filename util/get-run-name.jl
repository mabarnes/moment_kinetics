#!bin/julia --project
# Note the shebang above uses a relative path. This path is relative to the location
# where the script is run from, not the location of the script itself, so this is a bit
# fragile, but should be OK as this script should only ever be called from the top level
# of the moment_kinetics repo

"""
Get run name from an input file
"""

using TOML

inputfile = try
    ARGS[1]
catch BoundsError
    println("Must pass an input file name as the first command line argument")
    exit(1)
end

input = TOML.parsefile(inputfile)

if "run_name" ∈ keys(input)
    run_name = input["run_name"]
else
    # For branch with run name from input file name, should handle that here...
    run_name = basename(splitext(inputfile)[1])
end

println(run_name)
exit(0)
