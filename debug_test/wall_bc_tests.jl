module WallBCDebug

# Debug test using wall boundary conditions.

include("setup.jl")

# Create a temporary directory for test output
test_output_directory = get_MPI_tempdir()
mkpath(test_output_directory)


# Input parameters for the test
include("wall_bc_inputs.jl")

# Defines the test functions, using variables defined in the *_inputs.jl file
include("runtest_template.jl")

end # WallBCDebug


using .WallBCDebug

WallBCDebug.runtests()
