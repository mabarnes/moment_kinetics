module MMSDebug

# Debug test using wall boundary conditions.

include("setup.jl")

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)


# Input parameters for the test
include("mms_inputs.jl")

# Defines the test functions, using variables defined in the *_inputs.jl file
include("runtest_template.jl")

end # MMSDebug


using .MMSDebug

MMSDebug.runtests()
