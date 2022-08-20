module HarrisonThompsonDebug
# Test case with constant-in-space, delta-function-in-vpa source against
# analytic solution from [E. R. Harrison and W. B. Thompson. The low pressure
# plane symmetric discharge. Proc. Phys. Soc., 74:145, 1959]

include("setup.jl")

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# Input parameters for the test
include("harrisonthompson_inputs.jl")

# Defines the test functions, using variables defined in the *_inputs.jl file
include("runtest_template.jl")

end # HarrisonThompsonDebug


using .HarrisonThompsonDebug

HarrisonThompsonDebug.runtests()
