module RestartInterpolationDebug

include("setup.jl")

# Create a temporary directory for test output
test_output_directory = get_MPI_tempdir()
mkpath(test_output_directory)

# Input parameters for the test
include("restart_interpolation_inputs.jl")

run_moment_kinetics(base_input)

if moment_kinetics.file_io.io_has_parallel(Val(moment_kinetics.file_io.hdf5))
    base_output_file = realpath(joinpath(base_input["output"]["base_directory"], base_input["output"]["run_name"], string(base_input["output"]["run_name"], ".dfns.h5")))
else
    base_output_file = realpath(joinpath(base_input["output"]["base_directory"], base_input["output"]["run_name"], string(base_input["output"]["run_name"], ".dfns.0.h5")))
end

# Defines the test functions, using variables defined in the *_inputs.jl file
include("runtest_template.jl")

end # RestartInterpolationDebug


using .RestartInterpolationDebug

RestartInterpolationDebug.runtests(restart=RestartInterpolationDebug.base_output_file)
