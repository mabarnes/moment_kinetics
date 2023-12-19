"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    @testset "$name" begin
        # Provide some progress info
        println("    - bug-checking ", name)

        # Convert dict from symbol keys to String keys
        modified_inputs = Dict(String(k) => v for (k, v) in args)

        # Update default inputs with values to be changed
        input = merge(test_input, modified_inputs)

        input["run_name"] = name

        # run simulation
        run_moment_kinetics(input)
    end
end

function runtests()
    @testset "$test_type" begin
        println("$test_type tests")

        for input ∈ test_input_list
            run_test(input)
        end
    end
end
