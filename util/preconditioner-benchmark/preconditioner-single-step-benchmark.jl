using moment_kinetics
using moment_kinetics.communication
using moment_kinetics.load_data: get_run_info_no_setup, get_variable
using moment_kinetics.type_definitions

function single_step_benchmark(input, restart)
    if input isa String
        input = moment_kinetics.moment_kinetics_input.read_input_file(input)
    else
        # Ensure we do not modify the passed-in dictionary
        input = deepcopy(input)
    end
    base_directory = get(input["output"], "base_directory", "runs")
    output_dir = joinpath(base_directory, input["output"]["run_name"])

    timestepping = get(input, "timestepping", OptionsDict())
    timestepping["nstep"] = 1
    timestepping["write_after_fixed_step_count"] = true

    input["output"]["display_timing_info"] = false

    nonlinear_solver = get(input, "nonlinear_solver", OptionsDict())
    nonlinear_max_iterations = get(nonlinear_solver, "nonlinear_max_iterations", 20)

    try
        run_moment_kinetics(input; restart=restart)
    catch e
        println("run failed with error", e)
        return -2
    end

    ri = get_run_info_no_setup(output_dir)
    #nonlinear_iterations = get_variable(ri, "electron_advance_nonlinear_iterations")
    linear_iterations = get_variable(ri, "electron_advance_linear_iterations")
    n_solves = get_variable(ri, "electron_advance_n_solves")
    failure_counter = get_variable(ri, "failure_counter")

    if ri.nt > 2
        error("expected only one output timestep, got nt=$(ri.nt)")
    end
    if sum(failure_counter) > 0
        println("timestep failed - probably the iteration on one of the implicit steps failed to converge")
        return -1
    end
    if n_solves[end] != 3
        error("expected 3 implicit solves - assumes using the 'KennedyCarpenterARK324' scheme")
    end

    # Total number of linear iterations measures the total cost of the step.
    return linear_iterations[end]
end
