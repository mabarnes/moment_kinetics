"""
    compare_runs(run_info)

Plot/animate the differences between several runs, comparing each run to the first one in
`run_info`. Where different grids are used for (some of) the runs, all results will be
interpolated onto the grid of the first run.
"""
function compare_runs(run_info, run_info_dfns=run_info; plot_prefix, has_rdim=true,
                      has_zdim=true, is_1V=false, has_electrons=true, has_neutrals=true,
                      has_dfns=true)
    if isa(run_info, NamedTuple)
        # Single 'run_info' was passed. Nothing to compare.
        return nothing
    elseif length(run_info) ≤ 1
        # Single 'run_info' was passed. Nothing to compare.
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["compare_runs"])

    if !input.enable
        return nothing
    end

    println("Making comparison plots:")

    base_run = run_info[1]
    other_runs = run_info[2:end]

    difference_plot_prefix = plot_prefix * "difference_"
    plot_title = "difference from " * base_run.run_name

    for variable_name ∈ all_moment_variables
        if variable_name ∈ all_source_variables
            continue
        end
        plots_for_variable(other_runs, variable_name; plot_prefix=difference_plot_prefix,
                           has_rdim=has_rdim, has_zdim=has_zdim, is_1V=is_1V,
                           subtract_from_info=base_run,
                           interpolate_to_other_grid=input.interpolate_to_other_grid,
                           axis_args=(title=plot_title,))
    end

    # Plots from distribution function variables
    ############################################
    if has_dfns
        base_run_dfns = run_info_dfns[1]
        other_runs_dfns = run_info_dfns[2:end]

        dfn_variable_list = ion_dfn_variables
        if has_electrons
            dfn_variable_list = tuple(dfn_variable_list..., electron_dfn_variables...)
        end
        if has_neutrals
            dfn_variable_list = tuple(dfn_variable_list..., neutral_dfn_variables...)
        end
        for variable_name ∈ dfn_variable_list
            plots_for_dfn_variable(other_runs_dfns, variable_name;
                                   plot_prefix=difference_plot_prefix, has_rdim=has_rdim,
                                   has_zdim=has_zdim, is_1V=is_1V,
                                   subtract_from_info=base_run_dfns,
                                   interpolate_to_other_grid=input.interpolate_to_other_grid,
                                   axis_args=(title=plot_title,))
        end
    end

    return nothing
end
