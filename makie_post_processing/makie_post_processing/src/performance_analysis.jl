"""
    timing_data(run_info; plot_prefix=nothing, threshold=nothing,
                include_patterns=nothing, exclude_patterns=nothing, ranks=nothing,
                figsize=nothing, include_legend=true)

Plot timings from different parts of the `moment_kinetics` code. Only timings from
function calls during the time evolution loop are included, not from the setup, because we
plot versus time.

To reduce clutter, timings whose total time (at the final time point) is less than
`threshold` times the overall run time will be excluded. By default, `threshold` is
`1.0e-3`.

When there is more than one MPI rank present, the timings for each rank will be plotted
separately. The lines will be labelled with the MPI rank, with the position of the labels
moving along the lines one point at a time, to try to avoid overlapping many labels. If
the curves all overlap, this will look like one curve labelled by many MPI ranks.

There are many timers, so it can be useful to filter them to see only the most relevant
ones. By default all timers will be plotted. If `include_patterns` is passed, and
`exclude_patterns` is not, then only the total time and any timers that match
`include_patterns` (matches checked using `occursin()`) will be included in the plots. If
`exclude_patterns` is passed, then any timers that match (matches checked using
`occursin()`) `exclude_patterns` will be omitted, unless they match `include_patterns` in
which case they will still be included. If `ranks` is passed, then only the MPI ranks with
indices found in `ranks` will be included.

`figsize` can be passed to customize the size of the figures that plots are made on. This
can be useful because the legends may become very large when many timers are plotted, in
which case a larger figure might be needed.

`threshold`, `exclude_patterns`, `include_patterns`, `ranks`, and `figsize` can also be
set in `this_input_dict`. When this function is called as part of
[`makie_post_process`](@ref), [`input_dict`](@ref) is passed as `this_input_dict` so that
the settings are read from the post processing input file (by default
`post_processing_input.toml`). The function arguments take precedence, if they are given.

If you load GLMakie by doing `using GLMakie` before running this function, but after
calling `using makie_post_processing` (because `CairoMakie` is loaded when the module is
loaded and would take over if you load `GLMakie` before `makie_post_processing`), the
figures will be displayed in interactive windows. When you hover over a line some useful
information will be displayed.

Pass `include_legend=false` to remove legends from the figures. This is mostly useful for
interactive figures where hovering over the lines can show what they are, so that the
legend is not needed.
"""
function timing_data(run_info::Vector{Any}; plot_prefix=nothing, threshold=nothing,
                     include_patterns=nothing, exclude_patterns=nothing, ranks=nothing,
                     this_input_dict=nothing, figsize=nothing, include_legend=true)

    if this_input_dict !== nothing
        input = Dict_to_NamedTuple(this_input_dict["timing_data"])
    else
        input = nothing
    end

    if input !== nothing && !input.plot
        return nothing
    end

    println("Making timing data plots")

    if figsize === nothing
        if input !== nothing
            figsize = Tuple(input.figsize)
        else
            figsize = (600,800)
        end
    end

    run_time_fig, run_time_ax, run_time_legend_place =
        get_1d_ax(; xlabel="time", ylabel="execution time per output step (minutes)", get_legend_place=:below,
                    size=figsize)
    times_fig, times_ax, times_legend_place =
        get_1d_ax(; xlabel="time", ylabel="execution time per output step (s)", get_legend_place=:below,
                    size=figsize)
    ncalls_fig, ncalls_ax, ncalls_legend_place =
        get_1d_ax(; xlabel="time", ylabel="number of calls per output step", get_legend_place=:below,
                    size=figsize)
    allocs_fig, allocs_ax, allocs_legend_place =
        get_1d_ax(; xlabel="time", ylabel="allocations per output step (MB)", get_legend_place=:below,
                    size=figsize)

    for (irun,ri) ∈ enumerate(run_info)
        timing_data(ri; plot_prefix=plot_prefix, threshold=threshold,
                    include_patterns=include_patterns, exclude_patterns=exclude_patterns,
                    ranks=ranks, this_input_dict=this_input_dict, run_time_ax=run_time_ax,
                    times_ax=times_ax, ncalls_ax=ncalls_ax, allocs_ax=allocs_ax,
                    irun=irun, figsize=figsize)
    end

    if string(Makie.current_backend()) == "GLMakie"
        # Can make interactive plots

        backend = Makie.current_backend()

        if include_legend
            Legend(run_time_fig[2,1], run_time_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(run_time_fig)
        display(backend.Screen(), run_time_fig)

        if include_legend
            Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(times_fig)
        display(backend.Screen(), times_fig)

        if include_legend
            Legend(ncalls_fig[2,1], ncalls_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(ncalls_fig)
        display(backend.Screen(), ncalls_fig)

        if include_legend
            Legend(allocs_fig[2,1], allocs_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(allocs_fig)
        display(backend.Screen(), allocs_fig)
    elseif plot_prefix !== nothing
        if include_legend
            Legend(run_time_fig[2,1], run_time_ax; tellheight=true, tellwidth=true, merge=true)
        end
        outfile = plot_prefix * "run_time.pdf"
        save(outfile, run_time_fig)

        if include_legend
            Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=true, merge=true)
        end
        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(times_fig.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(times_fig)
        outfile = plot_prefix * "execution_times.pdf"
        save(outfile, times_fig)

        if include_legend
            Legend(ncalls_fig[2,1], ncalls_ax; tellheight=true, tellwidth=true, merge=true)
        end
        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(ncalls_fig.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(ncalls_fig)
        outfile = plot_prefix * "ncalls.pdf"
        save(outfile, ncalls_fig)

        if include_legend
            Legend(allocs_fig[2,1], allocs_ax; tellheight=true, tellwidth=true, merge=true)
        end
        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(allocs_fig.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(allocs_fig)
        outfile = plot_prefix * "allocations.pdf"
        save(outfile, allocs_fig)
    end

    return times_fig, ncalls_fig, allocs_fig
end

function timing_data(run_info; plot_prefix=nothing, threshold=nothing,
                     include_patterns=nothing, exclude_patterns=nothing, ranks=nothing,
                     this_input_dict=nothing, run_time_ax=nothing, times_ax=nothing,
                     ncalls_ax=nothing, allocs_ax=nothing, irun=1, figsize=nothing,
                     include_legend=true)

    if this_input_dict !== nothing
        input = Dict_to_NamedTuple(this_input_dict["timing_data"])
    else
        input = nothing
    end

    if input !== nothing && !input.plot
        return nothing
    end

    if figsize === nothing
        if input !== nothing
            figsize = Tuple(input.figsize)
        else
            figsize = (600,800)
        end
    end

    if threshold === nothing
        if input !== nothing
            threshold = input.threshold
        else
            threshold = 1.0e-2
        end
    end
    if isa(include_patterns, AbstractString)
        include_patterns = [include_patterns]
    end
    if isa(exclude_patterns, AbstractString)
        exclude_patterns = [exclude_patterns]
    end
    if input !== nothing && include_patterns === nothing
        include_patterns = input.include_patterns
        if length(include_patterns) == 0
            include_patterns = nothing
        end
    end
    if input !== nothing && exclude_patterns === nothing
        exclude_patterns = input.exclude_patterns
        if length(exclude_patterns) == 0
            exclude_patterns = nothing
        end
    end
    if input !== nothing && ranks === nothing
        ranks = input.ranks
    end

    if run_time_ax === nothing
        run_time_fig, run_time_ax, run_time_legend_place =
            get_1d_ax(; xlabel="time", ylabel="execution time per output step (minutes)",
                      get_legend_place=:below, size=figsize)
    else
        run_time_fig = nothing
    end
    if times_ax === nothing
        times_fig, times_ax, times_legend_place =
            get_1d_ax(; xlabel="time", ylabel="execution time per output step (s)",
                      get_legend_place=:below, size=figsize)
    else
        times_fig = nothing
    end
    if ncalls_ax === nothing
        ncalls_fig, ncalls_ax, ncalls_legend_place =
            get_1d_ax(; xlabel="time", ylabel="number of calls per output step", get_legend_place=:below)
    else
        ncalls_fig = nothing
    end
    if allocs_ax === nothing
        allocs_fig, allocs_ax, allocs_legend_place =
            get_1d_ax(; xlabel="time", ylabel="allocations per output step (MB)", get_legend_place=:below)
    else
        allocs_fig = nothing
    end

    linestyles = linestyle=[:solid, :dash, :dot, :dashdot, :dashdotdot]
    time_advance_timer_variables = [v for v ∈ run_info.timing_variable_names if occursin("time_advance! step", v)]
    time_variables = [v for v ∈ time_advance_timer_variables if startswith(v, "time:")]
    ncalls_variables = [v for v ∈ time_advance_timer_variables if startswith(v, "ncalls:")]
    allocs_variables = [v for v ∈ time_advance_timer_variables if startswith(v, "allocs:")]

    timing_group = "timing_data"

    function label_irank(ax, variable, irank, color, unit_conversion=1)
        if run_info.nrank > 1
            # Label curves with irank so we can tell which is which
            index = ((irank + 1) % (length(variable) - 1)) + 1
            with_theme
            text!(ax, run_info.time[index],
                  variable[index] * unit_conversion;
                  text="$irank", color=color)
        end
    end

    function check_include_exclude(variable_name)
        explicitly_included = (include_patterns !== nothing &&
                               any(occursin(p, variable_name) for p ∈ include_patterns))
        if exclude_patterns === nothing && include_patterns !== nothing
            excluded = !explicitly_included
        elseif exclude_patterns !== nothing
            if !explicitly_included &&
                    any(occursin(p, variable_name) for p ∈ exclude_patterns)
                excluded = true
            else
                excluded = false
            end
        else
            excluded = false
        end
        return excluded, explicitly_included
    end

    # Plot the run time per output step
    total_time_variable_name = "time_for_run"
    run_time = get_variable(run_info, total_time_variable_name * "_per_step",
                            group=timing_group)
    lines!(run_time_ax, run_info.time, run_time;
           label=run_info.run_name)

    # Plot the total time
    time_unit_conversion = 1.0e-9 # ns to s
    total_time_variable_name = "time:moment_kinetics;time_advance! step"
    total_time = get_variable(run_info, total_time_variable_name * "_per_step",
                              group=timing_group)
    for irank ∈ 0:run_info.nrank-1
        label = "time_advance! step"
        irank_slice = total_time[irank+1,:]
        lines!(times_ax, run_info.time, irank_slice .* time_unit_conversion;
               color=:black, linestyle=linestyles[irun], label=label,
               inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
        label_irank(times_ax, irank_slice, irank, :black, time_unit_conversion)
    end
    mean_total_time = mean(total_time)
    for (variable_counter, variable_name) ∈ enumerate(time_variables)
        if variable_name == total_time_variable_name
            # Plotted this already
            continue
        end
        excluded, explicitly_included = check_include_exclude(variable_name)
        if excluded
            continue
        end
        variable = get_variable(run_info, variable_name * "_per_step",
                                group=timing_group)
        if !explicitly_included && mean(variable) < threshold * mean_total_time
            # This variable takes a very small amount of time, so skip.
            continue
        end
        for irank ∈ 0:run_info.nrank-1
            label = split(variable_name, "time_advance! step;")[2]
            irank_slice = variable[irank+1,:]
            l = lines!(times_ax, run_info.time, irank_slice .* time_unit_conversion;
                       color=Cycled(variable_counter), linestyle=linestyles[irun],
                       label=label, inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
            label_irank(times_ax, irank_slice, irank, l.color, time_unit_conversion)
        end
    end

    # Plot the number of calls
    total_ncalls_variable_name = "ncalls:moment_kinetics;time_advance! step"
    total_ncalls = get_variable(run_info, total_ncalls_variable_name * "_per_step",
                              group=timing_group)
    for irank ∈ 0:run_info.nrank-1
        label = "time_advance! step"
        irank_slice = total_ncalls[irank+1,:]
        lines!(ncalls_ax, run_info.time, irank_slice; color=:black,
               linestyle=linestyles[irun], label=label,
               inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
        label_irank(ncalls_ax, irank_slice, irank, :black)
    end
    mean_total_ncalls = mean(total_ncalls)
    for (variable_counter, variable_name) ∈ enumerate(ncalls_variables)
        if variable_name == total_ncalls_variable_name
            # Plotted this already
            continue
        end
        excluded, explicitly_included = check_include_exclude(variable_name)
        if excluded
            continue
        end
        variable = get_variable(run_info, variable_name * "_per_step",
                                group=timing_group)
        if !explicitly_included && mean(variable) < threshold * mean_total_ncalls
            # This variable takes a very small number of calls, so skip.
            continue
        end
        for irank ∈ 0:run_info.nrank-1
            label = split(variable_name, "time_advance! step;")[2]
            irank_slice = variable[irank+1,:]
            l = lines!(ncalls_ax, run_info.time, irank_slice;
                       color=Cycled(variable_counter), linestyle=linestyles[irun],
                       label=label, inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
            label_irank(ncalls_ax, irank_slice, irank, l.color)
        end
    end

    # Plot the total allocs
    allocs_unit_conversion = 2^(-20) # bytes to MB
    total_allocs_variable_name = "allocs:moment_kinetics;time_advance! step"
    total_allocs = get_variable(run_info, total_allocs_variable_name * "_per_step",
                                group=timing_group)
    for irank ∈ 0:run_info.nrank-1
        label = "time_advance! step"
        irank_slice = total_allocs[irank+1,:]
        lines!(allocs_ax, run_info.time, irank_slice .* allocs_unit_conversion;
               color=:black, linestyle=linestyles[irun], label=label,
               inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
        label_irank(allocs_ax, irank_slice, irank, :black, allocs_unit_conversion)
    end
    mean_total_allocs = mean(total_allocs)
    for (variable_counter, variable_name) ∈ enumerate(allocs_variables)
        if variable_name == total_allocs_variable_name
            # Plotted this already
            continue
        end
        excluded, explicitly_included = check_include_exclude(variable_name)
        if excluded
            continue
        end
        variable = get_variable(run_info, variable_name * "_per_step",
                                group=timing_group)
        if !explicitly_included && mean(variable) < threshold * mean_total_allocs
            # This variable represents a very small amount of allocs, so skip.
            continue
        end
        for irank ∈ 0:run_info.nrank-1
            label = split(variable_name, "time_advance! step;")[2]
            irank_slice = variable[irank+1,:]
            l = lines!(allocs_ax, run_info.time, irank_slice .* allocs_unit_conversion;
                       color=Cycled(variable_counter), linestyle=linestyles[irun],
                       label=label, inspector_label=(self,i,p) -> "$(self.label[]) $irank\nx: $(p[1])\ny: $(p[2])")
            label_irank(allocs_ax, irank_slice, irank, l.color, allocs_unit_conversion)
        end
    end

    if times_fig !== nothing && plot_prefix === nothing &&
            string(Makie.current_backend()) == "GLMakie"

        # Can make interactive plots

        backend = Makie.current_backend()

        if include_legend
            Legend(run_time_fig[2,1], run_time_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(run_time_fig)
        display(backend.Screen(), run_time_fig)

        if include_legend
            Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(times_fig)
        display(backend.Screen(), times_fig)

        if include_legend
            Legend(ncalls_fig[2,1], ncalls_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(ncalls_fig)
        display(backend.Screen(), ncalls_fig)

        if include_legend
            Legend(allocs_fig[2,1], allocs_ax; tellheight=true, tellwidth=false,
                   merge=true)
        end
        DataInspector(allocs_fig)
        display(backend.Screen(), allocs_fig)
    else
        if run_time_fig !== nothing
            if include_legend
                Legend(run_time_fig[2,1], run_time_ax; tellheight=true, tellwidth=true, merge=true)
            end
            if plot_prefix !== nothing
                outfile = plot_prefix * "run_time.pdf"
                save(outfile, run_time_fig)
            end
        end

        if times_fig !== nothing
            if include_legend
                Legend(times_fig[2,1], times_ax; tellheight=true, tellwidth=true, merge=true)
            end
            # Ensure the first row width is 3/4 of the column width so that the plot does not
            # get squashed by the legend
            rowsize!(times_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(times_fig)
            if plot_prefix !== nothing
                outfile = plot_prefix * "execution_times.pdf"
                save(outfile, times_fig)
            end
        end

        if ncalls_fig !== nothing
            if include_legend
                Legend(ncalls_fig[2,1], ncalls_ax; tellheight=true, tellwidth=true, merge=true)
            end
            # Ensure the first row width is 3/4 of the column width so that the plot does not
            # get squashed by the legend
            rowsize!(ncalls_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(ncalls_fig)
            if plot_prefix !== nothing
                outfile = plot_prefix * "ncalls.pdf"
                save(outfile, ncalls_fig)
            end
        end

        if allocs_fig !== nothing
            if include_legend
                Legend(allocs_fig[2,1], allocs_ax; tellheight=true, tellwidth=true, merge=true)
            end
            # Ensure the first row width is 3/4 of the column width so that the plot does not
            # get squashed by the legend
            rowsize!(allocs_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(allocs_fig)
            if plot_prefix !== nothing
                outfile = plot_prefix * "allocations.pdf"
                save(outfile, allocs_fig)
            end
        end
    end

    return times_fig, ncalls_fig, allocs_fig
end

"""
    parallel_scaling(run_info; plot_prefix, this_input_dict=nothing,
                     weak=false)

Analyse the parallel scaling of a set of simulations. By default 'strong scaling' (run
time compared to number of processes, for a fixed simulation grid size), if `weak=true` is
passed instead does 'weak scaling' (the simulation grid size is varied in proportion to
the number of processes).

If `efficiency_reference_nproc` is set to a positive number in the input, the efficiency
is calculated relative to the run with this number of processes (which is assumed to be
one of the runs in run_info).

Note that there is no check that the grid size (or number of timesteps, etc.) stays the
same (for strong scaling) or varies with the number of processes (for weak scaling). This
function assumes that the runs input in `run_info` form a well-defined 'strong/weak
scaling' scan - if not then the plots are meaningless.
"""
function parallel_scaling(run_info; plot_prefix=nothing, this_input_dict=nothing,
                          weak=false)
    if !isa(run_info, Vector) || length(run_info) == 1
        # Doesn't make sense to do a strong scaling plot with only one run.
        return nothing
    end

    if this_input_dict !== nothing
        input = Dict_to_NamedTuple(this_input_dict["timing_data"])
    else
        input = Dict_to_NamedTuple(input_dict["timing_data"])
    end

    if input !== nothing && !input.plot_scaling
        return nothing
    end

    if weak
        println("Plotting weak scaling analysis")
    else
        println("Plotting strong scaling analysis")
    end

    timing_group = "timing_data"

    nproc = mk_int[]
    for ri ∈ run_info
        push!(nproc, ri.nrank)
    end

    sorting_indices = sortperm(nproc)
    nproc = nproc[sorting_indices]

    if input.efficiency_reference_nproc < 0
        # Just compare against the first point
        reference_index = 1
    else
        reference_index = findfirst((x)->x==input.efficiency_reference_nproc, nproc)
    end
    reference_nproc = nproc[reference_index]

    function add_to_plots(variable_name, ax_time, ax_efficiency; scatter=false,
                          plot_ideal=false)
        run_time = mk_float[]
        for ri ∈ run_info
            # Use the total time in `ssp_rk!()` as the thing to compare, so that we exclude
            # file I/O time, which may be relatively large in short runs done for parallel
            # scaling timings, but should be insignificant in production runs due to the large
            # number of steps between outputs.
            # Use `it` to select the last time point of each simulation, which gives the total
            # cumulative time spent in `ssp_rk!()`.
            # Convert from nanoseconds to seconds.
            push!(run_time,
                  get_variable(ri, variable_name;
                               group=timing_group, it=ri.nt)[1] / 1.0e9)
        end

        # All variables that this function is called for should have names starting with
        # "time:moment_kinetics;time_advance! step;", so remove this preface from the
        # names we put in the legend.
        variable_label = split(variable_name, "time:moment_kinetics;time_advance! step;")[2]

        run_time = run_time[sorting_indices]

        reference_time = run_time[reference_index]

        if scatter
            scatter!(ax_time, nproc, run_time, label=variable_label,
                     inspector_label=(self,i,p) -> "$(self.label[])\nx: $(p[1])\ny: $(p[2])")
        else
            lines!(ax_time, nproc, run_time, label=variable_label,
                   inspector_label=(self,i,p) -> "$(self.label[])\nx: $(p[1])\ny: $(p[2])")
        end

        if plot_ideal
            # Plot ideal scaling
            if weak
                ideal_scaling = fill(reference_time, length(nproc))
            else
                ideal_scaling = @. reference_time * reference_nproc / nproc
            end
            lines!(ax_time, nproc, ideal_scaling; linestyle=:dash, color=:grey)
        end

        # Make a plot of the efficiency vs. some point in the scan
        if weak
            efficiency = @. reference_time / run_time
        else
            efficiency = @. reference_time / run_time * reference_nproc / nproc
        end

        if scatter
            scatter!(ax_efficiency, nproc, efficiency, label=variable_label,
                     inspector_label=(self,i,p) -> "$(self.label[])\nx: $(p[1])\ny: $(p[2])")
        else
            lines!(ax_efficiency, nproc, efficiency, label=variable_label,
                   inspector_label=(self,i,p) -> "$(self.label[])\nx: $(p[1])\ny: $(p[2])")
        end

        if plot_ideal
            # Plot ideal scaling
            hlines!(ax_efficiency, 1.0; linestyle=:dash, color=:grey)
        end
    end

    fig_time, ax_time = get_1d_ax(xlabel="nproc", ylabel="run time (s)")
    fig_efficiency, ax_efficiency = get_1d_ax(xlabel="nproc", ylabel="efficiency vs. nproc=$reference_nproc")

    add_to_plots("time:moment_kinetics;time_advance! step;ssp_rk!", ax_time,
                 ax_efficiency; scatter=true, plot_ideal=true)

    if length(unique(nproc)) > 1
        # If there is only one nproc, does not make much sense to log-scale it, and trying
        # to would cause an error.
        ax_time.xscale = log2
        ax_efficiency.xscale = log2
    end
    if !weak
        ax_time.yscale = log10
    end

    if input.plot_scaling_all_timers
        fig_time_all, ax_time_all, legend_place_time_all =
            get_1d_ax(xlabel="nproc", ylabel="run time (s)", get_legend_place=:below)
        fig_efficiency_all, ax_efficiency_all, legend_place_efficiency_all =
            get_1d_ax(xlabel="nproc", ylabel="efficiency vs. nproc=$reference_nproc",
                      get_legend_place=:below)

        plot_ideal = true
        for variable_name ∈ run_info[1].timing_variable_names
            if !startswith(variable_name, "time:moment_kinetics;time_advance! step;ssp_rk!")
                # Only plot variables that are time variables and part of the time advance
                continue
            end
            add_to_plots(variable_name, ax_time_all, ax_efficiency_all; plot_ideal)
            # Only plot the 'ideal scaling' for the first variable, to avoid clutter.
            plot_ideal = false
        end

        if length(unique(nproc)) > 1
            # If there is only one nproc, does not make much sense to log-scale it, and
            # trying to would cause an error.
            ax_time_all.xscale = log2
            ax_efficiency_all.xscale = log2
        end
        if !weak
            ax_time_all.yscale = log10
        end

        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(fig_time_all.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(fig_time_all)

        # Ensure the first row width is 3/4 of the column width so that the plot does not
        # get squashed by the legend
        rowsize!(fig_efficiency_all.layout, 1, Aspect(1, 3/4))
        resize_to_layout!(fig_efficiency_all)
    end

    if plot_prefix === nothing && string(Makie.current_backend()) == "GLMakie"
        # Can make interactive plots

        backend = Makie.current_backend()

        DataInspector(fig_time)
        display(backend.Screen(), fig_time)

        DataInspector(fig_efficiency)
        display(backend.Screen(), fig_efficiency)

        if input.plot_scaling_all_timers
            DataInspector(fig_time_all)
            display(backend.Screen(), fig_time_all)

            DataInspector(fig_efficiency_all)
            display(backend.Screen(), fig_efficiency_all)
        end
    else
        if plot_prefix !== nothing
            if weak
                outfile = plot_prefix * "weak_scaling.pdf"
            else
                outfile = plot_prefix * "strong_scaling.pdf"
            end
            save(outfile, fig_time)
        end

        if plot_prefix !== nothing
            if weak
                outfile = plot_prefix * "weak_scaling_efficiency.pdf"
            else
                outfile = plot_prefix * "strong_scaling_efficiency.pdf"
            end
            save(outfile, fig_efficiency)
        end

        if input.plot_scaling_all_timers
            Legend(legend_place_time_all, ax_time_all; tellheight=true, tellwidth=true)
            Legend(legend_place_efficiency_all, ax_efficiency_all; tellheight=true,
                   tellwidth=true)

            if plot_prefix !== nothing
                if weak
                    outfile = plot_prefix * "weak_scaling_all.pdf"
                else
                    outfile = plot_prefix * "strong_scaling_all.pdf"
                end
                save(outfile, fig_time_all)
            end

            if plot_prefix !== nothing
                if weak
                    outfile = plot_prefix * "weak_scaling_efficiency_all.pdf"
                else
                    outfile = plot_prefix * "strong_scaling_efficiency_all.pdf"
                end
                save(outfile, fig_efficiency_all)
            end
        end
    end

    return nothing
end
